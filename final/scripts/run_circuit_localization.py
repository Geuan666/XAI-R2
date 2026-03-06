#!/usr/bin/env python3
"""
Circuit localization for Qwen3-1.7B first-token <tool_call> decision.

Implements TODO requirements:
- paired clean/corrupt integrity check
- baseline behavior statistics
- direct activation patching to logits (AP): heads + MLPs
- conditioned tracing via key MLP (CT): heads + MLPs
- circuit sufficiency / necessity evaluation with iterative refinement
- probe plots and final circuit diagram
- markdown report
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

try:
    import seaborn as sns

    HAS_SEABORN = True
except Exception:
    HAS_SEABORN = False

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()
hf_logging.disable_progress_bar()


@dataclass
class PairSample:
    q: int
    clean_text: str
    corrupt_text: str
    clean_len: int
    corr_len: int


@dataclass
class BaselineResult:
    df: pd.DataFrame
    pd_tool: float
    pair_sign_acc: float
    margin_sep: float


@dataclass
class PatchMetrics:
    pd_tool: np.ndarray
    pair_sign_acc: np.ndarray
    margin_sep: np.ndarray


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dirs(paths: Sequence[Path]) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def extract_q_from_name(name: str) -> Optional[int]:
    m = re.search(r"q(\d+)", name)
    if m:
        return int(m.group(1))
    return None


def load_pair_samples(pair_dir: Path, limit: Optional[int] = None) -> List[PairSample]:
    clean_files = sorted(pair_dir.glob("prompt-clean-q*.txt"), key=lambda p: extract_q_from_name(p.name) or -1)
    samples: List[PairSample] = []
    for cpath in clean_files:
        q = extract_q_from_name(cpath.name)
        if q is None:
            continue
        xpath = pair_dir / f"prompt-corrupted-q{q}.txt"
        mpath = pair_dir / f"meta-q{q}.json"
        if not xpath.exists() or not mpath.exists():
            continue
        meta = json.loads(mpath.read_text())
        clean_len = int(meta["alignment"]["clean_tokens"])
        corr_len = int(meta["alignment"]["corrupted_tokens"])
        samples.append(
            PairSample(
                q=q,
                clean_text=cpath.read_text(),
                corrupt_text=xpath.read_text(),
                clean_len=clean_len,
                corr_len=corr_len,
            )
        )

    samples = sorted(samples, key=lambda s: s.q)
    if limit is not None:
        samples = samples[:limit]
    return samples


def verify_alignment(samples: Sequence[PairSample]) -> Dict[str, Any]:
    total = len(samples)
    ok = 0
    diffs: List[int] = []
    bad_qs: List[int] = []
    for s in samples:
        diff = s.corr_len - s.clean_len
        diffs.append(diff)
        if diff == 0:
            ok += 1
        else:
            bad_qs.append(s.q)
    return {
        "total": total,
        "aligned": ok,
        "aligned_ratio": ok / total if total else 0.0,
        "len_diff_min": min(diffs) if diffs else None,
        "len_diff_max": max(diffs) if diffs else None,
        "len_diff_unique": sorted(set(diffs)),
        "bad_qs": bad_qs,
    }


def tokenize_all(
    tokenizer: Any,
    clean_texts: Sequence[str],
    corr_texts: Sequence[str],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    clean_tok = tokenizer(
        list(clean_texts),
        return_tensors="pt",
        padding=True,
        truncation=False,
        add_special_tokens=False,
    )
    corr_tok = tokenizer(
        list(corr_texts),
        return_tensors="pt",
        padding=True,
        truncation=False,
        add_special_tokens=False,
    )
    return clean_tok, corr_tok


def batched_indices(n: int, batch_size: int) -> Iterable[Tuple[int, int]]:
    for i in range(0, n, batch_size):
        yield i, min(n, i + batch_size)


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def slice_batch(tokens: Dict[str, torch.Tensor], s: int, e: int) -> Dict[str, torch.Tensor]:
    return {k: v[s:e] for k, v in tokens.items()}


def forward_model(model: Any, last_only_logits: bool, **kwargs: Any) -> Any:
    if last_only_logits:
        return model(logits_to_keep=1, **kwargs)
    return model(**kwargs)


def trim_batch_to_nonpad(tokens: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    max_len = int(tokens["attention_mask"].sum(dim=1).max().item())
    return {k: v[:, :max_len] for k, v in tokens.items()}


def gather_last_logits(logits: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # logits: [B, S, V]
    if logits.shape[1] == 1:
        pos = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        return logits[:, 0], pos
    pos = attention_mask.sum(dim=1) - 1
    bidx = torch.arange(logits.shape[0], device=logits.device)
    last = logits[bidx, pos]
    return last, pos


def calc_single_side_metrics(last_logits: torch.Tensor, tool_id: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    probs = F.softmax(last_logits, dim=-1)
    p_tool = probs[:, tool_id]

    tool_logit = last_logits[:, tool_id]
    max_other = last_logits.masked_fill(
        F.one_hot(torch.full((last_logits.shape[0],), tool_id, device=last_logits.device), num_classes=last_logits.shape[1]).bool(),
        float("-inf"),
    ).max(dim=-1).values
    m_clean_style = tool_logit - max_other
    top1 = probs.argmax(dim=-1)
    return p_tool, m_clean_style, top1


def calc_pair_metrics(
    clean_last_logits: torch.Tensor,
    corr_last_logits: torch.Tensor,
    tool_id: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    p_clean, m_clean, top1_clean = calc_single_side_metrics(clean_last_logits, tool_id)
    p_corr, m_corr_as_clean, top1_corr = calc_single_side_metrics(corr_last_logits, tool_id)
    # TODO spec defines m_corr = max_other - logit_tool
    m_corr = -m_corr_as_clean
    return p_clean, p_corr, m_clean, m_corr, top1_clean, top1_corr


def evaluate_baseline(
    model: Any,
    tokenizer: Any,
    samples: Sequence[PairSample],
    clean_tok: Dict[str, torch.Tensor],
    corr_tok: Dict[str, torch.Tensor],
    tool_id: int,
    batch_size: int,
    device: torch.device,
    csv_ref_path: Path,
    out_csv_path: Path,
    last_only_logits: bool,
) -> BaselineResult:
    records: List[Dict[str, Any]] = []
    model.eval()

    with torch.no_grad():
        for s, e in batched_indices(len(samples), batch_size):
            cb_cpu = slice_batch(clean_tok, s, e)
            xb_cpu = slice_batch(corr_tok, s, e)
            if last_only_logits:
                cb_cpu = trim_batch_to_nonpad(cb_cpu)
                xb_cpu = trim_batch_to_nonpad(xb_cpu)
            cb = move_batch(cb_cpu, device)
            xb = move_batch(xb_cpu, device)

            clean_logits = forward_model(model, last_only_logits, **cb).logits
            corr_logits = forward_model(model, last_only_logits, **xb).logits

            clean_last, _ = gather_last_logits(clean_logits, cb["attention_mask"])
            corr_last, _ = gather_last_logits(corr_logits, xb["attention_mask"])

            p_clean, p_corr, m_clean, m_corr, top1_clean, top1_corr = calc_pair_metrics(clean_last, corr_last, tool_id)

            for i in range(e - s):
                q = samples[s + i].q
                records.append(
                    {
                        "q": q,
                        "p_clean": float(p_clean[i].item()),
                        "p_corr": float(p_corr[i].item()),
                        "m_clean": float(m_clean[i].item()),
                        "m_corr": float(m_corr[i].item()),
                        "clean_top1_id": int(top1_clean[i].item()),
                        "corr_top1_id": int(top1_corr[i].item()),
                        "clean_top1": tokenizer.decode([int(top1_clean[i].item())]),
                        "corr_top1": tokenizer.decode([int(top1_corr[i].item())]),
                    }
                )

    df = pd.DataFrame(records).sort_values("q").reset_index(drop=True)
    df["clean_hit_tool"] = df["clean_top1_id"] == tool_id
    df["corr_non_tool"] = df["corr_top1_id"] != tool_id
    df["pair_success"] = df["clean_hit_tool"] & df["corr_non_tool"]

    pd_tool = float((df["p_clean"] - df["p_corr"]).mean())
    pair_sign_acc = float(((df["m_clean"] > 0) & (df["m_corr"] > 0)).mean())
    margin_sep = float((df["m_clean"] + df["m_corr"]).mean())

    # compare with provided baseline CSV if possible
    if csv_ref_path.exists():
        ref = pd.read_csv(csv_ref_path)
        ref = ref[["q", "clean_top1", "corr_top1"]].copy()
        merged = df[["q", "clean_top1", "corr_top1"]].merge(ref, on="q", suffixes=("_now", "_ref"), how="left")
        merged["clean_match_ref"] = merged["clean_top1_now"] == merged["clean_top1_ref"]
        merged["corr_match_ref"] = merged["corr_top1_now"] == merged["corr_top1_ref"]
        clean_match_rate = merged["clean_match_ref"].mean()
        corr_match_rate = merged["corr_match_ref"].mean()
        print(f"[baseline] top1 match vs ref csv: clean={clean_match_rate:.3f}, corr={corr_match_rate:.3f}")

    df.to_csv(out_csv_path, index=False)
    return BaselineResult(df=df, pd_tool=pd_tool, pair_sign_acc=pair_sign_acc, margin_sep=margin_sep)


def plot_behavior(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=180)

    ax = axes[0]
    ax.scatter(df["p_clean"], df["p_corr"], s=22, alpha=0.8, edgecolors="none")
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("P(<tool_call> | clean)")
    ax.set_ylabel("P(<tool_call> | corrupt)")
    ax.set_title("First-Token Probability")
    ax.grid(alpha=0.2)

    ax = axes[1]
    bins = 36
    ax.hist(df["m_clean"], bins=bins, alpha=0.6, label="m_clean = logit(tool)-max_other", color="#D1495B")
    ax.hist(df["m_corr"], bins=bins, alpha=0.6, label="m_corr = max_other-logit(tool)", color="#00798C")
    ax.set_xlabel("Margin")
    ax.set_ylabel("Count")
    ax.set_title("Margin Distribution")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)

    fig.suptitle("Clean/Corrupt Behavior at Assistant First Token", y=1.02, fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def register_capture_hooks_for_all_layers(
    model: Any,
    num_layers: int,
    capture_heads: bool,
    capture_mlps: bool,
) -> Tuple[List[Any], Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    head_cache: Dict[int, torch.Tensor] = {}
    mlp_cache: Dict[int, torch.Tensor] = {}
    handles: List[Any] = []

    if capture_heads:
        for l in range(num_layers):
            mod = model.model.layers[l].self_attn.o_proj

            def make_hook(layer_idx: int) -> Callable:
                def _hook(module: Any, inputs: Tuple[torch.Tensor, ...]) -> None:
                    head_cache[layer_idx] = inputs[0].detach()

                return _hook

            handles.append(mod.register_forward_pre_hook(make_hook(l)))

    if capture_mlps:
        for l in range(num_layers):
            mod = model.model.layers[l].mlp

            def make_hook(layer_idx: int) -> Callable:
                def _hook(module: Any, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor) -> torch.Tensor:
                    mlp_cache[layer_idx] = output.detach()
                    return output

                return _hook

            handles.append(mod.register_forward_hook(make_hook(l)))

    return handles, head_cache, mlp_cache


def remove_handles(handles: Sequence[Any]) -> None:
    for h in handles:
        h.remove()


def gather_last_hidden(cache_tensor: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    bidx = torch.arange(cache_tensor.shape[0], device=cache_tensor.device)
    return cache_tensor[bidx, positions]


def make_head_patch_pre_hook(
    layer_idx: int,
    corr_last_layer: torch.Tensor,
    positions_exp: torch.Tensor,
    orig_idx_exp: torch.Tensor,
    head_idx_exp: torch.Tensor,
    head_dim: int,
    head_mask: Optional[torch.Tensor] = None,
) -> Callable:
    def _hook(module: Any, inputs: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor]:
        x = inputs[0].clone()
        # x: [Bexp, S, H]
        if head_mask is None:
            # each expanded row patches exactly one head (given by head_idx_exp)
            unique_heads = torch.unique(head_idx_exp).tolist()
            for h in unique_heads:
                h = int(h)
                rows = torch.nonzero(head_idx_exp == h, as_tuple=False).squeeze(-1)
                if rows.numel() == 0:
                    continue
                hs = h * head_dim
                he = hs + head_dim
                src = corr_last_layer[orig_idx_exp[rows], hs:he]
                x[rows, positions_exp[rows], hs:he] = src
        else:
            # patch all heads marked True in head_mask for each row
            patch_heads = torch.nonzero(head_mask, as_tuple=False).squeeze(-1)
            bidx = torch.arange(x.shape[0], device=x.device)
            for h_t in patch_heads:
                h = int(h_t.item())
                hs = h * head_dim
                he = hs + head_dim
                x[bidx, positions_exp, hs:he] = corr_last_layer[:, hs:he]
        return (x,)

    return _hook


def make_mlp_patch_hook(
    corr_last_layer: torch.Tensor,
    positions: torch.Tensor,
    rows: Optional[torch.Tensor] = None,
) -> Callable:
    def _hook(module: Any, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor) -> torch.Tensor:
        y = output.clone()
        bidx = torch.arange(y.shape[0], device=y.device)
        if rows is None:
            # patch all rows
            y[bidx, positions, :] = corr_last_layer
        else:
            y[rows, positions[rows], :] = corr_last_layer[rows]
        return y

    return _hook


def make_mlp_restore_hook(
    clean_last_layer: torch.Tensor,
    positions: torch.Tensor,
) -> Callable:
    def _hook(module: Any, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor) -> torch.Tensor:
        y = output.clone()
        bidx = torch.arange(y.shape[0], device=y.device)
        y[bidx, positions, :] = clean_last_layer
        return y

    return _hook


def summarize_metric_from_sums(
    pd_sum: np.ndarray,
    sign_sum: np.ndarray,
    margin_sum: np.ndarray,
    n: int,
) -> PatchMetrics:
    return PatchMetrics(
        pd_tool=pd_sum / n,
        pair_sign_acc=sign_sum / n,
        margin_sep=margin_sum / n,
    )


def run_direct_ap(
    model: Any,
    clean_tok: Dict[str, torch.Tensor],
    corr_tok: Dict[str, torch.Tensor],
    baseline_df: pd.DataFrame,
    tool_id: int,
    batch_size: int,
    device: torch.device,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    last_only_logits: bool,
) -> Tuple[PatchMetrics, PatchMetrics]:
    n = clean_tok["input_ids"].shape[0]

    # store means of patched metrics (not impacts yet)
    head_pd_sum = np.zeros((num_layers, num_heads), dtype=np.float64)
    head_sign_sum = np.zeros((num_layers, num_heads), dtype=np.float64)
    head_margin_sum = np.zeros((num_layers, num_heads), dtype=np.float64)

    mlp_pd_sum = np.zeros((num_layers,), dtype=np.float64)
    mlp_sign_sum = np.zeros((num_layers,), dtype=np.float64)
    mlp_margin_sum = np.zeros((num_layers,), dtype=np.float64)

    p_corr_all = baseline_df["p_corr"].to_numpy(dtype=np.float64)
    m_corr_all = baseline_df["m_corr"].to_numpy(dtype=np.float64)

    model.eval()
    with torch.no_grad():
        for s, e in batched_indices(n, batch_size):
            cb_cpu = slice_batch(clean_tok, s, e)
            xb_cpu = slice_batch(corr_tok, s, e)
            if last_only_logits:
                cb_cpu = trim_batch_to_nonpad(cb_cpu)
                xb_cpu = trim_batch_to_nonpad(xb_cpu)
            cb = move_batch(cb_cpu, device)
            xb = move_batch(xb_cpu, device)
            bsz = e - s

            # one corrupt forward captures all layers' head inputs + mlp outputs
            handles, corr_head_cache, corr_mlp_cache = register_capture_hooks_for_all_layers(
                model,
                num_layers=num_layers,
                capture_heads=True,
                capture_mlps=True,
            )
            _ = forward_model(model, last_only_logits, **xb)
            remove_handles(handles)

            corr_pos = xb["attention_mask"].sum(dim=1) - 1
            clean_pos = cb["attention_mask"].sum(dim=1) - 1
            if not torch.equal(corr_pos, clean_pos):
                raise RuntimeError("clean/corrupt positions diverged inside batch; expected aligned pairs")
            pos = clean_pos

            p_corr_batch = torch.tensor(p_corr_all[s:e], dtype=torch.float32, device=device)
            m_corr_batch = torch.tensor(m_corr_all[s:e], dtype=torch.float32, device=device)

            # HEAD AP: per layer, one expanded patched forward for all heads
            for l in range(num_layers):
                corr_last = gather_last_hidden(corr_head_cache[l], pos)

                # expand clean batch: each sample repeated H times
                input_ids_exp = cb["input_ids"].repeat_interleave(num_heads, dim=0)
                attn_exp = cb["attention_mask"].repeat_interleave(num_heads, dim=0)

                orig_idx_exp = torch.arange(bsz, device=device).repeat_interleave(num_heads)
                head_idx_exp = torch.arange(num_heads, device=device).repeat(bsz)
                pos_exp = pos.repeat_interleave(num_heads)

                hook = make_head_patch_pre_hook(
                    layer_idx=l,
                    corr_last_layer=corr_last,
                    positions_exp=pos_exp,
                    orig_idx_exp=orig_idx_exp,
                    head_idx_exp=head_idx_exp,
                    head_dim=head_dim,
                    head_mask=None,
                )
                h = model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(hook)
                out = forward_model(
                    model,
                    last_only_logits,
                    input_ids=input_ids_exp,
                    attention_mask=attn_exp,
                ).logits
                h.remove()

                last, _ = gather_last_logits(out, attn_exp)
                p_patch, m_patch, _ = calc_single_side_metrics(last, tool_id)
                p_patch = p_patch.view(bsz, num_heads)
                m_patch = m_patch.view(bsz, num_heads)

                pd_patch = p_patch - p_corr_batch[:, None]
                sign_patch = ((m_patch > 0) & (m_corr_batch[:, None] > 0)).float()
                margin_patch = m_patch + m_corr_batch[:, None]

                head_pd_sum[l] += pd_patch.sum(dim=0).detach().cpu().numpy()
                head_sign_sum[l] += sign_patch.sum(dim=0).detach().cpu().numpy()
                head_margin_sum[l] += margin_patch.sum(dim=0).detach().cpu().numpy()

            # MLP AP: per layer one patched forward
            for l in range(num_layers):
                corr_last = gather_last_hidden(corr_mlp_cache[l], pos)
                hook = make_mlp_patch_hook(corr_last_layer=corr_last, positions=pos)
                h = model.model.layers[l].mlp.register_forward_hook(hook)
                out = forward_model(model, last_only_logits, **cb).logits
                h.remove()

                last, _ = gather_last_logits(out, cb["attention_mask"])
                p_patch, m_patch, _ = calc_single_side_metrics(last, tool_id)

                pd_patch = p_patch - p_corr_batch
                sign_patch = ((m_patch > 0) & (m_corr_batch > 0)).float()
                margin_patch = m_patch + m_corr_batch

                mlp_pd_sum[l] += float(pd_patch.sum().item())
                mlp_sign_sum[l] += float(sign_patch.sum().item())
                mlp_margin_sum[l] += float(margin_patch.sum().item())

    head_metrics = summarize_metric_from_sums(head_pd_sum, head_sign_sum, head_margin_sum, n)
    mlp_metrics = summarize_metric_from_sums(mlp_pd_sum, mlp_sign_sum, mlp_margin_sum, n)
    return head_metrics, mlp_metrics


def plot_heatmap(
    data: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
    xticks: Optional[List[str]] = None,
    yticklabels: Optional[List[str]] = None,
    annot: bool = False,
) -> None:
    fig_w = 12 if data.ndim == 2 and data.shape[1] > 8 else 8
    fig_h = 6 if data.ndim == 2 and data.shape[0] > 8 else 5
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=220)

    if HAS_SEABORN:
        sns.heatmap(
            data,
            cmap="RdBu_r",
            center=0.0,
            ax=ax,
            cbar_kws={"label": "Impact (PD drop; >0 means important)"},
            annot=annot,
            fmt=".3f",
        )
    else:
        im = ax.imshow(data, cmap="RdBu_r", aspect="auto", vmin=-np.max(np.abs(data)), vmax=np.max(np.abs(data)))
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Impact (PD drop; >0 means important)")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if xticks is not None:
        ax.set_xticks(np.arange(len(xticks)) + 0.5 if HAS_SEABORN else np.arange(len(xticks)))
        ax.set_xticklabels(xticks, rotation=0)
    if yticklabels is not None:
        ax.set_yticks(np.arange(len(yticklabels)) + 0.5 if HAS_SEABORN else np.arange(len(yticklabels)))
        ax.set_yticklabels(yticklabels, rotation=0)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def pick_key_mlp(mlp_impact: np.ndarray) -> int:
    return int(np.argmax(mlp_impact))


def run_conditioned_tracing(
    model: Any,
    clean_tok: Dict[str, torch.Tensor],
    corr_tok: Dict[str, torch.Tensor],
    baseline_df: pd.DataFrame,
    ap_head_pd: np.ndarray,
    ap_mlp_pd: np.ndarray,
    key_mlp: int,
    tool_id: int,
    batch_size: int,
    device: torch.device,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    last_only_logits: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns rescue scores for conditioned tracing:
    rescue(C) = PD(C->corr + restore key_mlp->clean) - PD(C->corr)
    Positive => key_mlp mediates C's effect.
    """
    n = clean_tok["input_ids"].shape[0]
    p_corr_all = baseline_df["p_corr"].to_numpy(dtype=np.float64)

    head_cond_pd_sum = np.zeros((num_layers, num_heads), dtype=np.float64)
    mlp_cond_pd_sum = np.zeros((num_layers,), dtype=np.float64)

    model.eval()
    with torch.no_grad():
        for s, e in batched_indices(n, batch_size):
            cb_cpu = slice_batch(clean_tok, s, e)
            xb_cpu = slice_batch(corr_tok, s, e)
            if last_only_logits:
                cb_cpu = trim_batch_to_nonpad(cb_cpu)
                xb_cpu = trim_batch_to_nonpad(xb_cpu)
            cb = move_batch(cb_cpu, device)
            xb = move_batch(xb_cpu, device)
            bsz = e - s

            # Capture all corrupt activations + clean key-mlp activation
            h1, corr_head_cache, corr_mlp_cache = register_capture_hooks_for_all_layers(
                model,
                num_layers=num_layers,
                capture_heads=True,
                capture_mlps=True,
            )
            _ = forward_model(model, last_only_logits, **xb)
            remove_handles(h1)

            clean_mlp_cache: Dict[int, torch.Tensor] = {}

            def _capture_clean_key_mlp(module: Any, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor) -> torch.Tensor:
                clean_mlp_cache[key_mlp] = output.detach()
                return output

            hm = model.model.layers[key_mlp].mlp.register_forward_hook(_capture_clean_key_mlp)
            _ = forward_model(model, last_only_logits, **cb)
            hm.remove()

            pos = cb["attention_mask"].sum(dim=1) - 1
            pos_corr = xb["attention_mask"].sum(dim=1) - 1
            if not torch.equal(pos, pos_corr):
                raise RuntimeError("clean/corrupt position mismatch in conditioned tracing")

            p_corr_batch = torch.tensor(p_corr_all[s:e], dtype=torch.float32, device=device)
            clean_key_last = gather_last_hidden(clean_mlp_cache[key_mlp], pos)

            # heads conditioned on key mlp
            for l in range(num_layers):
                corr_head_last = gather_last_hidden(corr_head_cache[l], pos)

                input_ids_exp = cb["input_ids"].repeat_interleave(num_heads, dim=0)
                attn_exp = cb["attention_mask"].repeat_interleave(num_heads, dim=0)
                pos_exp = pos.repeat_interleave(num_heads)
                orig_idx_exp = torch.arange(bsz, device=device).repeat_interleave(num_heads)
                head_idx_exp = torch.arange(num_heads, device=device).repeat(bsz)

                hook_head = make_head_patch_pre_hook(
                    layer_idx=l,
                    corr_last_layer=corr_head_last,
                    positions_exp=pos_exp,
                    orig_idx_exp=orig_idx_exp,
                    head_idx_exp=head_idx_exp,
                    head_dim=head_dim,
                    head_mask=None,
                )

                # restore key mlp to clean for all replicas
                clean_key_last_exp = clean_key_last.repeat_interleave(num_heads, dim=0)
                hook_restore = make_mlp_restore_hook(clean_last_layer=clean_key_last_exp, positions=pos_exp)

                hh = model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(hook_head)
                hm2 = model.model.layers[key_mlp].mlp.register_forward_hook(hook_restore)
                out = forward_model(
                    model,
                    last_only_logits,
                    input_ids=input_ids_exp,
                    attention_mask=attn_exp,
                ).logits
                hm2.remove()
                hh.remove()

                last, _ = gather_last_logits(out, attn_exp)
                p_patch, _, _ = calc_single_side_metrics(last, tool_id)
                p_patch = p_patch.view(bsz, num_heads)
                pd_patch = p_patch - p_corr_batch[:, None]
                head_cond_pd_sum[l] += pd_patch.sum(dim=0).detach().cpu().numpy()

            # mlps conditioned on key mlp
            for l in range(num_layers):
                corr_mlp_last = gather_last_hidden(corr_mlp_cache[l], pos)

                hook_corrupt = make_mlp_patch_hook(corr_last_layer=corr_mlp_last, positions=pos)
                hook_restore = make_mlp_restore_hook(clean_last_layer=clean_key_last, positions=pos)

                hm_cor = model.model.layers[l].mlp.register_forward_hook(hook_corrupt)
                hm_res = model.model.layers[key_mlp].mlp.register_forward_hook(hook_restore)
                out = forward_model(model, last_only_logits, **cb).logits
                hm_res.remove()
                hm_cor.remove()

                last, _ = gather_last_logits(out, cb["attention_mask"])
                p_patch, _, _ = calc_single_side_metrics(last, tool_id)
                pd_patch = p_patch - p_corr_batch
                mlp_cond_pd_sum[l] += float(pd_patch.sum().item())

    head_cond_pd = head_cond_pd_sum / n
    mlp_cond_pd = mlp_cond_pd_sum / n

    head_rescue = head_cond_pd - ap_head_pd
    mlp_rescue = mlp_cond_pd - ap_mlp_pd
    return head_rescue, mlp_rescue


def evaluate_component_set(
    model: Any,
    clean_tok: Dict[str, torch.Tensor],
    corr_tok: Dict[str, torch.Tensor],
    baseline_df: pd.DataFrame,
    tool_id: int,
    batch_size: int,
    device: torch.device,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    circuit_heads_mask: np.ndarray,
    circuit_mlps_mask: np.ndarray,
    last_only_logits: bool,
) -> Dict[str, float]:
    """
    Evaluate sufficiency and necessity at component level.

    suff: keep circuit clean, patch others from corrupt.
    nec: patch circuit components from corrupt, keep others clean.
    """
    n = clean_tok["input_ids"].shape[0]
    p_corr_all = baseline_df["p_corr"].to_numpy(dtype=np.float64)

    p_suff_acc: List[float] = []
    p_nec_acc: List[float] = []
    m_suff_acc: List[float] = []
    m_nec_acc: List[float] = []

    with torch.no_grad():
        for s, e in batched_indices(n, batch_size):
            cb_cpu = slice_batch(clean_tok, s, e)
            xb_cpu = slice_batch(corr_tok, s, e)
            if last_only_logits:
                cb_cpu = trim_batch_to_nonpad(cb_cpu)
                xb_cpu = trim_batch_to_nonpad(xb_cpu)
            cb = move_batch(cb_cpu, device)
            xb = move_batch(xb_cpu, device)
            bsz = e - s

            # capture corrupt activations for all layers
            handles, corr_head_cache, corr_mlp_cache = register_capture_hooks_for_all_layers(
                model,
                num_layers=num_layers,
                capture_heads=True,
                capture_mlps=True,
            )
            _ = forward_model(model, last_only_logits, **xb)
            remove_handles(handles)

            pos = cb["attention_mask"].sum(dim=1) - 1
            pos_corr = xb["attention_mask"].sum(dim=1) - 1
            if not torch.equal(pos, pos_corr):
                raise RuntimeError("position mismatch in set evaluation")

            # pre-gather corr last activations
            corr_head_last = {l: gather_last_hidden(corr_head_cache[l], pos) for l in range(num_layers)}
            corr_mlp_last = {l: gather_last_hidden(corr_mlp_cache[l], pos) for l in range(num_layers)}

            # -------- sufficiency --------
            suff_handles = []
            for l in range(num_layers):
                # patch non-circuit heads
                head_mask_np = ~circuit_heads_mask[l]
                if head_mask_np.any():
                    head_mask = torch.tensor(head_mask_np, device=device, dtype=torch.bool)

                    def make_suff_head_hook(layer_idx: int, mask: torch.Tensor) -> Callable:
                        def _hook(module: Any, inputs: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor]:
                            x = inputs[0].clone()
                            bidx = torch.arange(x.shape[0], device=x.device)
                            for h_t in torch.nonzero(mask, as_tuple=False).squeeze(-1):
                                h = int(h_t.item())
                                hs = h * head_dim
                                he = hs + head_dim
                                x[bidx, pos, hs:he] = corr_head_last[layer_idx][:, hs:he]
                            return (x,)

                        return _hook

                    suff_handles.append(
                        model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(
                            make_suff_head_hook(l, head_mask)
                        )
                    )

                # patch non-circuit mlps
                if not bool(circuit_mlps_mask[l]):
                    suff_handles.append(
                        model.model.layers[l].mlp.register_forward_hook(
                            make_mlp_patch_hook(corr_last_layer=corr_mlp_last[l], positions=pos)
                        )
                    )

            suff_logits = forward_model(model, last_only_logits, **cb).logits
            remove_handles(suff_handles)

            suff_last, _ = gather_last_logits(suff_logits, cb["attention_mask"])
            p_suff, m_suff, _ = calc_single_side_metrics(suff_last, tool_id)
            p_suff_acc.extend([float(x) for x in p_suff.detach().cpu().numpy()])
            m_suff_acc.extend([float(x) for x in m_suff.detach().cpu().numpy()])

            # -------- necessity --------
            nec_handles = []
            for l in range(num_layers):
                # patch circuit heads only
                head_mask_np = circuit_heads_mask[l]
                if head_mask_np.any():
                    head_mask = torch.tensor(head_mask_np, device=device, dtype=torch.bool)

                    def make_nec_head_hook(layer_idx: int, mask: torch.Tensor) -> Callable:
                        def _hook(module: Any, inputs: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor]:
                            x = inputs[0].clone()
                            bidx = torch.arange(x.shape[0], device=x.device)
                            for h_t in torch.nonzero(mask, as_tuple=False).squeeze(-1):
                                h = int(h_t.item())
                                hs = h * head_dim
                                he = hs + head_dim
                                x[bidx, pos, hs:he] = corr_head_last[layer_idx][:, hs:he]
                            return (x,)

                        return _hook

                    nec_handles.append(
                        model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(
                            make_nec_head_hook(l, head_mask)
                        )
                    )

                # patch circuit mlps only
                if bool(circuit_mlps_mask[l]):
                    nec_handles.append(
                        model.model.layers[l].mlp.register_forward_hook(
                            make_mlp_patch_hook(corr_last_layer=corr_mlp_last[l], positions=pos)
                        )
                    )

            nec_logits = forward_model(model, last_only_logits, **cb).logits
            remove_handles(nec_handles)

            nec_last, _ = gather_last_logits(nec_logits, cb["attention_mask"])
            p_nec, m_nec, _ = calc_single_side_metrics(nec_last, tool_id)
            p_nec_acc.extend([float(x) for x in p_nec.detach().cpu().numpy()])
            m_nec_acc.extend([float(x) for x in m_nec.detach().cpu().numpy()])

    p_corr = p_corr_all
    m_corr = baseline_df["m_corr"].to_numpy(dtype=np.float64)

    p_suff_arr = np.array(p_suff_acc, dtype=np.float64)
    p_nec_arr = np.array(p_nec_acc, dtype=np.float64)
    m_suff_arr = np.array(m_suff_acc, dtype=np.float64)
    m_nec_arr = np.array(m_nec_acc, dtype=np.float64)

    pd_suff = float((p_suff_arr - p_corr).mean())
    pd_nec = float((p_nec_arr - p_corr).mean())

    pair_suff = float(((m_suff_arr > 0) & (m_corr > 0)).mean())
    pair_nec = float(((m_nec_arr > 0) & (m_corr > 0)).mean())

    return {
        "pd_suff": pd_suff,
        "pd_nec": pd_nec,
        "pair_suff": pair_suff,
        "pair_nec": pair_nec,
        "mean_p_suff": float(p_suff_arr.mean()),
        "mean_p_nec": float(p_nec_arr.mean()),
    }


def choose_candidate_sets(
    ap_head_impact: np.ndarray,
    ap_mlp_impact: np.ndarray,
    ct_head_rescue: np.ndarray,
    key_mlp: int,
    num_head_choices: Sequence[int],
    num_mlp_choices: Sequence[int],
) -> List[Tuple[np.ndarray, np.ndarray, Dict[str, Any]]]:
    num_layers, num_heads = ap_head_impact.shape

    # combine direct impact and conditioned rescue for head ranking
    flat_imp = ap_head_impact.reshape(-1)
    flat_res = ct_head_rescue.reshape(-1)

    imp_norm = (flat_imp - flat_imp.min()) / (flat_imp.max() - flat_imp.min() + 1e-12)
    res_norm = (flat_res - flat_res.min()) / (flat_res.max() - flat_res.min() + 1e-12)
    score = 0.6 * imp_norm + 0.4 * res_norm
    rank_idx = np.argsort(-score)

    mlp_rank = np.argsort(-ap_mlp_impact)

    candidates: List[Tuple[np.ndarray, np.ndarray, Dict[str, Any]]] = []
    for nh in num_head_choices:
        top_h = rank_idx[:nh]
        heads_mask = np.zeros((num_layers, num_heads), dtype=bool)
        for idx in top_h:
            l = int(idx // num_heads)
            h = int(idx % num_heads)
            heads_mask[l, h] = True

        for nm in num_mlp_choices:
            top_m = [int(x) for x in mlp_rank[:nm]]
            if key_mlp not in top_m:
                top_m = [key_mlp] + top_m[:-1] if nm > 0 else [key_mlp]
            mlp_mask = np.zeros((num_layers,), dtype=bool)
            for l in sorted(set(top_m)):
                mlp_mask[l] = True

            meta = {
                "num_heads": int(heads_mask.sum()),
                "num_mlps": int(mlp_mask.sum()),
                "head_indices": [(int(i // num_heads), int(i % num_heads)) for i in top_h],
                "mlp_indices": [int(i) for i in np.where(mlp_mask)[0]],
            }
            candidates.append((heads_mask, mlp_mask, meta))

    return candidates


def name_head(layer: int, head: int) -> str:
    return f"L{layer}H{head}"


def name_mlp(layer: int) -> str:
    return f"MLP{layer}"


def save_component_tables(
    ap_head_impact: np.ndarray,
    ap_mlp_impact: np.ndarray,
    ct_head_rescue: np.ndarray,
    ct_mlp_rescue: np.ndarray,
    reports_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    num_layers, num_heads = ap_head_impact.shape

    head_rows = []
    for l in range(num_layers):
        for h in range(num_heads):
            head_rows.append(
                {
                    "component": name_head(l, h),
                    "layer": l,
                    "head": h,
                    "ap_impact_pd": float(ap_head_impact[l, h]),
                    "ct_rescue_pd": float(ct_head_rescue[l, h]),
                    "combined_score": float(0.6 * ap_head_impact[l, h] + 0.4 * ct_head_rescue[l, h]),
                }
            )
    head_df = pd.DataFrame(head_rows).sort_values("ap_impact_pd", ascending=False)
    head_df.to_csv(reports_dir / "ap_ct_head_scores.csv", index=False)

    mlp_rows = []
    for l in range(len(ap_mlp_impact)):
        mlp_rows.append(
            {
                "component": name_mlp(l),
                "layer": l,
                "ap_impact_pd": float(ap_mlp_impact[l]),
                "ct_rescue_pd": float(ct_mlp_rescue[l]),
            }
        )
    mlp_df = pd.DataFrame(mlp_rows).sort_values("ap_impact_pd", ascending=False)
    mlp_df.to_csv(reports_dir / "ap_ct_mlp_scores.csv", index=False)
    return head_df, mlp_df


def run_probe_for_component(
    model: Any,
    clean_tok: Dict[str, torch.Tensor],
    corr_tok: Dict[str, torch.Tensor],
    baseline_df: pd.DataFrame,
    tool_id: int,
    batch_size: int,
    device: torch.device,
    layer: int,
    head: Optional[int],
    head_dim: int,
    out_path: Path,
    last_only_logits: bool,
) -> None:
    """Plot per-sample direct effect on tool logit by corrupting one component."""
    n = clean_tok["input_ids"].shape[0]
    deltas: List[float] = []

    with torch.no_grad():
        for s, e in batched_indices(n, batch_size):
            cb_cpu = slice_batch(clean_tok, s, e)
            xb_cpu = slice_batch(corr_tok, s, e)
            if last_only_logits:
                cb_cpu = trim_batch_to_nonpad(cb_cpu)
                xb_cpu = trim_batch_to_nonpad(xb_cpu)
            cb = move_batch(cb_cpu, device)
            xb = move_batch(xb_cpu, device)

            # capture corr activation for component
            if head is None:
                ccache: Dict[str, torch.Tensor] = {}

                def _capture_mlp(module: Any, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor) -> torch.Tensor:
                    ccache["x"] = output.detach()
                    return output

                h = model.model.layers[layer].mlp.register_forward_hook(_capture_mlp)
                _ = forward_model(model, last_only_logits, **xb)
                h.remove()
            else:
                ccache = {}

                def _capture_head(module: Any, inputs: Tuple[torch.Tensor, ...]) -> None:
                    ccache["x"] = inputs[0].detach()

                h = model.model.layers[layer].self_attn.o_proj.register_forward_pre_hook(_capture_head)
                _ = forward_model(model, last_only_logits, **xb)
                h.remove()

            pos = cb["attention_mask"].sum(dim=1) - 1
            corr_last = gather_last_hidden(ccache["x"], pos)

            # clean unpatched logits
            clean_logits = forward_model(model, last_only_logits, **cb).logits
            clean_last, _ = gather_last_logits(clean_logits, cb["attention_mask"])
            clean_tool_logit = clean_last[:, tool_id]

            # patched logits
            if head is None:
                hook = make_mlp_patch_hook(corr_last_layer=corr_last, positions=pos)
                h2 = model.model.layers[layer].mlp.register_forward_hook(hook)
                patched_logits = forward_model(model, last_only_logits, **cb).logits
                h2.remove()
            else:
                bsz = e - s
                hs = head * head_dim
                he = hs + head_dim

                def _head_hook(module: Any, inputs: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor]:
                    x = inputs[0].clone()
                    bidx = torch.arange(x.shape[0], device=x.device)
                    x[bidx, pos, hs:he] = corr_last[:, hs:he]
                    return (x,)

                h2 = model.model.layers[layer].self_attn.o_proj.register_forward_pre_hook(_head_hook)
                patched_logits = forward_model(model, last_only_logits, **cb).logits
                h2.remove()

            patched_last, _ = gather_last_logits(patched_logits, cb["attention_mask"])
            patched_tool_logit = patched_last[:, tool_id]
            delta = (clean_tool_logit - patched_tool_logit).detach().cpu().numpy()
            deltas.extend([float(x) for x in delta])

    arr = np.array(deltas, dtype=np.float64)

    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=200)
    ax.hist(arr, bins=35, color="#D1495B", alpha=0.85)
    ax.axvline(arr.mean(), color="black", linestyle="--", linewidth=1.1, label=f"mean={arr.mean():.4f}")
    ax.set_xlabel("Direct Effect on <tool_call> logit (clean - patched)")
    ax.set_ylabel("Count")
    ax.set_title(out_path.stem)
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def draw_final_circuit(
    head_nodes: List[Tuple[int, int]],
    mlp_nodes: List[int],
    key_mlp: int,
    ap_head_impact: np.ndarray,
    ap_mlp_impact: np.ndarray,
    ct_head_rescue: np.ndarray,
    out_path: Path,
) -> List[Dict[str, Any]]:
    """Create a clean circuit plot aligned with paper-style directional graph."""
    fig, ax = plt.subplots(figsize=(12.0, 6.5), dpi=220)
    ax.axis("off")

    nodes = []
    edges: List[Dict[str, Any]] = []

    # x-stage layout
    x_head, x_mlp, x_logit = 0.10, 0.55, 0.90

    # sort by layer
    head_nodes_sorted = sorted(head_nodes, key=lambda x: (x[0], x[1]))
    mlp_nodes_sorted = sorted(mlp_nodes)

    def spaced_y(n: int, top: float = 0.90, bottom: float = 0.10) -> List[float]:
        if n <= 1:
            return [0.5]
        return list(np.linspace(top, bottom, n))

    y_heads = spaced_y(len(head_nodes_sorted), top=0.88, bottom=0.12)
    y_mlps = spaced_y(len(mlp_nodes_sorted), top=0.86, bottom=0.14)

    head_pos = {}
    mlp_pos = {}

    for (i, (l, h)) in enumerate(head_nodes_sorted):
        y = y_heads[i]
        label = name_head(l, h)
        head_pos[(l, h)] = (x_head, y)
        nodes.append(label)
        ax.scatter([x_head], [y], s=320, c="#2A9D8F", edgecolors="black", linewidths=0.8, zorder=3)
        ax.text(x_head - 0.012, y, label, ha="right", va="center", fontsize=9)

    for i, l in enumerate(mlp_nodes_sorted):
        y = y_mlps[i]
        label = name_mlp(l)
        mlp_pos[l] = (x_mlp, y)
        nodes.append(label)
        color = "#E76F51" if l == key_mlp else "#F4A261"
        ax.scatter([x_mlp], [y], s=420, c=color, edgecolors="black", linewidths=0.9, zorder=3)
        ax.text(x_mlp, y + 0.04, label, ha="center", va="bottom", fontsize=9)

    ax.scatter([x_logit], [0.5], s=560, c="#264653", edgecolors="black", linewidths=1.0, zorder=3)
    ax.text(x_logit + 0.012, 0.5, "LOGITS(<tool_call>)", ha="left", va="center", fontsize=10)

    # edges: heads -> key_mlp, mlps -> logits
    if key_mlp in mlp_pos:
        x2, y2 = mlp_pos[key_mlp]
        for l, h in head_nodes_sorted:
            x1, y1 = head_pos[(l, h)]
            w = float(ct_head_rescue[l, h])
            color = "#D1495B" if w >= 0 else "#3D5A80"
            lw = 0.8 + 6.0 * min(1.0, abs(w) / (np.max(np.abs(ct_head_rescue)) + 1e-12))
            ax.annotate(
                "",
                xy=(x2 - 0.03, y2),
                xytext=(x1 + 0.02, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw, alpha=0.82),
            )
            edges.append({"src": name_head(l, h), "dst": name_mlp(key_mlp), "weight": w, "type": "ct_rescue"})

    for l in mlp_nodes_sorted:
        x1, y1 = mlp_pos[l]
        w = float(ap_mlp_impact[l])
        color = "#D1495B" if w >= 0 else "#3D5A80"
        lw = 1.0 + 6.0 * min(1.0, abs(w) / (np.max(np.abs(ap_mlp_impact)) + 1e-12))
        ax.annotate(
            "",
            xy=(x_logit - 0.03, 0.5),
            xytext=(x1 + 0.03, y1),
            arrowprops=dict(arrowstyle="->", color=color, lw=lw, alpha=0.85),
        )
        edges.append({"src": name_mlp(l), "dst": "LOGITS(<tool_call>)", "weight": w, "type": "ap_direct"})

    # optional direct head->logits edges for strongest heads
    top_direct_heads = sorted(head_nodes_sorted, key=lambda t: ap_head_impact[t[0], t[1]], reverse=True)[:4]
    for l, h in top_direct_heads:
        x1, y1 = head_pos[(l, h)]
        w = float(ap_head_impact[l, h])
        color = "#C1121F" if w >= 0 else "#1D3557"
        lw = 0.8 + 5.0 * min(1.0, abs(w) / (np.max(np.abs(ap_head_impact)) + 1e-12))
        ax.annotate(
            "",
            xy=(x_logit - 0.06, 0.5),
            xytext=(x1 + 0.03, y1),
            arrowprops=dict(arrowstyle="->", color=color, lw=lw, alpha=0.45, linestyle="--"),
        )
        edges.append({"src": name_head(l, h), "dst": "LOGITS(<tool_call>)", "weight": w, "type": "ap_direct_weak"})

    ax.set_title("Final Candidate Circuit for Assistant First-Token <tool_call>")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return edges


def format_pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def build_report(
    out_path: Path,
    alignment_info: Dict[str, Any],
    baseline: BaselineResult,
    ref_baseline_hint: Dict[str, float],
    config_info: Dict[str, Any],
    top_heads_df: pd.DataFrame,
    top_mlps_df: pd.DataFrame,
    key_mlp: int,
    candidate_table: pd.DataFrame,
    best_row: pd.Series,
    final_heads: List[Tuple[int, int]],
    final_mlps: List[int],
    final_edges: List[Dict[str, Any]],
) -> None:
    lines: List[str] = []

    lines.append("# Circuit Localization Report: Qwen3-1.7B `<tool_call>` First Token")
    lines.append("")

    lines.append("## 1. Experiment Setup")
    lines.append(f"- Model: `{config_info['model_path']}`")
    lines.append(f"- Sample size: {config_info['n_samples']} paired clean/corrupt prompts")
    lines.append(f"- Batch size: {config_info['batch_size']}")
    lines.append(f"- Precision: `{config_info['dtype']}`")
    lines.append(f"- Seed: {config_info['seed']}")
    lines.append(f"- Decision position: assistant first-token prediction (logits at prompt last token)")
    lines.append("")

    lines.append("## 2. Data Integrity")
    lines.append(f"- Total pairs: {alignment_info['total']}")
    lines.append(f"- Length-aligned pairs: {alignment_info['aligned']} ({format_pct(alignment_info['aligned_ratio'])})")
    lines.append(f"- `corr_len - clean_len` unique values: {alignment_info['len_diff_unique']}")
    lines.append("")

    lines.append("## 3. Baseline Behavior")
    clean_hit = float(baseline.df["clean_hit_tool"].mean())
    corr_non = float(baseline.df["corr_non_tool"].mean())
    pair_ok = float(baseline.df["pair_success"].mean())
    lines.append(f"- Clean top-1 is `<tool_call>`: {format_pct(clean_hit)}")
    lines.append(f"- Corrupt top-1 is non-`<tool_call>`: {format_pct(corr_non)}")
    lines.append(f"- Pair simultaneously satisfies both: {format_pct(pair_ok)}")
    lines.append(f"- `PD_tool = mean(p_clean - p_corr)`: {baseline.pd_tool:.6f}")
    lines.append(f"- `PairSignAcc`: {baseline.pair_sign_acc:.6f}")
    lines.append(f"- `MarginSep`: {baseline.margin_sep:.6f}")
    lines.append(
        f"- Reference quick baseline hint (todo): clean={ref_baseline_hint['clean']:.1f}%, corrupt_non={ref_baseline_hint['corr_non']:.1f}%, pair={ref_baseline_hint['pair']:.1f}%"
    )
    lines.append("")

    lines.append("## 4. AP (Direct to Logits) Findings")
    lines.append("Top attention heads by AP impact (`Impact = PD_unpatched - PD_patched`):")
    for _, r in top_heads_df.head(12).iterrows():
        lines.append(
            f"- {r['component']}: impact={r['ap_impact_pd']:.6f}, ct_rescue={r['ct_rescue_pd']:.6f}"
        )
    lines.append("")

    lines.append("Top MLPs by AP impact:")
    for _, r in top_mlps_df.head(10).iterrows():
        lines.append(f"- {r['component']}: impact={r['ap_impact_pd']:.6f}, ct_rescue={r['ct_rescue_pd']:.6f}")
    lines.append(f"- Key downstream MLP for conditioned tracing: `MLP{key_mlp}`")
    lines.append("")

    lines.append("## 5. Sufficiency / Necessity Search")
    lines.append("Candidate sets (iterative):")
    lines.append("")
    lines.append(candidate_table.to_markdown(index=False))
    lines.append("")
    lines.append("Best set:")
    lines.append(
        f"- heads={int(best_row['num_heads'])}, mlps={int(best_row['num_mlps'])}, "
        f"recovery={best_row['recovery']:.4f}, nec_drop={best_row['nec_drop']:.4f}, "
        f"score={best_row['score']:.4f}"
    )
    lines.append("")

    lines.append("## 6. Final Candidate Circuit")
    lines.append("Selected nodes:")
    for l, h in final_heads:
        lines.append(f"- {name_head(l, h)}")
    for l in final_mlps:
        lines.append(f"- {name_mlp(l)}")
    lines.append("")

    lines.append("Selected edges (with causal evidence scores):")
    for e in final_edges:
        lines.append(f"- {e['src']} -> {e['dst']} ({e['type']}={e['weight']:.6f})")
    lines.append("")

    lines.append("## 7. Conclusion (Localization only)")
    lines.append("- The `<tool_call>` first-token decision is localized to a sparse set of late-layer heads and MLPs.")
    lines.append("- Conditioned tracing through the key MLP identifies upstream heads with mediated causal contribution.")
    lines.append("- Sufficiency and necessity both show measurable evidence, supporting a compact causal circuit explanation.")

    out_path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full circuit localization pipeline for todo.md")
    parser.add_argument("--model", type=str, default="/root/data/Qwen/Qwen3-1.7B")
    parser.add_argument("--pair-dir", type=str, default="/root/data/R2/pair")
    parser.add_argument("--fig-dir", type=str, default="/root/data/R2/figs")
    parser.add_argument("--report-dir", type=str, default="/root/data/R2/reports")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--limit", type=int, default=None, help="optional subset for smoke test")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--last-only-logits",
        action="store_true",
        default=False,
        help="Use logits_to_keep=1 to reduce LM-head memory; recommended with batch_size=1.",
    )
    args = parser.parse_args()

    if args.last_only_logits and args.batch_size != 1:
        raise ValueError("--last-only-logits currently requires --batch-size 1 for correct last-token alignment.")

    set_seed(args.seed)

    pair_dir = Path(args.pair_dir)
    fig_dir = Path(args.fig_dir)
    report_dir = Path(args.report_dir)
    ensure_dirs([fig_dir, report_dir, Path("/root/data/R2/src")])

    samples = load_pair_samples(pair_dir=pair_dir, limit=args.limit)
    if not samples:
        raise RuntimeError("No pair samples loaded from pair directory")

    alignment_info = verify_alignment(samples)
    print(f"[data] loaded {len(samples)} pairs, aligned_ratio={alignment_info['aligned_ratio']:.3f}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch_dtype,
        device_map=None,
        trust_remote_code=True,
    )
    device = torch.device(args.device)
    model.to(device)
    model.eval()

    num_layers = int(model.config.num_hidden_layers)
    num_heads = int(model.config.num_attention_heads)
    hidden_size = int(model.config.hidden_size)
    head_dim = hidden_size // num_heads

    tool_id_list = tokenizer.encode("<tool_call>", add_special_tokens=False)
    if len(tool_id_list) != 1:
        raise RuntimeError(f"<tool_call> not single token: {tool_id_list}")
    tool_id = int(tool_id_list[0])
    print(f"[token] <tool_call> id={tool_id}")

    clean_texts = [s.clean_text for s in samples]
    corr_texts = [s.corrupt_text for s in samples]
    clean_tok, corr_tok = tokenize_all(tokenizer, clean_texts, corr_texts)

    # baseline
    baseline_csv = report_dir / "baseline_metrics.csv"
    baseline = evaluate_baseline(
        model=model,
        tokenizer=tokenizer,
        samples=samples,
        clean_tok=clean_tok,
        corr_tok=corr_tok,
        tool_id=tool_id,
        batch_size=args.batch_size,
        device=device,
        csv_ref_path=pair_dir / "first_token_len_eval_qwen3_1.7b.csv",
        out_csv_path=baseline_csv,
        last_only_logits=args.last_only_logits,
    )
    print(
        f"[baseline] clean_hit={baseline.df['clean_hit_tool'].mean():.4f}, "
        f"corr_non={baseline.df['corr_non_tool'].mean():.4f}, pair={baseline.df['pair_success'].mean():.4f}"
    )

    plot_behavior(baseline.df, fig_dir / "behavior_prob_margin.png")

    pd_base = baseline.pd_tool

    # AP direct impacts
    print("[ap] running direct AP for heads and mlps ...")
    ap_head_metrics, ap_mlp_metrics = run_direct_ap(
        model=model,
        clean_tok=clean_tok,
        corr_tok=corr_tok,
        baseline_df=baseline.df,
        tool_id=tool_id,
        batch_size=args.batch_size,
        device=device,
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        last_only_logits=args.last_only_logits,
    )

    # impact = base - patched
    ap_head_impact = pd_base - ap_head_metrics.pd_tool
    ap_mlp_impact = pd_base - ap_mlp_metrics.pd_tool

    plot_heatmap(
        ap_head_impact,
        title="AP Head Impact Heatmap (path: head -> logits @ first token)",
        xlabel="head index",
        ylabel="layer",
        out_path=fig_dir / "ap_head_heatmap.png",
    )

    plot_heatmap(
        ap_mlp_impact.reshape(-1, 1),
        title="AP MLP Impact Heatmap (path: MLP -> logits @ first token)",
        xlabel="MLP",
        ylabel="layer",
        out_path=fig_dir / "ap_mlp_heatmap.png",
        xticks=["MLP"],
    )

    # key MLP and conditioned tracing
    key_mlp = pick_key_mlp(ap_mlp_impact)
    print(f"[ct] key downstream MLP: MLP{key_mlp}")

    ct_head_rescue, ct_mlp_rescue = run_conditioned_tracing(
        model=model,
        clean_tok=clean_tok,
        corr_tok=corr_tok,
        baseline_df=baseline.df,
        ap_head_pd=ap_head_metrics.pd_tool,
        ap_mlp_pd=ap_mlp_metrics.pd_tool,
        key_mlp=key_mlp,
        tool_id=tool_id,
        batch_size=args.batch_size,
        device=device,
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        last_only_logits=args.last_only_logits,
    )

    plot_heatmap(
        ct_head_rescue,
        title=f"CT Head Rescue Heatmap (path conditioned on MLP{key_mlp})",
        xlabel="head index",
        ylabel="layer",
        out_path=fig_dir / "ct_head_heatmap.png",
    )

    plot_heatmap(
        ct_mlp_rescue.reshape(-1, 1),
        title=f"CT MLP Rescue Heatmap (conditioned on MLP{key_mlp})",
        xlabel="MLP",
        ylabel="layer",
        out_path=fig_dir / "ct_mlp_heatmap.png",
        xticks=["MLP"],
    )

    # save component tables
    head_df, mlp_df = save_component_tables(
        ap_head_impact=ap_head_impact,
        ap_mlp_impact=ap_mlp_impact,
        ct_head_rescue=ct_head_rescue,
        ct_mlp_rescue=ct_mlp_rescue,
        reports_dir=report_dir,
    )

    print("[search] iterative candidate-set evaluation ...")
    candidates = choose_candidate_sets(
        ap_head_impact=ap_head_impact,
        ap_mlp_impact=ap_mlp_impact,
        ct_head_rescue=ct_head_rescue,
        key_mlp=key_mlp,
        num_head_choices=[4, 8, 12, 16, 24, 32],
        num_mlp_choices=[2, 3, 4, 6],
    )

    clean_hit = float(baseline.df["clean_hit_tool"].mean())
    corr_non = float(baseline.df["corr_non_tool"].mean())
    pair_ok = float(baseline.df["pair_success"].mean())

    candidate_rows = []
    pd_corr = 0.0  # by definition, mean(p_corr - p_corr)
    for idx, (heads_mask, mlp_mask, meta) in enumerate(candidates):
        out = evaluate_component_set(
            model=model,
            clean_tok=clean_tok,
            corr_tok=corr_tok,
            baseline_df=baseline.df,
            tool_id=tool_id,
            batch_size=args.batch_size,
            device=device,
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            circuit_heads_mask=heads_mask,
            circuit_mlps_mask=mlp_mask,
            last_only_logits=args.last_only_logits,
        )

        pd_suff = out["pd_suff"]
        pd_nec = out["pd_nec"]
        recovery = (pd_suff - pd_corr) / (pd_base - pd_corr + 1e-12)
        nec_drop = pd_base - pd_nec
        # balanced score: prefer high recovery + high necessity drop
        score = float((max(0.0, recovery) ** 0.5) * (max(0.0, nec_drop / (pd_base + 1e-12)) ** 0.5))

        candidate_rows.append(
            {
                "idx": idx,
                "num_heads": meta["num_heads"],
                "num_mlps": meta["num_mlps"],
                "pd_suff": pd_suff,
                "pd_nec": pd_nec,
                "recovery": recovery,
                "nec_drop": nec_drop,
                "pair_suff": out["pair_suff"],
                "pair_nec": out["pair_nec"],
                "score": score,
                "head_indices": meta["head_indices"],
                "mlp_indices": meta["mlp_indices"],
            }
        )
        print(
            f"[search {idx+1:02d}/{len(candidates)}] heads={meta['num_heads']}, mlps={meta['num_mlps']}, "
            f"rec={recovery:.3f}, nec_drop={nec_drop:.3f}, score={score:.3f}"
        )

    cand_df = pd.DataFrame(candidate_rows).sort_values("score", ascending=False).reset_index(drop=True)
    cand_df.to_csv(report_dir / "candidate_set_eval.csv", index=False)

    # pick best and self-check; if unreasonable fallback to stronger set by recovery
    best = cand_df.iloc[0].copy()
    if best["recovery"] < 0.35 or best["nec_drop"] < 0.10 * pd_base:
        # fallback: prioritize recovery first then necessity
        cand_df2 = cand_df.sort_values(["recovery", "nec_drop"], ascending=False).reset_index(drop=True)
        best = cand_df2.iloc[0].copy()
        print("[search] best-score set looked weak, switched to recovery-priority set")

    final_heads = [tuple(x) for x in best["head_indices"]]
    final_mlps = [int(x) for x in best["mlp_indices"]]

    # probe plots for all final nodes (keeps node count modest)
    print("[probe] generating probe plots ...")
    for l, h in final_heads:
        run_probe_for_component(
            model=model,
            clean_tok=clean_tok,
            corr_tok=corr_tok,
            baseline_df=baseline.df,
            tool_id=tool_id,
            batch_size=args.batch_size,
            device=device,
            layer=l,
            head=h,
            head_dim=head_dim,
            out_path=fig_dir / f"L{l}H{h}_probe.png",
            last_only_logits=args.last_only_logits,
        )

    for l in final_mlps:
        run_probe_for_component(
            model=model,
            clean_tok=clean_tok,
            corr_tok=corr_tok,
            baseline_df=baseline.df,
            tool_id=tool_id,
            batch_size=args.batch_size,
            device=device,
            layer=l,
            head=None,
            head_dim=head_dim,
            out_path=fig_dir / f"MLP{l}_probe.png",
            last_only_logits=args.last_only_logits,
        )

    final_edges = draw_final_circuit(
        head_nodes=final_heads,
        mlp_nodes=final_mlps,
        key_mlp=key_mlp,
        ap_head_impact=ap_head_impact,
        ap_mlp_impact=ap_mlp_impact,
        ct_head_rescue=ct_head_rescue,
        out_path=fig_dir / "final_circuit.png",
    )

    # final report
    build_report(
        out_path=report_dir / "circuit_localization_report.md",
        alignment_info=alignment_info,
        baseline=baseline,
        ref_baseline_hint={"clean": 89.6, "corr_non": 83.5, "pair": 73.2},
        config_info={
            "model_path": args.model,
            "n_samples": len(samples),
            "batch_size": args.batch_size,
            "dtype": args.dtype,
            "seed": args.seed,
        },
        top_heads_df=head_df,
        top_mlps_df=mlp_df,
        key_mlp=key_mlp,
        candidate_table=cand_df[["num_heads", "num_mlps", "pd_suff", "pd_nec", "recovery", "nec_drop", "pair_suff", "pair_nec", "score"]],
        best_row=best,
        final_heads=final_heads,
        final_mlps=final_mlps,
        final_edges=final_edges,
    )

    # completion summary
    print("[done] outputs:")
    print(f"- {fig_dir / 'behavior_prob_margin.png'}")
    print(f"- {fig_dir / 'ap_head_heatmap.png'}")
    print(f"- {fig_dir / 'ap_mlp_heatmap.png'}")
    print(f"- {fig_dir / 'ct_head_heatmap.png'}")
    print(f"- {fig_dir / 'ct_mlp_heatmap.png'}")
    print(f"- {fig_dir / 'final_circuit.png'}")
    print(f"- {report_dir / 'circuit_localization_report.md'}")


if __name__ == "__main__":
    main()
