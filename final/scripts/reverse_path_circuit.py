#!/usr/bin/env python3
"""
Reverse path-patching circuit search for first-token <tool_call>.

Goal:
- start from logits (top AP MLP),
- trace upstream attention head via CT rescue,
- verify targeted path patch metrics,
- render a paper-style A/B/C circuit figure.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()
hf_logging.disable_progress_bar()

from run_circuit_localization import (
    batched_indices,
    forward_model,
    gather_last_hidden,
    gather_last_logits,
    load_pair_samples,
    make_mlp_patch_hook,
    make_mlp_restore_hook,
    move_batch,
    register_capture_hooks_for_all_layers,
    remove_handles,
    set_seed,
    slice_batch,
    tokenize_all,
    trim_batch_to_nonpad,
)


@dataclass
class SelectedPath:
    key_mlp: int
    head_layer: int
    head_idx: int
    mlp_ap_impact: float
    head_ap_impact: float
    head_ct_rescue: float


def calc_tool_prob(last_logits: torch.Tensor, tool_id: int) -> torch.Tensor:
    return F.softmax(last_logits, dim=-1)[:, tool_id]


def load_existing_scores(table_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    head_df = pd.read_csv(table_dir / "ap_ct_head_scores.csv")
    mlp_df = pd.read_csv(table_dir / "ap_ct_mlp_scores.csv")
    base_df = pd.read_csv(table_dir / "baseline_metrics.csv")
    head_df = head_df.sort_values(["ap_impact_pd", "ct_rescue_pd"], ascending=[False, False]).reset_index(drop=True)
    mlp_df = mlp_df.sort_values("ap_impact_pd", ascending=False).reset_index(drop=True)
    base_df = base_df.sort_values("q").reset_index(drop=True)
    return head_df, mlp_df, base_df


def pick_reverse_path(
    head_df: pd.DataFrame,
    mlp_df: pd.DataFrame,
    force_key_mlp: Optional[int] = None,
    force_head: Optional[Tuple[int, int]] = None,
) -> SelectedPath:
    if force_key_mlp is not None:
        key_mlp = int(force_key_mlp)
        mlp_row = mlp_df[mlp_df["layer"] == key_mlp]
        if len(mlp_row) == 0:
            raise ValueError(f"key MLP layer {key_mlp} not found in ap_ct_mlp_scores.csv")
        mlp_ap = float(mlp_row.iloc[0]["ap_impact_pd"])
    else:
        mlp_row = mlp_df.iloc[0]
        key_mlp = int(mlp_row["layer"])
        mlp_ap = float(mlp_row["ap_impact_pd"])

    if force_head is not None:
        hl, hh = force_head
        cands = head_df[(head_df["layer"] == hl) & (head_df["head"] == hh)]
    else:
        # Reverse step: among positive CT rescue heads, prefer strong AP + strong CT
        cands = head_df[(head_df["ct_rescue_pd"] > 0) & (head_df["layer"] <= key_mlp)].copy()
        if len(cands) == 0:
            cands = head_df.copy()
        cands["rank_score"] = 0.55 * cands["ct_rescue_pd"] + 0.45 * cands["ap_impact_pd"]
        cands = cands.sort_values(["rank_score", "ct_rescue_pd", "ap_impact_pd"], ascending=False)

    if len(cands) == 0:
        raise RuntimeError("No head candidate available for reverse path selection")

    row = cands.iloc[0]
    return SelectedPath(
        key_mlp=key_mlp,
        head_layer=int(row["layer"]),
        head_idx=int(row["head"]),
        mlp_ap_impact=mlp_ap,
        head_ap_impact=float(row["ap_impact_pd"]),
        head_ct_rescue=float(row["ct_rescue_pd"]),
    )


def run_single_mlp_patch_pd(
    model: Any,
    clean_tok: Dict[str, torch.Tensor],
    corr_tok: Dict[str, torch.Tensor],
    p_corr_ref: np.ndarray,
    layer: int,
    tool_id: int,
    batch_size: int,
    device: torch.device,
) -> float:
    n = clean_tok["input_ids"].shape[0]
    patched_probs = np.zeros(n, dtype=np.float64)

    with torch.no_grad():
        for s, e in batched_indices(n, batch_size):
            cb_cpu = trim_batch_to_nonpad(slice_batch(clean_tok, s, e))
            xb_cpu = trim_batch_to_nonpad(slice_batch(corr_tok, s, e))
            cb = move_batch(cb_cpu, device)
            xb = move_batch(xb_cpu, device)

            corr_cache: Dict[str, torch.Tensor] = {}

            def _capture_mlp(module: Any, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor) -> torch.Tensor:
                corr_cache["x"] = output.detach()
                return output

            hcap = model.model.layers[layer].mlp.register_forward_hook(_capture_mlp)
            _ = forward_model(model, True, **xb)
            hcap.remove()

            pos = cb["attention_mask"].sum(dim=1) - 1
            corr_last = gather_last_hidden(corr_cache["x"], pos)

            hpatch = model.model.layers[layer].mlp.register_forward_hook(
                make_mlp_patch_hook(corr_last_layer=corr_last, positions=pos)
            )
            logits = forward_model(model, True, **cb).logits
            hpatch.remove()

            last, _ = gather_last_logits(logits, cb["attention_mask"])
            patched_probs[s:e] = calc_tool_prob(last, tool_id).float().detach().cpu().numpy()

    return float(np.mean(patched_probs - p_corr_ref))


def run_single_head_patch_pd(
    model: Any,
    clean_tok: Dict[str, torch.Tensor],
    corr_tok: Dict[str, torch.Tensor],
    p_corr_ref: np.ndarray,
    layer: int,
    head: int,
    head_dim: int,
    tool_id: int,
    batch_size: int,
    device: torch.device,
) -> float:
    n = clean_tok["input_ids"].shape[0]
    patched_probs = np.zeros(n, dtype=np.float64)

    hs = head * head_dim
    he = hs + head_dim

    with torch.no_grad():
        for s, e in batched_indices(n, batch_size):
            cb_cpu = trim_batch_to_nonpad(slice_batch(clean_tok, s, e))
            xb_cpu = trim_batch_to_nonpad(slice_batch(corr_tok, s, e))
            cb = move_batch(cb_cpu, device)
            xb = move_batch(xb_cpu, device)

            corr_cache: Dict[str, torch.Tensor] = {}

            def _capture_head(module: Any, inputs: Tuple[torch.Tensor, ...]) -> None:
                corr_cache["x"] = inputs[0].detach()

            hcap = model.model.layers[layer].self_attn.o_proj.register_forward_pre_hook(_capture_head)
            _ = forward_model(model, True, **xb)
            hcap.remove()

            pos = cb["attention_mask"].sum(dim=1) - 1
            corr_last = gather_last_hidden(corr_cache["x"], pos)

            def _patch_head(module: Any, inputs: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor]:
                x = inputs[0].clone()
                bidx = torch.arange(x.shape[0], device=x.device)
                x[bidx, pos, hs:he] = corr_last[:, hs:he]
                return (x,)

            hpatch = model.model.layers[layer].self_attn.o_proj.register_forward_pre_hook(_patch_head)
            logits = forward_model(model, True, **cb).logits
            hpatch.remove()

            last, _ = gather_last_logits(logits, cb["attention_mask"])
            patched_probs[s:e] = calc_tool_prob(last, tool_id).float().detach().cpu().numpy()

    return float(np.mean(patched_probs - p_corr_ref))


def run_head_patch_with_key_restore_pd(
    model: Any,
    clean_tok: Dict[str, torch.Tensor],
    corr_tok: Dict[str, torch.Tensor],
    p_corr_ref: np.ndarray,
    head_layer: int,
    head_idx: int,
    key_mlp: int,
    head_dim: int,
    tool_id: int,
    batch_size: int,
    device: torch.device,
) -> float:
    n = clean_tok["input_ids"].shape[0]
    patched_probs = np.zeros(n, dtype=np.float64)

    hs = head_idx * head_dim
    he = hs + head_dim

    with torch.no_grad():
        for s, e in batched_indices(n, batch_size):
            cb_cpu = trim_batch_to_nonpad(slice_batch(clean_tok, s, e))
            xb_cpu = trim_batch_to_nonpad(slice_batch(corr_tok, s, e))
            cb = move_batch(cb_cpu, device)
            xb = move_batch(xb_cpu, device)

            # capture corrupt selected head
            corr_cache: Dict[str, torch.Tensor] = {}

            def _capture_head(module: Any, inputs: Tuple[torch.Tensor, ...]) -> None:
                corr_cache["x"] = inputs[0].detach()

            hcap = model.model.layers[head_layer].self_attn.o_proj.register_forward_pre_hook(_capture_head)
            _ = forward_model(model, True, **xb)
            hcap.remove()

            # capture clean key-MLP output
            clean_key_cache: Dict[str, torch.Tensor] = {}

            def _capture_key(module: Any, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor) -> torch.Tensor:
                clean_key_cache["x"] = output.detach()
                return output

            hkey = model.model.layers[key_mlp].mlp.register_forward_hook(_capture_key)
            _ = forward_model(model, True, **cb)
            hkey.remove()

            pos = cb["attention_mask"].sum(dim=1) - 1
            corr_last = gather_last_hidden(corr_cache["x"], pos)
            clean_key_last = gather_last_hidden(clean_key_cache["x"], pos)

            def _patch_head(module: Any, inputs: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor]:
                x = inputs[0].clone()
                bidx = torch.arange(x.shape[0], device=x.device)
                x[bidx, pos, hs:he] = corr_last[:, hs:he]
                return (x,)

            hpatch = model.model.layers[head_layer].self_attn.o_proj.register_forward_pre_hook(_patch_head)
            hres = model.model.layers[key_mlp].mlp.register_forward_hook(
                make_mlp_restore_hook(clean_last_layer=clean_key_last, positions=pos)
            )
            logits = forward_model(model, True, **cb).logits
            hres.remove()
            hpatch.remove()

            last, _ = gather_last_logits(logits, cb["attention_mask"])
            patched_probs[s:e] = calc_tool_prob(last, tool_id).float().detach().cpu().numpy()

    return float(np.mean(patched_probs - p_corr_ref))


def _draw_node(ax: plt.Axes, center: Tuple[float, float], text: str, w: float = 0.42, h: float = 0.14) -> None:
    x, y = center
    rect = Rectangle((x - w / 2, y - h / 2), w, h, linewidth=1.0, edgecolor="#9e9e9e", facecolor="#efefef", zorder=2)
    ax.add_patch(rect)
    ax.text(x, y, text, ha="center", va="center", fontsize=10.5, zorder=3)


def _draw_arrow(
    ax: plt.Axes,
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    color: str,
    lw: float = 1.4,
    ls: str = "-",
    alpha: float = 1.0,
    rad: float = 0.0,
) -> None:
    ar = FancyArrowPatch(
        p1,
        p2,
        arrowstyle="-|>",
        mutation_scale=11,
        linewidth=lw,
        linestyle=ls,
        color=color,
        alpha=alpha,
        connectionstyle=f"arc3,rad={rad}",
        zorder=1,
    )
    ax.add_patch(ar)


def _draw_main_graph(
    ax: plt.Axes,
    x0: float,
    attn_label_low: str,
    attn_label_high: str,
    mlp_label_low: str,
    mlp_label_high: str,
    input_label: str,
    draw_mlp_to_logits: bool,
    normal_color: str,
) -> Dict[str, Tuple[float, float]]:
    pos = {
        "logits": (x0 + 0.45, 0.90),
        "attn_low": (x0 + 0.18, 0.47),
        "attn_high": (x0 + 0.18, 0.69),
        "mlp_low": (x0 + 0.72, 0.47),
        "mlp_high": (x0 + 0.72, 0.69),
        "embed": (x0 + 0.45, 0.27),
        "input": (x0 + 0.45, 0.09),
    }

    _draw_node(ax, pos["logits"], "Logits")
    _draw_node(ax, pos["attn_low"], attn_label_low)
    _draw_node(ax, pos["attn_high"], attn_label_high)
    _draw_node(ax, pos["mlp_low"], mlp_label_low)
    _draw_node(ax, pos["mlp_high"], mlp_label_high)
    _draw_node(ax, pos["embed"], "Token Embeddings\n+ Earlier Layers", w=0.62)
    _draw_node(ax, pos["input"], input_label, w=0.66)

    # Core residual-stream style links (paper-like dense panel)
    _draw_arrow(ax, pos["input"], pos["embed"], normal_color, lw=1.3)
    _draw_arrow(ax, pos["embed"], pos["attn_low"], normal_color, lw=1.1)
    _draw_arrow(ax, pos["embed"], pos["attn_high"], normal_color, lw=1.1)
    _draw_arrow(ax, pos["embed"], pos["mlp_low"], normal_color, lw=1.1)
    _draw_arrow(ax, pos["embed"], pos["mlp_high"], normal_color, lw=1.1)
    _draw_arrow(ax, pos["embed"], pos["logits"], normal_color, lw=1.0)
    _draw_arrow(ax, pos["attn_low"], pos["attn_high"], normal_color, lw=1.0)
    _draw_arrow(ax, pos["attn_low"], pos["mlp_low"], normal_color, lw=1.0)
    _draw_arrow(ax, pos["attn_low"], pos["mlp_high"], normal_color, lw=1.0)
    _draw_arrow(ax, pos["attn_low"], pos["logits"], normal_color, lw=1.0, rad=0.02)
    _draw_arrow(ax, pos["attn_high"], pos["mlp_low"], normal_color, lw=1.0)
    _draw_arrow(ax, pos["attn_high"], pos["mlp_high"], normal_color, lw=1.0)
    _draw_arrow(ax, pos["attn_high"], pos["logits"], normal_color, lw=1.0, rad=-0.02)
    _draw_arrow(ax, pos["mlp_low"], pos["mlp_high"], normal_color, lw=1.0)
    _draw_arrow(ax, pos["mlp_low"], pos["logits"], normal_color, lw=1.0, rad=0.06)
    if draw_mlp_to_logits:
        _draw_arrow(ax, pos["mlp_high"], pos["logits"], normal_color, lw=1.0, rad=-0.06)

    return pos


def draw_triptych_figure(
    out_path: Path,
    path: SelectedPath,
    pd_base: float,
    pd_mlp_patch: float,
    pd_head_patch: float,
    pd_head_patch_key_restore: float,
) -> None:
    normal_color = "#1298ad"
    corrupt_color = "#e53935"
    both_color = "#8a3ffc"

    attn_low = f"Attention\nLayer {max(0, path.head_layer - 1)}"
    attn_high = f"Attention\nLayer {path.head_layer}"
    mlp_low = f"MLP\n{max(0, path.key_mlp - 1)}"
    mlp_high = f"MLP\n{path.key_mlp}"

    fig, ax = plt.subplots(figsize=(16.8, 7.0), dpi=220)
    ax.set_xlim(0.0, 4.25)
    ax.set_ylim(-0.35, 1.02)
    ax.axis("off")

    panel_x = [0.00, 1.35, 2.70]

    # Panel A: normal
    pos_a = _draw_main_graph(
        ax=ax,
        x0=panel_x[0],
        attn_label_low=attn_low,
        attn_label_high=attn_high,
        mlp_label_low=mlp_low,
        mlp_label_high=mlp_high,
        input_label='Clean input\n(pair prompt)',
        draw_mlp_to_logits=True,
        normal_color=normal_color,
    )
    ax.text(panel_x[0] + 0.00, 0.98, "A", fontsize=18, fontweight="bold", ha="left", va="top")

    # Panel B: patch (MLP_k, logits) with corrupt
    pos_b = _draw_main_graph(
        ax=ax,
        x0=panel_x[1],
        attn_label_low=attn_low,
        attn_label_high=attn_high,
        mlp_label_low=mlp_low,
        mlp_label_high=mlp_high,
        input_label='Clean input\n(pair prompt)',
        draw_mlp_to_logits=False,
        normal_color=normal_color,
    )
    ax.text(panel_x[1] + 0.00, 0.98, "B", fontsize=18, fontweight="bold", ha="left", va="top")

    corr_b = {
        "mlp": (panel_x[1] + 1.03, 0.73),
        "embed": (panel_x[1] + 1.03, 0.27),
        "input": (panel_x[1] + 1.03, 0.09),
    }
    _draw_node(ax, corr_b["mlp"], mlp_high)
    _draw_node(ax, corr_b["embed"], "Token Embeddings\n+ Earlier Layers", w=0.62)
    _draw_node(ax, corr_b["input"], 'Corrupt input\n(pair prompt)', w=0.66)
    _draw_arrow(ax, corr_b["input"], corr_b["embed"], corrupt_color, lw=1.4, ls=(0, (1.2, 3.0)))
    _draw_arrow(ax, corr_b["embed"], corr_b["mlp"], corrupt_color, lw=1.4, ls=(0, (1.2, 3.0)))
    _draw_arrow(ax, corr_b["mlp"], pos_b["logits"], corrupt_color, lw=1.5, ls=(0, (1.2, 3.0)))

    # Panel C: patch (Head, MLP_k, logits)-style path
    pos_c = _draw_main_graph(
        ax=ax,
        x0=panel_x[2],
        attn_label_low=attn_low,
        attn_label_high=attn_high,
        mlp_label_low=mlp_low,
        mlp_label_high=mlp_high,
        input_label='Clean input\n(pair prompt)',
        draw_mlp_to_logits=False,
        normal_color=normal_color,
    )
    ax.text(panel_x[2] + 0.00, 0.98, "C", fontsize=18, fontweight="bold", ha="left", va="top")

    corr_c = {
        "attn": (panel_x[2] + 1.03, 0.47),
        "mlp": (panel_x[2] + 1.03, 0.73),
        "embed": (panel_x[2] + 1.03, 0.27),
        "input": (panel_x[2] + 1.03, 0.09),
    }
    _draw_node(ax, corr_c["attn"], f"Head\nL{path.head_layer}H{path.head_idx}", w=0.40)
    _draw_node(ax, corr_c["mlp"], mlp_high)
    _draw_node(ax, corr_c["embed"], "Token Embeddings\n+ Earlier Layers", w=0.62)
    _draw_node(ax, corr_c["input"], 'Corrupt input\n(pair prompt)', w=0.66)
    _draw_arrow(ax, corr_c["input"], corr_c["embed"], corrupt_color, lw=1.4, ls=(0, (1.2, 3.0)))
    _draw_arrow(ax, corr_c["embed"], corr_c["attn"], corrupt_color, lw=1.4, ls=(0, (1.2, 3.0)))
    _draw_arrow(ax, corr_c["attn"], corr_c["mlp"], corrupt_color, lw=1.4, ls=(0, (1.2, 3.0)))
    _draw_arrow(ax, corr_c["mlp"], pos_c["logits"], both_color, lw=1.6, ls=(0, (4.0, 2.4)))
    _draw_arrow(ax, pos_c["embed"], corr_c["mlp"], normal_color, lw=1.3)

    # legend (paper-style)
    lx = panel_x[2] + 1.42
    ly = 0.90
    _draw_arrow(ax, (lx - 0.10, ly), (lx + 0.10, ly), normal_color, lw=1.6)
    ax.text(lx + 0.13, ly, "Normal Input", va="center", fontsize=11)
    _draw_arrow(ax, (lx - 0.10, ly - 0.07), (lx + 0.10, ly - 0.07), corrupt_color, lw=1.6, ls=(0, (1.2, 3.0)))
    ax.text(lx + 0.13, ly - 0.07, "Corrupt Input", va="center", fontsize=11)
    _draw_arrow(ax, (lx - 0.10, ly - 0.14), (lx + 0.10, ly - 0.14), both_color, lw=1.8, ls=(0, (4.0, 2.4)))
    ax.text(lx + 0.13, ly - 0.14, "Both Inputs", va="center", fontsize=11)

    cap = (
        f"Reverse Path Patching Circuit for <tool_call>. "
        f"A: Unpatched PD={pd_base:.4f}. "
        f"B: Patch (MLP{path.key_mlp}, logits) -> corrupt, PD={pd_mlp_patch:.4f}. "
        f"C: Patch (L{path.head_layer}H{path.head_idx}, MLP{path.key_mlp}, logits)-style path, "
        f"PD={pd_head_patch:.4f}, and PD={pd_head_patch_key_restore:.4f} when restoring MLP{path.key_mlp}."
    )
    ax.text(0.02, -0.24, cap, fontsize=11, ha="left", va="top")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def maybe_parse_force_head(s: Optional[str]) -> Optional[Tuple[int, int]]:
    if s is None:
        return None
    if ":" not in s:
        raise ValueError("--force-head must be like '24:6'")
    a, b = s.split(":", 1)
    return int(a), int(b)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="/root/data/Qwen/Qwen3-1.7B")
    parser.add_argument("--pair-dir", type=str, default="/root/data/R2/pair")
    parser.add_argument("--table-dir", type=str, default="/root/data/R2/final/tables")
    parser.add_argument("--fig-dir", type=str, default="/root/data/R2/final/figs")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--force-key-mlp", type=int, default=None)
    parser.add_argument("--force-head", type=str, default=None, help="format: L:H, e.g. 24:6")
    args = parser.parse_args()

    set_seed(args.seed)

    pair_dir = Path(args.pair_dir)
    table_dir = Path(args.table_dir)
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    head_df, mlp_df, base_df = load_existing_scores(table_dir)
    selected = pick_reverse_path(
        head_df=head_df,
        mlp_df=mlp_df,
        force_key_mlp=args.force_key_mlp,
        force_head=maybe_parse_force_head(args.force_head),
    )

    samples = load_pair_samples(pair_dir)
    if len(samples) != len(base_df):
        raise RuntimeError(
            f"sample count mismatch: pair={len(samples)} vs baseline_csv={len(base_df)}; rerun baseline with same subset first."
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype_map[args.dtype],
        device_map=None,
        trust_remote_code=True,
    )
    device = torch.device("cuda")
    model.to(device)
    model.eval()

    num_heads = int(model.config.num_attention_heads)
    hidden_size = int(model.config.hidden_size)
    head_dim = hidden_size // num_heads

    tool_ids = tokenizer.encode("<tool_call>", add_special_tokens=False)
    if len(tool_ids) != 1:
        raise RuntimeError(f"<tool_call> must be one token, got ids={tool_ids}")
    tool_id = int(tool_ids[0])

    clean_texts = [s.clean_text for s in samples]
    corr_texts = [s.corrupt_text for s in samples]
    clean_tok, corr_tok = tokenize_all(tokenizer, clean_texts, corr_texts)

    p_corr_ref = base_df["p_corr"].to_numpy(dtype=np.float64)
    pd_base = float((base_df["p_clean"] - base_df["p_corr"]).mean())

    print(
        f"[reverse] selected key MLP={selected.key_mlp}, head=L{selected.head_layer}H{selected.head_idx}, "
        f"table_ap_mlp={selected.mlp_ap_impact:.6f}, table_ap_head={selected.head_ap_impact:.6f}, table_ct={selected.head_ct_rescue:.6f}"
    )

    pd_mlp_patch = run_single_mlp_patch_pd(
        model=model,
        clean_tok=clean_tok,
        corr_tok=corr_tok,
        p_corr_ref=p_corr_ref,
        layer=selected.key_mlp,
        tool_id=tool_id,
        batch_size=args.batch_size,
        device=device,
    )
    pd_head_patch = run_single_head_patch_pd(
        model=model,
        clean_tok=clean_tok,
        corr_tok=corr_tok,
        p_corr_ref=p_corr_ref,
        layer=selected.head_layer,
        head=selected.head_idx,
        head_dim=head_dim,
        tool_id=tool_id,
        batch_size=args.batch_size,
        device=device,
    )
    pd_head_patch_key_restore = run_head_patch_with_key_restore_pd(
        model=model,
        clean_tok=clean_tok,
        corr_tok=corr_tok,
        p_corr_ref=p_corr_ref,
        head_layer=selected.head_layer,
        head_idx=selected.head_idx,
        key_mlp=selected.key_mlp,
        head_dim=head_dim,
        tool_id=tool_id,
        batch_size=args.batch_size,
        device=device,
    )

    ap_drop_mlp = pd_base - pd_mlp_patch
    ap_drop_head = pd_base - pd_head_patch
    ct_rescue_head = pd_head_patch_key_restore - pd_head_patch
    mediated_frac = ct_rescue_head / (ap_drop_head + 1e-12)

    metrics_rows = [
        {"metric": "pd_base", "value": pd_base},
        {"metric": "pd_mlp_patch", "value": pd_mlp_patch},
        {"metric": "pd_head_patch", "value": pd_head_patch},
        {"metric": "pd_head_patch_key_restore", "value": pd_head_patch_key_restore},
        {"metric": "ap_drop_mlp", "value": ap_drop_mlp},
        {"metric": "ap_drop_head", "value": ap_drop_head},
        {"metric": "ct_rescue_head", "value": ct_rescue_head},
        {"metric": "mediated_fraction_head_via_key_mlp", "value": mediated_frac},
    ]
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(table_dir / "reverse_path_metrics.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "key_mlp": selected.key_mlp,
                "head_layer": selected.head_layer,
                "head_idx": selected.head_idx,
                "table_ap_mlp_impact": selected.mlp_ap_impact,
                "table_ap_head_impact": selected.head_ap_impact,
                "table_ct_head_rescue": selected.head_ct_rescue,
                "measured_ap_mlp_drop": ap_drop_mlp,
                "measured_ap_head_drop": ap_drop_head,
                "measured_ct_head_rescue": ct_rescue_head,
            }
        ]
    )
    summary.to_csv(table_dir / "reverse_selected_path.csv", index=False)

    draw_triptych_figure(
        out_path=fig_dir / "reverse_circuit_triptych.png",
        path=selected,
        pd_base=pd_base,
        pd_mlp_patch=pd_mlp_patch,
        pd_head_patch=pd_head_patch,
        pd_head_patch_key_restore=pd_head_patch_key_restore,
    )

    # Optional compact node list for future expansion
    top_heads = head_df[(head_df["ct_rescue_pd"] > 0) & (head_df["layer"] <= selected.key_mlp)].copy()
    top_heads = top_heads.sort_values(["ct_rescue_pd", "ap_impact_pd"], ascending=False).head(8)
    top_heads.to_csv(table_dir / "reverse_top_heads_via_key_mlp.csv", index=False)

    print("[done] reverse-path outputs:")
    print(f"- {table_dir / 'reverse_selected_path.csv'}")
    print(f"- {table_dir / 'reverse_path_metrics.csv'}")
    print(f"- {table_dir / 'reverse_top_heads_via_key_mlp.csv'}")
    print(f"- {fig_dir / 'reverse_circuit_triptych.png'}")


if __name__ == "__main__":
    main()
