#!/usr/bin/env python3
"""
Second-pass refinement for circuit localization:
- statistical significance (bootstrap CIs)
- per-sample sufficiency/necessity traces for best circuit
- paper-aligned figure organization
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

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

# reuse the validated core utilities
from run_circuit_localization import (
    batched_indices,
    calc_single_side_metrics,
    forward_model,
    gather_last_hidden,
    gather_last_logits,
    load_pair_samples,
    make_head_patch_pre_hook,
    make_mlp_patch_hook,
    move_batch,
    register_capture_hooks_for_all_layers,
    remove_handles,
    set_seed,
    slice_batch,
    tokenize_all,
    trim_batch_to_nonpad,
)


def bootstrap_ci(arr: np.ndarray, alpha: float = 0.05) -> Tuple[float, float, float]:
    mean = float(np.mean(arr))
    lo = float(np.quantile(arr, alpha / 2))
    hi = float(np.quantile(arr, 1 - alpha / 2))
    return mean, lo, hi


def bootstrap_core_metrics(
    p_clean: np.ndarray,
    p_corr: np.ndarray,
    p_suff: np.ndarray,
    p_nec: np.ndarray,
    clean_hit: np.ndarray,
    corr_non: np.ndarray,
    pair_success: np.ndarray,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(p_clean)

    rows = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)

        pc = p_clean[idx]
        px = p_corr[idx]
        ps = p_suff[idx]
        pn = p_nec[idx]

        pd_base = float(np.mean(pc - px))
        pd_suff = float(np.mean(ps - px))
        pd_nec = float(np.mean(pn - px))
        recovery = float(pd_suff / (pd_base + 1e-12))
        nec_drop = float(pd_base - pd_nec)

        rows.append(
            {
                "clean_hit": float(np.mean(clean_hit[idx])),
                "corr_non": float(np.mean(corr_non[idx])),
                "pair_success": float(np.mean(pair_success[idx])),
                "pd_base": pd_base,
                "pd_suff": pd_suff,
                "pd_nec": pd_nec,
                "recovery": recovery,
                "nec_drop": nec_drop,
            }
        )

    return pd.DataFrame(rows)


def run_best_circuit_per_sample(
    model: Any,
    clean_tok: Dict[str, torch.Tensor],
    corr_tok: Dict[str, torch.Tensor],
    heads_mask: np.ndarray,
    mlps_mask: np.ndarray,
    tool_id: int,
    batch_size: int,
    device: torch.device,
    num_layers: int,
    head_dim: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return p_suff and p_nec per sample for chosen circuit."""
    n = clean_tok["input_ids"].shape[0]
    p_suff_all = np.zeros(n, dtype=np.float64)
    p_nec_all = np.zeros(n, dtype=np.float64)

    with torch.no_grad():
        for s, e in batched_indices(n, batch_size):
            cb_cpu = trim_batch_to_nonpad(slice_batch(clean_tok, s, e))
            xb_cpu = trim_batch_to_nonpad(slice_batch(corr_tok, s, e))
            cb = move_batch(cb_cpu, device)
            xb = move_batch(xb_cpu, device)

            handles, corr_head_cache, corr_mlp_cache = register_capture_hooks_for_all_layers(
                model,
                num_layers=num_layers,
                capture_heads=True,
                capture_mlps=True,
            )
            _ = forward_model(model, True, **xb)
            remove_handles(handles)

            pos = cb["attention_mask"].sum(dim=1) - 1
            corr_head_last = {l: gather_last_hidden(corr_head_cache[l], pos) for l in range(num_layers)}
            corr_mlp_last = {l: gather_last_hidden(corr_mlp_cache[l], pos) for l in range(num_layers)}

            # sufficiency: keep circuit clean, corrupt everything else
            suff_handles = []
            for l in range(num_layers):
                non_circuit_heads = ~heads_mask[l]
                if non_circuit_heads.any():
                    mask = torch.tensor(non_circuit_heads, device=device, dtype=torch.bool)

                    def make_hook(layer_idx: int, mask_t: torch.Tensor):
                        def _hook(module: Any, inputs: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor]:
                            x = inputs[0].clone()
                            bidx = torch.arange(x.shape[0], device=x.device)
                            for ht in torch.nonzero(mask_t, as_tuple=False).squeeze(-1):
                                h = int(ht.item())
                                hs = h * head_dim
                                he = hs + head_dim
                                x[bidx, pos, hs:he] = corr_head_last[layer_idx][:, hs:he]
                            return (x,)

                        return _hook

                    suff_handles.append(model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(make_hook(l, mask)))

                if not bool(mlps_mask[l]):
                    suff_handles.append(
                        model.model.layers[l].mlp.register_forward_hook(
                            make_mlp_patch_hook(corr_last_layer=corr_mlp_last[l], positions=pos)
                        )
                    )

            suff_logits = forward_model(model, True, **cb).logits
            remove_handles(suff_handles)
            suff_last, _ = gather_last_logits(suff_logits, cb["attention_mask"])
            p_suff, _, _ = calc_single_side_metrics(suff_last, tool_id)
            p_suff_all[s:e] = p_suff.detach().cpu().numpy()

            # necessity: corrupt only circuit components
            nec_handles = []
            for l in range(num_layers):
                circuit_heads = heads_mask[l]
                if circuit_heads.any():
                    mask = torch.tensor(circuit_heads, device=device, dtype=torch.bool)

                    def make_hook(layer_idx: int, mask_t: torch.Tensor):
                        def _hook(module: Any, inputs: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor]:
                            x = inputs[0].clone()
                            bidx = torch.arange(x.shape[0], device=x.device)
                            for ht in torch.nonzero(mask_t, as_tuple=False).squeeze(-1):
                                h = int(ht.item())
                                hs = h * head_dim
                                he = hs + head_dim
                                x[bidx, pos, hs:he] = corr_head_last[layer_idx][:, hs:he]
                            return (x,)

                        return _hook

                    nec_handles.append(model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(make_hook(l, mask)))

                if bool(mlps_mask[l]):
                    nec_handles.append(
                        model.model.layers[l].mlp.register_forward_hook(
                            make_mlp_patch_hook(corr_last_layer=corr_mlp_last[l], positions=pos)
                        )
                    )

            nec_logits = forward_model(model, True, **cb).logits
            remove_handles(nec_handles)
            nec_last, _ = gather_last_logits(nec_logits, cb["attention_mask"])
            p_nec, _, _ = calc_single_side_metrics(nec_last, tool_id)
            p_nec_all[s:e] = p_nec.detach().cpu().numpy()

    return p_suff_all, p_nec_all


def selected_component_impacts(
    model: Any,
    clean_tok: Dict[str, torch.Tensor],
    corr_tok: Dict[str, torch.Tensor],
    tool_id: int,
    batch_size: int,
    device: torch.device,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    selected_heads: List[Tuple[int, int]],
    selected_mlps: List[int],
    p_clean_ref: np.ndarray,
) -> pd.DataFrame:
    n = clean_tok["input_ids"].shape[0]
    impacts: Dict[str, np.ndarray] = {}

    for l, h in selected_heads:
        impacts[f"L{l}H{h}"] = np.zeros(n, dtype=np.float64)
    for l in selected_mlps:
        impacts[f"MLP{l}"] = np.zeros(n, dtype=np.float64)

    with torch.no_grad():
        for s, e in batched_indices(n, batch_size):
            cb_cpu = trim_batch_to_nonpad(slice_batch(clean_tok, s, e))
            xb_cpu = trim_batch_to_nonpad(slice_batch(corr_tok, s, e))
            cb = move_batch(cb_cpu, device)
            xb = move_batch(xb_cpu, device)

            handles, corr_head_cache, corr_mlp_cache = register_capture_hooks_for_all_layers(
                model,
                num_layers=num_layers,
                capture_heads=True,
                capture_mlps=True,
            )
            _ = forward_model(model, True, **xb)
            remove_handles(handles)

            pos = cb["attention_mask"].sum(dim=1) - 1

            # selected heads
            for l, h in selected_heads:
                corr_last = gather_last_hidden(corr_head_cache[l], pos)
                hs = h * head_dim
                he = hs + head_dim

                def _hook(module: Any, inputs: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor]:
                    x = inputs[0].clone()
                    bidx = torch.arange(x.shape[0], device=x.device)
                    x[bidx, pos, hs:he] = corr_last[:, hs:he]
                    return (x,)

                hh = model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(_hook)
                logits = forward_model(model, True, **cb).logits
                hh.remove()
                last, _ = gather_last_logits(logits, cb["attention_mask"])
                p_patch, _, _ = calc_single_side_metrics(last, tool_id)

                # impact_i = p_clean_i - p_patch_i
                impacts[f"L{l}H{h}"][s:e] = p_clean_ref[s:e] - p_patch.detach().cpu().numpy()

            # selected mlps
            for l in selected_mlps:
                corr_last = gather_last_hidden(corr_mlp_cache[l], pos)
                hm = model.model.layers[l].mlp.register_forward_hook(
                    make_mlp_patch_hook(corr_last_layer=corr_last, positions=pos)
                )
                logits = forward_model(model, True, **cb).logits
                hm.remove()
                last, _ = gather_last_logits(logits, cb["attention_mask"])
                p_patch, _, _ = calc_single_side_metrics(last, tool_id)

                impacts[f"MLP{l}"][s:e] = p_clean_ref[s:e] - p_patch.detach().cpu().numpy()

    out = pd.DataFrame({"idx": np.arange(n)})
    for k, v in impacts.items():
        out[k] = v
    return out


def summarize_component_significance(impact_df: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    comp_cols = [c for c in impact_df.columns if c != "idx"]
    n = len(impact_df)

    rows = []
    for c in comp_cols:
        vals = impact_df[c].to_numpy(dtype=np.float64)
        boots = np.zeros(n_boot, dtype=np.float64)
        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)
            boots[b] = vals[idx].mean()
        mean, lo, hi = bootstrap_ci(boots)
        p_non_pos = float(np.mean(boots <= 0))
        rows.append(
            {
                "component": c,
                "impact_mean": float(vals.mean()),
                "impact_ci_low": lo,
                "impact_ci_high": hi,
                "p_boot_le_0": p_non_pos,
            }
        )

    return pd.DataFrame(rows).sort_values("impact_mean", ascending=False)


def plot_probability_heatmap(mat: np.ndarray, row_labels: Sequence[str], out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(14, 3.8), dpi=220)
    if HAS_SEABORN:
        sns.heatmap(
            mat,
            cmap="Blues",
            vmin=0.0,
            vmax=1.0,
            ax=ax,
            cbar_kws={"label": "P(<tool_call>)"},
        )
    else:
        im = ax.imshow(mat, aspect="auto", cmap="Blues", vmin=0.0, vmax=1.0)
        cb = fig.colorbar(im, ax=ax)
        cb.set_label("P(<tool_call>)")

    ax.set_yticks(np.arange(len(row_labels)) + 0.5 if HAS_SEABORN else np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, rotation=0)
    ax.set_xlabel("Sorted Pair Index")
    ax.set_ylabel("Condition")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_probability_slice(
    p_clean: np.ndarray,
    p_corr: np.ndarray,
    p_suff: np.ndarray,
    p_nec: np.ndarray,
    out_path: Path,
) -> None:
    order = np.argsort(-(p_clean - p_corr))
    x = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(12.5, 4.8), dpi=220)
    ax.plot(x, p_clean[order], label="clean", linewidth=2.0, color="#1f77b4")
    ax.plot(x, p_corr[order], label="corrupt", linewidth=1.8, color="#ff7f0e")
    ax.plot(x, p_suff[order], label="sufficiency (best circuit)", linewidth=1.8, color="#2ca02c")
    ax.plot(x, p_nec[order], label="necessity (best circuit)", linewidth=1.8, color="#d62728")
    ax.set_xlabel("Pair index sorted by (p_clean - p_corr)")
    ax.set_ylabel("P(<tool_call>)")
    ax.set_title("Probability Slice Across Pairs (paper-style slice)")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_logits_ipp(ap_head: np.ndarray, ap_mlp: np.ndarray, out_path: Path) -> None:
    # combine into [layer, heads + mlp]
    mat = np.concatenate([ap_head, ap_mlp[:, None]], axis=1)
    fig, ax = plt.subplots(figsize=(10.5, 6.2), dpi=220)
    if HAS_SEABORN:
        sns.heatmap(
            mat,
            cmap="RdBu",
            center=0.0,
            ax=ax,
            cbar_kws={"label": "Impact on PD_tool"},
        )
    else:
        vmax = float(np.max(np.abs(mat)))
        im = ax.imshow(mat, aspect="auto", cmap="RdBu", vmin=-vmax, vmax=vmax)
        cb = fig.colorbar(im, ax=ax)
        cb.set_label("Impact on PD_tool")

    xlabels = [str(i) for i in range(ap_head.shape[1])] + ["mlp"]
    ax.set_xticks(np.arange(len(xlabels)) + 0.5 if HAS_SEABORN else np.arange(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=0)
    ax.set_xlabel("Head / MLP")
    ax.set_ylabel("Layer")
    ax.set_title("IPP: Direct Contributions to Logits (paper-style logits-ipp)")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_top_component_ci(ci_df: pd.DataFrame, out_path: Path, top_k: int = 18) -> None:
    top = ci_df.head(top_k).copy()
    y = np.arange(len(top))[::-1]
    mean = top["impact_mean"].to_numpy()
    lo = top["impact_ci_low"].to_numpy()
    hi = top["impact_ci_high"].to_numpy()

    fig, ax = plt.subplots(figsize=(10.5, 6.8), dpi=220)
    ax.barh(y, mean, color="#4C78A8", alpha=0.85)
    ax.errorbar(mean, y, xerr=[mean - lo, hi - mean], fmt="none", ecolor="black", elinewidth=1.1, capsize=2)
    ax.set_yticks(y)
    ax.set_yticklabels(top["component"].tolist())
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Impact mean on P(<tool_call>) with 95% bootstrap CI")
    ax.set_title("Top Components with Uncertainty (bootstrap)")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def format_ci(mean: float, lo: float, hi: float, pct: bool = False) -> str:
    if pct:
        return f"{100*mean:.2f}% [{100*lo:.2f}%, {100*hi:.2f}%]"
    return f"{mean:.6f} [{lo:.6f}, {hi:.6f}]"


def write_v2_report(
    out_path: Path,
    boot_df: pd.DataFrame,
    comp_ci: pd.DataFrame,
    best_meta: Dict[str, Any],
    figure_notes: List[str],
) -> None:
    lines: List[str] = []

    def ci(col: str, pct: bool = False) -> str:
        m, lo, hi = bootstrap_ci(boot_df[col].to_numpy())
        return format_ci(m, lo, hi, pct=pct)

    lines.append("# Circuit Localization Report v2 (Significance + Paper-Style Figures)")
    lines.append("")
    lines.append("## 1. Statistical Robustness (Bootstrap)")
    lines.append(f"- clean hit rate: {ci('clean_hit', pct=True)}")
    lines.append(f"- corrupt non-tool rate: {ci('corr_non', pct=True)}")
    lines.append(f"- pair success rate: {ci('pair_success', pct=True)}")
    lines.append(f"- PD_base: {ci('pd_base')}")
    lines.append(f"- PD_suff (best circuit): {ci('pd_suff')}")
    lines.append(f"- PD_nec (best circuit): {ci('pd_nec')}")
    lines.append(f"- Recovery: {ci('recovery')}")
    lines.append(f"- Necessity drop: {ci('nec_drop')}")
    lines.append("")

    lines.append("## 2. Best Circuit (from iterative search)")
    lines.append(
        f"- heads={best_meta['num_heads']}, mlps={best_meta['num_mlps']}, "
        f"recovery={best_meta['recovery']:.4f}, nec_drop={best_meta['nec_drop']:.4f}, score={best_meta['score']:.4f}"
    )
    lines.append("")

    lines.append("## 3. Top Component Significance")
    for _, r in comp_ci.head(20).iterrows():
        lines.append(
            f"- {r['component']}: mean={r['impact_mean']:.6f}, "
            f"95%CI=[{r['impact_ci_low']:.6f}, {r['impact_ci_high']:.6f}], "
            f"P_boot(<=0)={r['p_boot_le_0']:.4f}"
        )
    lines.append("")

    lines.append("## 4. Figure Mapping to Paper")
    for n in figure_notes:
        lines.append(f"- {n}")

    out_path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="/root/data/Qwen/Qwen3-1.7B")
    parser.add_argument("--pair-dir", type=str, default="/root/data/R2/pair")
    parser.add_argument("--fig-dir", type=str, default="/root/data/R2/figs")
    parser.add_argument("--report-dir", type=str, default="/root/data/R2/reports")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--top-heads", type=int, default=12)
    parser.add_argument("--top-mlps", type=int, default=6)
    args = parser.parse_args()

    set_seed(args.seed)

    report_dir = Path(args.report_dir)
    fig_dir = Path(args.fig_dir)
    pair_dir = Path(args.pair_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # load existing outputs from first pass
    baseline_df = pd.read_csv(report_dir / "baseline_metrics.csv").sort_values("q").reset_index(drop=True)
    head_df = pd.read_csv(report_dir / "ap_ct_head_scores.csv")
    mlp_df = pd.read_csv(report_dir / "ap_ct_mlp_scores.csv")
    cand_df = pd.read_csv(report_dir / "candidate_set_eval.csv")
    best = cand_df.sort_values("score", ascending=False).iloc[0]

    best_heads = [tuple(x) for x in ast.literal_eval(best["head_indices"])]
    best_mlps = [int(x) for x in ast.literal_eval(best["mlp_indices"])]

    selected_heads = [(int(r["layer"]), int(r["head"])) for _, r in head_df.head(args.top_heads).iterrows()]
    selected_mlps = [int(r["layer"]) for _, r in mlp_df.head(args.top_mlps).iterrows()]

    samples = load_pair_samples(pair_dir)
    clean_texts = [s.clean_text for s in samples]
    corr_texts = [s.corrupt_text for s in samples]

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
    device = torch.device("cuda")
    model.to(device)
    model.eval()

    num_layers = int(model.config.num_hidden_layers)
    num_heads = int(model.config.num_attention_heads)
    hidden_size = int(model.config.hidden_size)
    head_dim = hidden_size // num_heads

    tool_id = tokenizer.encode("<tool_call>", add_special_tokens=False)[0]

    clean_tok, corr_tok = tokenize_all(tokenizer, clean_texts, corr_texts)

    # best circuit masks
    heads_mask = np.zeros((num_layers, num_heads), dtype=bool)
    for l, h in best_heads:
        heads_mask[l, h] = True
    mlps_mask = np.zeros((num_layers,), dtype=bool)
    for l in best_mlps:
        mlps_mask[l] = True

    # p_suff / p_nec per sample
    p_suff, p_nec = run_best_circuit_per_sample(
        model=model,
        clean_tok=clean_tok,
        corr_tok=corr_tok,
        heads_mask=heads_mask,
        mlps_mask=mlps_mask,
        tool_id=tool_id,
        batch_size=args.batch_size,
        device=device,
        num_layers=num_layers,
        head_dim=head_dim,
    )

    # component impacts + CIs
    p_clean = baseline_df["p_clean"].to_numpy(dtype=np.float64)
    p_corr = baseline_df["p_corr"].to_numpy(dtype=np.float64)

    impact_df = selected_component_impacts(
        model=model,
        clean_tok=clean_tok,
        corr_tok=corr_tok,
        tool_id=tool_id,
        batch_size=args.batch_size,
        device=device,
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        selected_heads=selected_heads,
        selected_mlps=selected_mlps,
        p_clean_ref=p_clean,
    )
    impact_df.insert(0, "q", baseline_df["q"].to_numpy())
    impact_df.to_csv(report_dir / "selected_component_impacts_per_sample.csv", index=False)

    comp_ci = summarize_component_significance(impact_df.drop(columns=["q"]), n_boot=args.n_bootstrap, seed=args.seed)
    comp_ci.to_csv(report_dir / "selected_component_impact_ci.csv", index=False)

    # bootstrap core metrics
    boot_df = bootstrap_core_metrics(
        p_clean=p_clean,
        p_corr=p_corr,
        p_suff=p_suff,
        p_nec=p_nec,
        clean_hit=baseline_df["clean_hit_tool"].astype(float).to_numpy(),
        corr_non=baseline_df["corr_non_tool"].astype(float).to_numpy(),
        pair_success=baseline_df["pair_success"].astype(float).to_numpy(),
        n_boot=args.n_bootstrap,
        seed=args.seed,
    )
    boot_df.to_csv(report_dir / "bootstrap_metrics_v2.csv", index=False)

    # figure 1: paper-style probability heatmap (clean/corr/suff/nec)
    order = np.argsort(-(p_clean - p_corr))
    mat_prob = np.vstack([p_clean[order], p_corr[order], p_suff[order], p_nec[order]])
    plot_probability_heatmap(
        mat=mat_prob,
        row_labels=["clean", "corrupt", "suff(best)", "nec(best)"],
        out_path=fig_dir / "probability_heatmap.png",
        title="First-Token <tool_call> Probability Heatmap (paper-style)",
    )

    # figure 2: patched probability heatmap (rows=selected interventions)
    # convert impact back to patched probability: p_patch = p_clean - impact
    interventions = []
    rows = []
    for c in comp_ci.head(min(12, len(comp_ci)))["component"].tolist():
        v = impact_df[c].to_numpy(dtype=np.float64)
        rows.append((p_clean - v)[order])
        interventions.append(c)
    rows = np.vstack(rows)
    plot_probability_heatmap(
        mat=rows,
        row_labels=interventions,
        out_path=fig_dir / "patched_probability_heatmap.png",
        title="Patched Probability Heatmap (top interventions)",
    )

    # figure 3: probability slice
    plot_probability_slice(
        p_clean=p_clean,
        p_corr=p_corr,
        p_suff=p_suff,
        p_nec=p_nec,
        out_path=fig_dir / "probability_slice.png",
    )

    # figure 4: logits-ipp combined map
    ap_head = head_df.pivot(index="layer", columns="head", values="ap_impact_pd").sort_index().to_numpy()
    ap_mlp = mlp_df.sort_values("layer")["ap_impact_pd"].to_numpy()
    plot_logits_ipp(ap_head=ap_head, ap_mlp=ap_mlp, out_path=fig_dir / "logits_ipp.png")

    # figure 5: uncertainty bars
    plot_top_component_ci(comp_ci, out_path=fig_dir / "top_components_ci.png", top_k=min(18, len(comp_ci)))

    # v2 report
    figure_notes = [
        "`figs/probability_heatmap.png` aligns to paper `probability-heatmap` (organized by sorted sample axis and probability scale).",
        "`figs/probability_slice.png` aligns to paper `probability-slice` (line slice across ordered samples).",
        "`figs/logits_ipp.png` aligns to paper `logits-ipp` (layer x [heads+mlp], diverging centered at 0).",
        "`figs/patched_probability_heatmap.png` aligns to paper `patched-probability-heatmap` (intervention-conditioned probability matrix).",
        "`figs/top_components_ci.png` adds uncertainty quantification not shown in original paper but needed for statistical persuasiveness.",
    ]

    write_v2_report(
        out_path=report_dir / "circuit_localization_report_v2.md",
        boot_df=boot_df,
        comp_ci=comp_ci,
        best_meta={
            "num_heads": int(best["num_heads"]),
            "num_mlps": int(best["num_mlps"]),
            "recovery": float(best["recovery"]),
            "nec_drop": float(best["nec_drop"]),
            "score": float(best["score"]),
        },
        figure_notes=figure_notes,
    )

    # export per-sample suff/nec
    out_per = baseline_df[["q", "p_clean", "p_corr"]].copy()
    out_per["p_suff_best"] = p_suff
    out_per["p_nec_best"] = p_nec
    out_per.to_csv(report_dir / "best_circuit_per_sample_probs.csv", index=False)

    print("[done] refinement outputs:")
    print(f"- {report_dir / 'bootstrap_metrics_v2.csv'}")
    print(f"- {report_dir / 'selected_component_impact_ci.csv'}")
    print(f"- {report_dir / 'best_circuit_per_sample_probs.csv'}")
    print(f"- {report_dir / 'circuit_localization_report_v2.md'}")
    print(f"- {fig_dir / 'probability_heatmap.png'}")
    print(f"- {fig_dir / 'patched_probability_heatmap.png'}")
    print(f"- {fig_dir / 'probability_slice.png'}")
    print(f"- {fig_dir / 'logits_ipp.png'}")
    print(f"- {fig_dir / 'top_components_ci.png'}")


if __name__ == "__main__":
    main()
