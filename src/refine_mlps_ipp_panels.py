#!/usr/bin/env python3
"""
Iterative refinement v3:
- paper-style multi-panel MLP-conditioned IPP figure
- overlap/correlation summaries for upstream head sets
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
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

from run_circuit_localization import (
    load_pair_samples,
    run_conditioned_tracing,
    set_seed,
    tokenize_all,
)


def build_ap_mats(head_df: pd.DataFrame, mlp_df: pd.DataFrame, num_layers: int, num_heads: int) -> Tuple[np.ndarray, np.ndarray]:
    ap_head = np.zeros((num_layers, num_heads), dtype=np.float64)
    ap_mlp = np.zeros((num_layers,), dtype=np.float64)

    for _, r in head_df.iterrows():
        l = int(r["layer"])
        h = int(r["head"])
        ap_head[l, h] = float(r["ap_impact_pd"])

    for _, r in mlp_df.iterrows():
        l = int(r["layer"])
        ap_mlp[l] = float(r["ap_impact_pd"])

    return ap_head, ap_mlp


def top_heads_from_mat(mat: np.ndarray, k: int) -> List[Tuple[int, int, float]]:
    flat = mat.reshape(-1)
    idx = np.argsort(-flat)[:k]
    n_heads = mat.shape[1]
    out = []
    for i in idx:
        l = int(i // n_heads)
        h = int(i % n_heads)
        out.append((l, h, float(mat[l, h])))
    return out


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def plot_mlps_ipp_panels(
    ct_dict: Dict[int, np.ndarray],
    out_path: Path,
    top_k_annot: int = 4,
) -> None:
    keys = sorted(ct_dict.keys())
    n = len(keys)
    cols = 2
    rows = int(np.ceil(n / cols))

    # robust symmetric range for shared color scale
    all_vals = np.concatenate([ct_dict[k].reshape(-1) for k in keys])
    vmax = float(np.quantile(np.abs(all_vals), 0.995))
    vmax = max(vmax, 1e-6)

    fig, axes = plt.subplots(rows, cols, figsize=(12.5, 5.0 * rows), dpi=220, constrained_layout=True)
    axes = np.atleast_1d(axes).reshape(rows, cols)

    for i, k in enumerate(keys):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        mat = ct_dict[k]
        if HAS_SEABORN:
            sns.heatmap(
                mat,
                cmap="RdBu_r",
                center=0.0,
                vmin=-vmax,
                vmax=vmax,
                ax=ax,
                cbar=False,
            )
        else:
            ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)

        ax.set_title(f"MLP{k}")
        ax.set_xlabel("Head")
        ax.set_ylabel("Layer")

        tops = top_heads_from_mat(mat, top_k_annot)
        text = "\n".join([f"L{l}H{h}: {v:.3f}" for l, h, v in tops])
        ax.text(
            1.01,
            0.5,
            text,
            transform=ax.transAxes,
            va="center",
            ha="left",
            fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="0.8"),
        )

    # hide unused axes
    for j in range(n, rows * cols):
        r, c = divmod(j, cols)
        axes[r, c].axis("off")

    # shared colorbar
    from matplotlib.cm import ScalarMappable
    import matplotlib.colors as mcolors

    sm = ScalarMappable(norm=mcolors.Normalize(vmin=-vmax, vmax=vmax), cmap="RdBu_r")
    cbar = fig.colorbar(sm, ax=axes, fraction=0.022, pad=0.02)
    cbar.set_label("CT Rescue Impact (center=0)")

    fig.suptitle("IPP via Multiple Key MLPs (paper-style mlps-ipp panels)", y=1.02, fontsize=15)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_overlap_heatmap(overlap: np.ndarray, labels: List[str], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.2), dpi=220)
    if HAS_SEABORN:
        sns.heatmap(overlap, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, xticklabels=labels, yticklabels=labels, ax=ax)
    else:
        im = ax.imshow(overlap, cmap="Blues", vmin=0, vmax=1)
        for i in range(overlap.shape[0]):
            for j in range(overlap.shape[1]):
                ax.text(j, i, f"{overlap[i,j]:.2f}", ha="center", va="center", fontsize=8)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title("Top-Head Set Overlap Across Key MLPs (Jaccard)")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_correlation_bar(corr_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=220)
    x = np.arange(len(corr_df))
    y = corr_df["pearson_r_ap_vs_ct"].to_numpy()
    ax.bar(x, y, color="#4C78A8")
    ax.set_xticks(x)
    ax.set_xticklabels([f"MLP{int(v)}" for v in corr_df["key_mlp"].to_numpy()], rotation=0)
    ax.set_ylim(min(-0.1, y.min() - 0.05), max(1.0, y.max() + 0.05))
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_ylabel("Pearson r")
    ax.set_title("Correlation: AP Head Impact vs CT Rescue")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def write_v3_note(
    out_path: Path,
    corr_df: pd.DataFrame,
    top_df: pd.DataFrame,
    overlap: pd.DataFrame,
) -> None:
    lines: List[str] = []
    lines.append("# MLP-IPP Panel Refinement Note (v3)")
    lines.append("")
    lines.append("## 1. Key MLP Panels")
    lines.append("- Multiple key downstream MLPs were traced with conditioned path patching to test whether upstream head structure is stable or MLP-specific.")
    lines.append("- Figure: `figs/mlps_ipp_panels.png`")
    lines.append("")

    lines.append("## 2. AP-CT Agreement")
    for _, r in corr_df.iterrows():
        lines.append(f"- MLP{int(r['key_mlp'])}: Pearson r(AP head impact, CT rescue) = {r['pearson_r_ap_vs_ct']:.4f}")
    lines.append("")

    lines.append("## 3. Top Upstream Heads per Key MLP")
    for key_mlp in sorted(top_df["key_mlp"].unique()):
        lines.append(f"- MLP{int(key_mlp)}:")
        for _, r in top_df[top_df["key_mlp"] == key_mlp].head(8).iterrows():
            lines.append(f"  - L{int(r['layer'])}H{int(r['head'])}: rescue={r['ct_rescue']:.6f}")
    lines.append("")

    lines.append("## 4. Overlap Matrix")
    lines.append("- Top-head set overlap (Jaccard, top-16 each panel):")
    lines.append("")
    lines.append(overlap.to_markdown(index=True))

    out_path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="/root/data/Qwen/Qwen3-1.7B")
    parser.add_argument("--pair-dir", type=str, default="/root/data/R2/pair")
    parser.add_argument("--report-dir", type=str, default="/root/data/R2/reports")
    parser.add_argument("--fig-dir", type=str, default="/root/data/R2/figs")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--top-key-mlps", type=int, default=4)
    parser.add_argument("--top-heads-overlap", type=int, default=16)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    set_seed(args.seed)

    report_dir = Path(args.report_dir)
    fig_dir = Path(args.fig_dir)
    pair_dir = Path(args.pair_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    baseline_df = pd.read_csv(report_dir / "baseline_metrics.csv").sort_values("q").reset_index(drop=True)
    head_df = pd.read_csv(report_dir / "ap_ct_head_scores.csv")
    mlp_df = pd.read_csv(report_dir / "ap_ct_mlp_scores.csv")

    # choose key MLPs by positive AP impact
    mlp_pos = mlp_df[mlp_df["ap_impact_pd"] > 0].sort_values("ap_impact_pd", ascending=False)
    key_mlps = [int(x) for x in mlp_pos["layer"].head(args.top_key_mlps).tolist()]
    if not key_mlps:
        key_mlps = [int(mlp_df.sort_values("ap_impact_pd", ascending=False).iloc[0]["layer"])]

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

    pd_base = float((baseline_df["p_clean"] - baseline_df["p_corr"]).mean())

    ap_head_impact, ap_mlp_impact = build_ap_mats(head_df, mlp_df, num_layers=num_layers, num_heads=num_heads)
    ap_head_pd = pd_base - ap_head_impact
    ap_mlp_pd = pd_base - ap_mlp_impact

    ct_panels: Dict[int, np.ndarray] = {}
    corr_rows = []
    top_rows = []

    for key_mlp in key_mlps:
        head_rescue, _ = run_conditioned_tracing(
            model=model,
            clean_tok=clean_tok,
            corr_tok=corr_tok,
            baseline_df=baseline_df,
            ap_head_pd=ap_head_pd,
            ap_mlp_pd=ap_mlp_pd,
            key_mlp=key_mlp,
            tool_id=tool_id,
            batch_size=args.batch_size,
            device=device,
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            last_only_logits=True,
        )

        ct_panels[key_mlp] = head_rescue

        # AP vs CT correlation across all heads
        x = ap_head_impact.reshape(-1)
        y = head_rescue.reshape(-1)
        r = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else 0.0
        corr_rows.append({"key_mlp": key_mlp, "pearson_r_ap_vs_ct": r})

        tops = top_heads_from_mat(head_rescue, k=32)
        for rank, (l, h, v) in enumerate(tops, start=1):
            top_rows.append(
                {
                    "key_mlp": key_mlp,
                    "rank": rank,
                    "layer": l,
                    "head": h,
                    "component": f"L{l}H{h}",
                    "ct_rescue": v,
                }
            )

    corr_df = pd.DataFrame(corr_rows).sort_values("key_mlp")
    top_df = pd.DataFrame(top_rows).sort_values(["key_mlp", "rank"]) 

    corr_df.to_csv(report_dir / "mlps_ipp_ap_ct_correlation.csv", index=False)
    top_df.to_csv(report_dir / "mlps_ipp_top_heads.csv", index=False)

    # overlap matrix of top-k head sets
    labels = [f"MLP{k}" for k in sorted(key_mlps)]
    sets = {}
    for k in sorted(key_mlps):
        top_k = top_df[top_df["key_mlp"] == k].head(args.top_heads_overlap)
        sets[k] = set([(int(r["layer"]), int(r["head"])) for _, r in top_k.iterrows()])

    overlap = np.zeros((len(key_mlps), len(key_mlps)), dtype=np.float64)
    key_sorted = sorted(key_mlps)
    for i, ki in enumerate(key_sorted):
        for j, kj in enumerate(key_sorted):
            overlap[i, j] = jaccard(sets[ki], sets[kj])

    overlap_df = pd.DataFrame(overlap, index=[f"MLP{k}" for k in key_sorted], columns=[f"MLP{k}" for k in key_sorted])
    overlap_df.to_csv(report_dir / "mlps_ipp_top_head_overlap.csv", index=True)

    # figures
    plot_mlps_ipp_panels(ct_panels, fig_dir / "mlps_ipp_panels.png", top_k_annot=4)
    plot_overlap_heatmap(overlap, labels=labels, out_path=fig_dir / "mlps_ipp_overlap.png")
    plot_correlation_bar(corr_df, fig_dir / "mlps_ipp_ap_ct_correlation.png")

    write_v3_note(
        out_path=report_dir / "circuit_localization_report_v3_note.md",
        corr_df=corr_df,
        top_df=top_df,
        overlap=overlap_df,
    )

    print("[done] v3 outputs:")
    print(f"- key_mlps={key_mlps}")
    print(f"- {fig_dir / 'mlps_ipp_panels.png'}")
    print(f"- {fig_dir / 'mlps_ipp_overlap.png'}")
    print(f"- {fig_dir / 'mlps_ipp_ap_ct_correlation.png'}")
    print(f"- {report_dir / 'mlps_ipp_top_heads.csv'}")
    print(f"- {report_dir / 'mlps_ipp_ap_ct_correlation.csv'}")
    print(f"- {report_dir / 'mlps_ipp_top_head_overlap.csv'}")
    print(f"- {report_dir / 'circuit_localization_report_v3_note.md'}")


if __name__ == "__main__":
    main()
