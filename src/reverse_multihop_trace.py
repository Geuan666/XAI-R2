#!/usr/bin/env python3
"""
Multi-hop reverse tracing from logits via conditioned path patching.

Workflow:
1) choose root key MLP(s) by AP impact (or user-specified)
2) for each key MLP, run conditioned tracing CT(C | key_mlp)
3) keep many positive head contributors and top positive upstream MLP contributors
4) recurse on upstream MLPs by depth
5) output graph/tables + per-key MLP head panels
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()
hf_logging.disable_progress_bar()

try:
    import seaborn as sns

    HAS_SEABORN = True
except Exception:
    HAS_SEABORN = False

from run_circuit_localization import (
    load_pair_samples,
    run_conditioned_tracing,
    set_seed,
    tokenize_all,
)


@dataclass
class KeyTraceResult:
    key_mlp: int
    depth: int
    head_rescue: np.ndarray
    mlp_rescue: np.ndarray


def build_ap_mats(head_df: pd.DataFrame, mlp_df: pd.DataFrame, num_layers: int, num_heads: int) -> Tuple[np.ndarray, np.ndarray]:
    ap_head = np.zeros((num_layers, num_heads), dtype=np.float64)
    ap_mlp = np.zeros((num_layers,), dtype=np.float64)

    for _, r in head_df.iterrows():
        ap_head[int(r["layer"]), int(r["head"])] = float(r["ap_impact_pd"])
    for _, r in mlp_df.iterrows():
        ap_mlp[int(r["layer"])] = float(r["ap_impact_pd"])
    return ap_head, ap_mlp


def parse_root_mlps(s: Optional[str], mlp_df: pd.DataFrame, top_k: int) -> List[int]:
    if s is not None and s.strip():
        vals = [int(x.strip()) for x in s.split(",") if x.strip()]
        if len(vals) == 0:
            raise ValueError("parsed empty --root-mlps")
        return vals
    pos = mlp_df[mlp_df["ap_impact_pd"] > 0].sort_values("ap_impact_pd", ascending=False)
    roots = [int(x) for x in pos["layer"].head(top_k).tolist()]
    if len(roots) == 0:
        roots = [int(mlp_df.sort_values("ap_impact_pd", ascending=False).iloc[0]["layer"])]
    return roots


def select_top_heads(
    head_rescue: np.ndarray,
    key_mlp: int,
    k: int,
    min_rescue: float,
) -> List[Tuple[int, int, float]]:
    rows = []
    n_layers, n_heads = head_rescue.shape
    for l in range(n_layers):
        if l > key_mlp:
            continue
        for h in range(n_heads):
            v = float(head_rescue[l, h])
            if v >= min_rescue:
                rows.append((l, h, v))
    rows.sort(key=lambda x: x[2], reverse=True)
    return rows[:k]


def select_upstream_mlps(
    mlp_rescue: np.ndarray,
    key_mlp: int,
    k: int,
    min_rescue: float,
    min_layer: int = 0,
) -> List[Tuple[int, float]]:
    rows = []
    for l, v in enumerate(mlp_rescue):
        if l >= key_mlp:
            continue
        if l < min_layer:
            continue
        vv = float(v)
        if vv >= min_rescue:
            rows.append((l, vv))
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows[:k]


def plot_key_panels(
    key_to_head_rescue: Dict[int, np.ndarray],
    out_path: Path,
    top_k_annot: int = 5,
) -> None:
    keys = sorted(key_to_head_rescue.keys())
    if len(keys) == 0:
        return
    cols = 2
    rows = int(np.ceil(len(keys) / cols))

    all_vals = np.concatenate([key_to_head_rescue[k].reshape(-1) for k in keys])
    vmax = float(np.quantile(np.abs(all_vals), 0.995))
    vmax = max(vmax, 1e-6)

    fig, axes = plt.subplots(rows, cols, figsize=(13.6, 5.0 * rows), dpi=220)
    axes = np.atleast_1d(axes).reshape(rows, cols)

    for i, key in enumerate(keys):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        mat = key_to_head_rescue[key]
        if HAS_SEABORN:
            sns.heatmap(mat, cmap="RdBu_r", center=0.0, vmin=-vmax, vmax=vmax, ax=ax, cbar=False)
        else:
            ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)

        ax.set_title(f"Key MLP{key}")
        ax.set_xlabel("Head")
        ax.set_ylabel("Layer")

        flat = mat.reshape(-1)
        idx = np.argsort(-flat)[:top_k_annot]
        n_heads = mat.shape[1]
        txt = []
        for j in idx:
            l = int(j // n_heads)
            h = int(j % n_heads)
            txt.append(f"L{l}H{h}: {mat[l,h]:.3f}")
        ax.text(
            0.985,
            0.03,
            "\n".join(txt),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7,
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.8", alpha=0.85),
        )

    for j in range(len(keys), rows * cols):
        r, c = divmod(j, cols)
        axes[r, c].axis("off")

    from matplotlib.cm import ScalarMappable
    import matplotlib.colors as mcolors

    sm = ScalarMappable(norm=mcolors.Normalize(vmin=-vmax, vmax=vmax), cmap="RdBu_r")
    fig.subplots_adjust(right=0.90, wspace=0.28, hspace=0.22, top=0.90)
    cax = fig.add_axes([0.915, 0.16, 0.018, 0.70])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("CT Rescue Impact (center=0)")

    fig.suptitle("Multi-hop Reverse Tracing: Key-MLP Head Panels", y=0.975, fontsize=15)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _draw_arrow(
    ax: plt.Axes,
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    color: str,
    lw: float,
    alpha: float = 0.9,
    ls: str = "-",
) -> None:
    a = FancyArrowPatch(
        p1,
        p2,
        arrowstyle="-|>",
        mutation_scale=9,
        linewidth=lw,
        color=color,
        alpha=alpha,
        linestyle=ls,
        connectionstyle="arc3,rad=0.03",
    )
    ax.add_patch(a)


def plot_multihop_graph(
    out_path: Path,
    edges_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    num_layers: int,
) -> None:
    fig, ax = plt.subplots(figsize=(13.4, 8.4), dpi=220)
    ax.axis("off")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.0, 1.0)

    # x by depth (logits at right)
    logits_x = 0.95
    mlp_depth_to_x = {0: 0.78, 1: 0.56, 2: 0.34, 3: 0.18}

    # place mlps
    pos: Dict[str, Tuple[float, float]] = {"LOGITS": (logits_x, 0.50)}
    mlp_nodes = nodes_df[nodes_df["node_type"] == "mlp"].copy()
    head_nodes = nodes_df[nodes_df["node_type"] == "head"].copy()

    for _, r in mlp_nodes.iterrows():
        name = str(r["node"])
        depth = int(r["depth"])
        layer = int(r["layer"])
        x = mlp_depth_to_x.get(depth, 0.10)
        y = 0.08 + 0.84 * (layer / max(1, num_layers - 1))
        pos[name] = (x, y)

    # place heads based on min target depth
    head_depth_map: Dict[str, int] = {}
    for _, e in edges_df[edges_df["edge_type"] == "head_to_mlp"].iterrows():
        src = str(e["src"])
        d = int(e["dst_depth"])
        if src not in head_depth_map or d < head_depth_map[src]:
            head_depth_map[src] = d
    for _, r in head_nodes.iterrows():
        name = str(r["node"])
        layer = int(r["layer"])
        d = head_depth_map.get(name, 0)
        x = mlp_depth_to_x.get(d, 0.10) - 0.10
        y = 0.08 + 0.84 * (layer / max(1, num_layers - 1))
        pos[name] = (x, y)

    # draw edges
    if len(edges_df) > 0:
        vmax = float(np.max(np.abs(edges_df["weight"].to_numpy())))
        vmax = max(vmax, 1e-6)
    else:
        vmax = 1.0
    for _, e in edges_df.iterrows():
        src = str(e["src"])
        dst = str(e["dst"])
        if src not in pos or dst not in pos:
            continue
        w = float(e["weight"])
        lw = 0.8 + 4.2 * min(1.0, abs(w) / vmax)
        if e["edge_type"] == "head_to_mlp":
            _draw_arrow(ax, pos[src], pos[dst], color="#d1495b", lw=lw, alpha=0.74)
        elif e["edge_type"] == "mlp_to_mlp":
            _draw_arrow(ax, pos[src], pos[dst], color="#e76f51", lw=lw, alpha=0.82)
        else:  # mlp_to_logits
            _draw_arrow(ax, pos[src], pos[dst], color="#6f42c1", lw=lw, alpha=0.90, ls=(0, (3.5, 2.0)))

    # draw nodes
    ax.scatter([pos["LOGITS"][0]], [pos["LOGITS"][1]], s=600, c="#264653", edgecolors="black", linewidths=1.0, zorder=3)
    ax.text(pos["LOGITS"][0] + 0.010, pos["LOGITS"][1], "LOGITS(<tool_call>)", va="center", ha="left", fontsize=10)

    for _, r in mlp_nodes.iterrows():
        name = str(r["node"])
        x, y = pos[name]
        is_root = bool(r.get("is_root", False))
        color = "#f4a261" if not is_root else "#e76f51"
        size = 360 if not is_root else 430
        ax.scatter([x], [y], s=size, c=color, edgecolors="black", linewidths=0.9, zorder=3)
        ax.text(x + 0.007, y + 0.012, name, fontsize=8.6, ha="left", va="bottom")

    for _, r in head_nodes.iterrows():
        name = str(r["node"])
        x, y = pos[name]
        ax.scatter([x], [y], s=230, c="#2a9d8f", edgecolors="black", linewidths=0.75, zorder=3)
        ax.text(x - 0.006, y, name, fontsize=7.7, ha="right", va="center")

    ax.text(0.01, 0.985, "Multi-hop Reverse Circuit Backtrace", transform=ax.transAxes, fontsize=14, va="top")
    ax.text(
        0.01,
        0.955,
        "Edges: head->MLP (CT rescue), MLP->MLP (CT rescue), MLP->logits (AP direct).",
        transform=ax.transAxes,
        fontsize=10,
        va="top",
    )

    # tiny legend
    ly = 0.90
    _draw_arrow(ax, (0.02, ly), (0.08, ly), color="#d1495b", lw=2.0, alpha=0.8)
    ax.text(0.085, ly, "Head -> MLP", va="center", fontsize=9)
    _draw_arrow(ax, (0.20, ly), (0.26, ly), color="#e76f51", lw=2.0, alpha=0.85)
    ax.text(0.265, ly, "MLP -> MLP", va="center", fontsize=9)
    _draw_arrow(ax, (0.38, ly), (0.44, ly), color="#6f42c1", lw=2.0, alpha=0.9, ls=(0, (3.5, 2.0)))
    ax.text(0.445, ly, "MLP -> Logits", va="center", fontsize=9)

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="/root/data/Qwen/Qwen3-1.7B")
    parser.add_argument("--pair-dir", type=str, default="/root/data/R2/pair")
    parser.add_argument("--table-dir", type=str, default="/root/data/R2/final/tables")
    parser.add_argument("--fig-dir", type=str, default="/root/data/R2/final/figs")
    parser.add_argument("--output-prefix", type=str, default="reverse_multihop")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)

    parser.add_argument("--root-mlps", type=str, default="27", help="comma list, e.g. '27' or '27,22'")
    parser.add_argument("--default-root-topk", type=int, default=1)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--top-upstream-mlps", type=int, default=2)
    parser.add_argument("--root-heads", type=int, default=24)
    parser.add_argument("--child-heads", type=int, default=10)
    parser.add_argument("--min-head-rescue", type=float, default=0.006)
    parser.add_argument("--min-mlp-rescue", type=float, default=0.004)
    parser.add_argument("--min-direct-mlp-ap", type=float, default=0.010)
    parser.add_argument("--min-upstream-layer", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)

    table_dir = Path(args.table_dir)
    fig_dir = Path(args.fig_dir)
    pair_dir = Path(args.pair_dir)
    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    baseline_df = pd.read_csv(table_dir / "baseline_metrics.csv").sort_values("q").reset_index(drop=True)
    head_df = pd.read_csv(table_dir / "ap_ct_head_scores.csv")
    mlp_df = pd.read_csv(table_dir / "ap_ct_mlp_scores.csv")

    roots = parse_root_mlps(args.root_mlps, mlp_df, args.default_root_topk)
    print(f"[roots] {roots}")

    samples = load_pair_samples(pair_dir)
    if len(samples) != len(baseline_df):
        raise RuntimeError(
            f"sample count mismatch: pair={len(samples)} vs baseline={len(baseline_df)}. "
            "Use matching subset first."
        )

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
    clean_texts = [s.clean_text for s in samples]
    corr_texts = [s.corrupt_text for s in samples]
    clean_tok, corr_tok = tokenize_all(tokenizer, clean_texts, corr_texts)

    pd_base = float((baseline_df["p_clean"] - baseline_df["p_corr"]).mean())
    ap_head_impact, ap_mlp_impact = build_ap_mats(head_df, mlp_df, num_layers=num_layers, num_heads=num_heads)
    ap_head_pd = pd_base - ap_head_impact
    ap_mlp_pd = pd_base - ap_mlp_impact

    queue: deque[Tuple[int, int]] = deque()
    mlp_depth: Dict[int, int] = {}
    for r in roots:
        queue.append((r, 0))
        mlp_depth[r] = 0

    key_results: Dict[int, KeyTraceResult] = {}
    edges: List[Dict[str, Any]] = []
    visited_order: List[int] = []

    while queue:
        key_mlp, depth = queue.popleft()
        if depth > args.max_depth:
            continue
        if key_mlp in key_results:
            continue

        print(f"[trace] depth={depth} key=MLP{key_mlp}")
        head_rescue, mlp_rescue = run_conditioned_tracing(
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
        key_results[key_mlp] = KeyTraceResult(key_mlp=key_mlp, depth=depth, head_rescue=head_rescue, mlp_rescue=mlp_rescue)
        visited_order.append(key_mlp)

        # direct MLP->logits edge if AP direct is non-trivial
        ap_direct = float(ap_mlp_impact[key_mlp])
        if ap_direct >= args.min_direct_mlp_ap or depth == 0:
            edges.append(
                {
                    "src": f"MLP{key_mlp}",
                    "dst": "LOGITS",
                    "edge_type": "mlp_to_logits",
                    "weight": ap_direct,
                    "src_depth": depth,
                    "dst_depth": -1,
                }
            )

        k_heads = args.root_heads if depth == 0 else args.child_heads
        top_heads = select_top_heads(head_rescue, key_mlp=key_mlp, k=k_heads, min_rescue=args.min_head_rescue)
        for l, h, v in top_heads:
            edges.append(
                {
                    "src": f"L{l}H{h}",
                    "dst": f"MLP{key_mlp}",
                    "edge_type": "head_to_mlp",
                    "weight": v,
                    "src_depth": depth,
                    "dst_depth": depth,
                }
            )

        up_mlps = select_upstream_mlps(
            mlp_rescue,
            key_mlp=key_mlp,
            k=args.top_upstream_mlps,
            min_rescue=args.min_mlp_rescue,
            min_layer=args.min_upstream_layer,
        )
        for u, v in up_mlps:
            edges.append(
                {
                    "src": f"MLP{u}",
                    "dst": f"MLP{key_mlp}",
                    "edge_type": "mlp_to_mlp",
                    "weight": v,
                    "src_depth": depth + 1,
                    "dst_depth": depth,
                }
            )
            new_depth = depth + 1
            if new_depth <= args.max_depth:
                if u not in mlp_depth or new_depth < mlp_depth[u]:
                    mlp_depth[u] = new_depth
                    queue.append((u, new_depth))

    edges_df = pd.DataFrame(edges)
    if len(edges_df) == 0:
        raise RuntimeError("No edges collected; thresholds may be too strict.")

    # build node table from edges
    node_rows = [{"node": "LOGITS", "node_type": "logits", "layer": num_layers - 1, "depth": -1, "is_root": False}]
    mlp_nodes = sorted({int(x.replace("MLP", "")) for x in edges_df["src"].tolist() + edges_df["dst"].tolist() if str(x).startswith("MLP")})
    head_nodes = sorted({x for x in edges_df["src"].tolist() + edges_df["dst"].tolist() if str(x).startswith("L") and "H" in str(x)})
    root_set = set(roots)
    for m in mlp_nodes:
        node_rows.append(
            {
                "node": f"MLP{m}",
                "node_type": "mlp",
                "layer": m,
                "depth": int(mlp_depth.get(m, 0)),
                "is_root": bool(m in root_set),
            }
        )
    for s in head_nodes:
        ss = str(s)
        l = int(ss.split("H")[0][1:])
        node_rows.append({"node": ss, "node_type": "head", "layer": l, "depth": -1, "is_root": False})
    nodes_df = pd.DataFrame(node_rows)

    # summary per key-mlp
    summary_rows = []
    for key in visited_order:
        res = key_results[key]
        hs = select_top_heads(res.head_rescue, key_mlp=key, k=8, min_rescue=args.min_head_rescue)
        ms = select_upstream_mlps(
            res.mlp_rescue,
            key_mlp=key,
            k=6,
            min_rescue=args.min_mlp_rescue,
            min_layer=args.min_upstream_layer,
        )
        summary_rows.append(
            {
                "key_mlp": key,
                "depth": res.depth,
                "top_heads": "; ".join([f"L{l}H{h}:{v:.4f}" for l, h, v in hs]),
                "top_upstream_mlps": "; ".join([f"MLP{m}:{v:.4f}" for m, v in ms]),
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values(["depth", "key_mlp"]).reset_index(drop=True)

    # save tables
    edges_csv = table_dir / f"{args.output_prefix}_edges.csv"
    nodes_csv = table_dir / f"{args.output_prefix}_nodes.csv"
    summary_csv = table_dir / f"{args.output_prefix}_summary.csv"
    edges_df.to_csv(edges_csv, index=False)
    nodes_df.to_csv(nodes_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    # key panels
    key_to_head = {k: key_results[k].head_rescue for k in sorted(key_results)}
    key_panels_png = fig_dir / f"{args.output_prefix}_key_panels.png"
    graph_png = fig_dir / f"{args.output_prefix}_circuit.png"
    plot_key_panels(key_to_head, key_panels_png, top_k_annot=5)
    plot_multihop_graph(graph_png, edges_df=edges_df, nodes_df=nodes_df, num_layers=num_layers)

    print("[done] multihop outputs:")
    print(f"- visited key mlps: {sorted(key_results.keys())}")
    print(f"- {edges_csv}")
    print(f"- {nodes_csv}")
    print(f"- {summary_csv}")
    print(f"- {key_panels_png}")
    print(f"- {graph_png}")


if __name__ == "__main__":
    main()
