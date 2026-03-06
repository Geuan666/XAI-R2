#!/usr/bin/env python3
"""
Reverse path-patching (rich version):
- select a 2-MLP + 2-head reverse circuit from existing AP/CT tables
- run targeted path-patching metrics for multiple links
- render paper-style A/B/C triptych with richer circuit structure
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
class ReverseCircuit:
    key_mlp: int
    support_mlp: int
    head_key: Tuple[int, int]
    head_support: Tuple[int, int]
    key_mlp_ap: float
    support_mlp_ap: float
    support_mlp_ct_to_key: float
    head_key_ap: float
    head_key_ct_to_key: float
    head_support_ct_to_support: float


def calc_tool_prob(last_logits: torch.Tensor, tool_id: int) -> torch.Tensor:
    return F.softmax(last_logits, dim=-1)[:, tool_id]


def load_tables(table_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    head_df = pd.read_csv(table_dir / "ap_ct_head_scores.csv")
    mlp_df = pd.read_csv(table_dir / "ap_ct_mlp_scores.csv")
    base_df = pd.read_csv(table_dir / "baseline_metrics.csv").sort_values("q").reset_index(drop=True)

    top_heads_path = table_dir / "mlps_ipp_top_heads.csv"
    top_heads_df: Optional[pd.DataFrame] = None
    if top_heads_path.exists():
        top_heads_df = pd.read_csv(top_heads_path)
    return head_df, mlp_df, base_df, top_heads_df


def select_reverse_circuit(
    head_df: pd.DataFrame,
    mlp_df: pd.DataFrame,
    top_heads_df: Optional[pd.DataFrame],
    force_key_mlp: Optional[int],
    force_support_mlp: Optional[int],
) -> ReverseCircuit:
    head_df = head_df.sort_values(["ap_impact_pd", "ct_rescue_pd"], ascending=[False, False]).reset_index(drop=True)
    mlp_df = mlp_df.sort_values("ap_impact_pd", ascending=False).reset_index(drop=True)

    if force_key_mlp is None:
        key_row = mlp_df.iloc[0]
    else:
        sel = mlp_df[mlp_df["layer"] == force_key_mlp]
        if len(sel) == 0:
            raise ValueError(f"force key mlp {force_key_mlp} not found")
        key_row = sel.iloc[0]
    key_mlp = int(key_row["layer"])

    # support MLP: positive CT-to-key and decent AP. prefer layers below key.
    support_cands = mlp_df[
        (mlp_df["layer"] != key_mlp) & (mlp_df["ct_rescue_pd"] > 0) & (mlp_df["layer"] < key_mlp)
    ].copy()
    if len(support_cands) == 0:
        support_cands = mlp_df[(mlp_df["layer"] != key_mlp) & (mlp_df["layer"] < key_mlp)].copy()
    support_cands["support_score"] = 0.6 * support_cands["ap_impact_pd"] + 0.4 * np.maximum(
        0.0, support_cands["ct_rescue_pd"]
    )

    if force_support_mlp is not None:
        sel = support_cands[support_cands["layer"] == force_support_mlp]
        if len(sel) == 0:
            raise ValueError(f"force support mlp {force_support_mlp} not found/invalid")
        support_row = sel.iloc[0]
    else:
        support_row = support_cands.sort_values("support_score", ascending=False).iloc[0]
    support_mlp = int(support_row["layer"])

    # head via key_mlp: use AP+CT key-conditioned score
    hk = head_df[(head_df["ct_rescue_pd"] > 0) & (head_df["layer"] <= key_mlp)].copy()
    if len(hk) == 0:
        hk = head_df.copy()
    hk["rank_score"] = 0.55 * hk["ct_rescue_pd"] + 0.45 * hk["ap_impact_pd"]
    hk = hk.sort_values(["rank_score", "ct_rescue_pd", "ap_impact_pd"], ascending=False)
    hk_row = hk.iloc[0]
    head_key = (int(hk_row["layer"]), int(hk_row["head"]))

    # head via support mlp: if we have multi-key table, use it; fallback to same head ranking.
    if top_heads_df is not None and "key_mlp" in top_heads_df.columns:
        hs_cands = top_heads_df[top_heads_df["key_mlp"] == support_mlp].copy()
        if len(hs_cands) > 0:
            hs_row = hs_cands.sort_values("ct_rescue", ascending=False).iloc[0]
            head_support = (int(hs_row["layer"]), int(hs_row["head"]))
            hs_ct = float(hs_row["ct_rescue"])
        else:
            hs_row = hk.iloc[1] if len(hk) > 1 else hk.iloc[0]
            head_support = (int(hs_row["layer"]), int(hs_row["head"]))
            hs_ct = float(hs_row["ct_rescue_pd"])
    else:
        hs_row = hk.iloc[1] if len(hk) > 1 else hk.iloc[0]
        head_support = (int(hs_row["layer"]), int(hs_row["head"]))
        hs_ct = float(hs_row["ct_rescue_pd"])

    return ReverseCircuit(
        key_mlp=key_mlp,
        support_mlp=support_mlp,
        head_key=head_key,
        head_support=head_support,
        key_mlp_ap=float(key_row["ap_impact_pd"]),
        support_mlp_ap=float(support_row["ap_impact_pd"]),
        support_mlp_ct_to_key=float(support_row["ct_rescue_pd"]),
        head_key_ap=float(hk_row["ap_impact_pd"]),
        head_key_ct_to_key=float(hk_row["ct_rescue_pd"]),
        head_support_ct_to_support=hs_ct,
    )


def compute_pd_intervention(
    model: Any,
    clean_tok: Dict[str, torch.Tensor],
    corr_tok: Dict[str, torch.Tensor],
    p_corr_ref: np.ndarray,
    patch_heads: Sequence[Tuple[int, int]],
    patch_mlps: Sequence[int],
    restore_mlps: Sequence[int],
    tool_id: int,
    batch_size: int,
    device: torch.device,
    num_layers: int,
    head_dim: int,
) -> float:
    n = clean_tok["input_ids"].shape[0]
    p_out = np.zeros(n, dtype=np.float64)

    patch_head_layers = sorted(set([l for l, _ in patch_heads]))
    patch_mlp_layers = sorted(set([int(x) for x in patch_mlps]))
    restore_mlp_layers = sorted(set([int(x) for x in restore_mlps]))

    with torch.no_grad():
        for s, e in batched_indices(n, batch_size):
            cb_cpu = trim_batch_to_nonpad(slice_batch(clean_tok, s, e))
            xb_cpu = trim_batch_to_nonpad(slice_batch(corr_tok, s, e))
            cb = move_batch(cb_cpu, device)
            xb = move_batch(xb_cpu, device)

            pos = cb["attention_mask"].sum(dim=1) - 1
            pos_corr = xb["attention_mask"].sum(dim=1) - 1
            if not torch.equal(pos, pos_corr):
                raise RuntimeError("clean/corrupt position mismatch")

            # capture all needed corrupt activations
            hcap, corr_head_cache, corr_mlp_cache = register_capture_hooks_for_all_layers(
                model,
                num_layers=num_layers,
                capture_heads=len(patch_heads) > 0,
                capture_mlps=len(patch_mlp_layers) > 0,
            )
            _ = forward_model(model, True, **xb)
            remove_handles(hcap)

            clean_mlp_cache: Dict[int, torch.Tensor] = {}
            if len(restore_mlp_layers) > 0:
                h_clean = []
                for l in restore_mlp_layers:
                    def make_capture(layer_idx: int):
                        def _capture(module: Any, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor) -> torch.Tensor:
                            clean_mlp_cache[layer_idx] = output.detach()
                            return output
                        return _capture

                    h_clean.append(model.model.layers[l].mlp.register_forward_hook(make_capture(l)))
                _ = forward_model(model, True, **cb)
                remove_handles(h_clean)

            hooks = []
            # patch heads first
            for l, h in patch_heads:
                corr_last = gather_last_hidden(corr_head_cache[l], pos)
                hs = h * head_dim
                he = hs + head_dim

                def make_head_patch(corr_last_layer: torch.Tensor, hs_: int, he_: int):
                    def _hook(module: Any, inputs: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor]:
                        x = inputs[0].clone()
                        bidx = torch.arange(x.shape[0], device=x.device)
                        x[bidx, pos, hs_:he_] = corr_last_layer[:, hs_:he_]
                        return (x,)
                    return _hook

                hooks.append(
                    model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(
                        make_head_patch(corr_last, hs, he)
                    )
                )

            # patch mlps with corrupt
            for l in patch_mlp_layers:
                corr_last = gather_last_hidden(corr_mlp_cache[l], pos)
                hooks.append(
                    model.model.layers[l].mlp.register_forward_hook(
                        make_mlp_patch_hook(corr_last_layer=corr_last, positions=pos)
                    )
                )

            # restore mlps with clean (registered last -> override patch if same layer)
            for l in restore_mlp_layers:
                clean_last = gather_last_hidden(clean_mlp_cache[l], pos)
                hooks.append(
                    model.model.layers[l].mlp.register_forward_hook(
                        make_mlp_restore_hook(clean_last_layer=clean_last, positions=pos)
                    )
                )

            logits = forward_model(model, True, **cb).logits
            remove_handles(hooks)

            last, _ = gather_last_logits(logits, cb["attention_mask"])
            p_out[s:e] = calc_tool_prob(last, tool_id).float().detach().cpu().numpy()

    return float(np.mean(p_out - p_corr_ref))


def _node(ax: plt.Axes, center: Tuple[float, float], label: str, w: float = 0.42, h: float = 0.14) -> None:
    x, y = center
    rect = Rectangle((x - w / 2, y - h / 2), w, h, linewidth=1.0, edgecolor="#9e9e9e", facecolor="#efefef", zorder=2)
    ax.add_patch(rect)
    ax.text(x, y, label, ha="center", va="center", fontsize=9.8, zorder=3)


def _arrow(
    ax: plt.Axes,
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    color: str,
    lw: float = 1.2,
    ls: str = "-",
    alpha: float = 1.0,
    rad: float = 0.0,
) -> None:
    a = FancyArrowPatch(
        p1,
        p2,
        arrowstyle="-|>",
        mutation_scale=10,
        linewidth=lw,
        linestyle=ls,
        color=color,
        alpha=alpha,
        connectionstyle=f"arc3,rad={rad}",
        zorder=1,
    )
    ax.add_patch(a)


def _panel_base(
    ax: plt.Axes,
    x0: float,
    attn_low: str,
    attn_high: str,
    mlp_low: str,
    mlp_high: str,
    input_label: str,
    normal_color: str,
    draw_high_to_logits: bool = True,
) -> Dict[str, Tuple[float, float]]:
    pos = {
        "logits": (x0 + 0.45, 0.88),
        "attn_high": (x0 + 0.17, 0.63),
        "attn_low": (x0 + 0.17, 0.41),
        "mlp_high": (x0 + 0.73, 0.63),
        "mlp_low": (x0 + 0.73, 0.41),
        "embed": (x0 + 0.45, 0.20),
        "input": (x0 + 0.45, 0.02),
    }
    _node(ax, pos["logits"], "Logits")
    _node(ax, pos["attn_high"], attn_high)
    _node(ax, pos["attn_low"], attn_low)
    _node(ax, pos["mlp_high"], mlp_high)
    _node(ax, pos["mlp_low"], mlp_low)
    _node(ax, pos["embed"], "Token Embeddings\n+ Earlier Layers", w=0.60, h=0.15)
    _node(ax, pos["input"], input_label, w=0.64, h=0.15)

    _arrow(ax, pos["input"], pos["embed"], normal_color, lw=1.2)
    _arrow(ax, pos["embed"], pos["attn_low"], normal_color, lw=1.0)
    _arrow(ax, pos["embed"], pos["attn_high"], normal_color, lw=1.0)
    _arrow(ax, pos["embed"], pos["mlp_low"], normal_color, lw=1.0)
    _arrow(ax, pos["embed"], pos["mlp_high"], normal_color, lw=1.0)
    _arrow(ax, pos["embed"], pos["logits"], normal_color, lw=1.0)
    _arrow(ax, pos["attn_low"], pos["attn_high"], normal_color, lw=1.0)
    _arrow(ax, pos["attn_low"], pos["mlp_low"], normal_color, lw=1.0)
    _arrow(ax, pos["attn_low"], pos["mlp_high"], normal_color, lw=1.0)
    _arrow(ax, pos["attn_low"], pos["logits"], normal_color, lw=1.0, rad=0.03)
    _arrow(ax, pos["attn_high"], pos["mlp_low"], normal_color, lw=1.0)
    _arrow(ax, pos["attn_high"], pos["mlp_high"], normal_color, lw=1.0)
    _arrow(ax, pos["attn_high"], pos["logits"], normal_color, lw=1.0, rad=-0.03)
    _arrow(ax, pos["mlp_low"], pos["mlp_high"], normal_color, lw=1.0)
    _arrow(ax, pos["mlp_low"], pos["logits"], normal_color, lw=1.0, rad=0.05)
    if draw_high_to_logits:
        _arrow(ax, pos["mlp_high"], pos["logits"], normal_color, lw=1.0, rad=-0.05)
    return pos


def draw_triptych_rich(
    out_path: Path,
    circuit: ReverseCircuit,
    pd_base: float,
    pd_mlp_key_patch: float,
    pd_head_key_patch: float,
    pd_head_key_restore_key: float,
    pd_head_support_patch: float,
    pd_head_support_restore_support: float,
    pd_head_support_restore_chain: float,
) -> None:
    normal = "#1298ad"
    corrupt = "#e53935"
    both = "#8a3ffc"

    attn_high = f"Attention\nLayer {circuit.head_key[0]}"
    attn_low = f"Attention\nLayer {circuit.head_support[0]}"
    mlp_high = f"MLP\n{circuit.key_mlp}"
    mlp_low = f"MLP\n{circuit.support_mlp}"

    fig, ax = plt.subplots(figsize=(16.2, 8.0), dpi=220)
    ax.set_xlim(0.0, 5.45)
    ax.set_ylim(-0.30, 0.98)
    ax.axis("off")

    xA, xB, xC = 0.00, 1.75, 3.50

    pA = _panel_base(
        ax,
        xA,
        attn_low=attn_low,
        attn_high=attn_high,
        mlp_low=mlp_low,
        mlp_high=mlp_high,
        input_label="Clean input\n(pair prompt)",
        normal_color=normal,
        draw_high_to_logits=True,
    )
    ax.text(xA + 0.00, 0.95, "A", fontsize=20, fontweight="bold", va="top")

    pB = _panel_base(
        ax,
        xB,
        attn_low=attn_low,
        attn_high=attn_high,
        mlp_low=mlp_low,
        mlp_high=mlp_high,
        input_label="Clean input\n(pair prompt)",
        normal_color=normal,
        draw_high_to_logits=False,
    )
    ax.text(xB + 0.00, 0.95, "B", fontsize=20, fontweight="bold", va="top")

    b_corr = {
        "mlp_high": (xB + 1.10, 0.63),
        "embed": (xB + 1.10, 0.20),
        "input": (xB + 1.10, 0.02),
    }
    _node(ax, b_corr["mlp_high"], mlp_high)
    _node(ax, b_corr["embed"], "Token Embeddings\n+ Earlier Layers", w=0.60, h=0.15)
    _node(ax, b_corr["input"], "Corrupt input\n(pair prompt)", w=0.64, h=0.15)
    _arrow(ax, b_corr["input"], b_corr["embed"], corrupt, lw=1.4, ls=(0, (1.2, 3.0)))
    _arrow(ax, b_corr["embed"], b_corr["mlp_high"], corrupt, lw=1.4, ls=(0, (1.2, 3.0)))
    _arrow(ax, b_corr["mlp_high"], pB["logits"], corrupt, lw=1.5, ls=(0, (1.2, 3.0)))

    pC = _panel_base(
        ax,
        xC,
        attn_low=attn_low,
        attn_high=attn_high,
        mlp_low=mlp_low,
        mlp_high=mlp_high,
        input_label="Clean input\n(pair prompt)",
        normal_color=normal,
        draw_high_to_logits=False,
    )
    ax.text(xC + 0.00, 0.95, "C", fontsize=20, fontweight="bold", va="top")

    c_corr = {
        "attn_high": (xC + 1.00, 0.63),
        "attn_low": (xC + 1.00, 0.41),
        "mlp_low": (xC + 1.32, 0.41),
        "mlp_high": (xC + 1.32, 0.63),
        "embed": (xC + 1.16, 0.20),
        "input": (xC + 1.16, 0.02),
    }
    _node(ax, c_corr["attn_high"], f"Head\nL{circuit.head_key[0]}H{circuit.head_key[1]}", w=0.36)
    _node(ax, c_corr["attn_low"], f"Head\nL{circuit.head_support[0]}H{circuit.head_support[1]}", w=0.36)
    _node(ax, c_corr["mlp_low"], mlp_low)
    _node(ax, c_corr["mlp_high"], mlp_high)
    _node(ax, c_corr["embed"], "Token Embeddings\n+ Earlier Layers", w=0.60, h=0.15)
    _node(ax, c_corr["input"], "Corrupt input\n(pair prompt)", w=0.64, h=0.15)

    _arrow(ax, c_corr["input"], c_corr["embed"], corrupt, lw=1.4, ls=(0, (1.2, 3.0)))
    _arrow(ax, c_corr["embed"], c_corr["attn_high"], corrupt, lw=1.4, ls=(0, (1.2, 3.0)))
    _arrow(ax, c_corr["embed"], c_corr["attn_low"], corrupt, lw=1.4, ls=(0, (1.2, 3.0)))
    _arrow(ax, c_corr["attn_low"], c_corr["mlp_low"], corrupt, lw=1.4, ls=(0, (1.2, 3.0)))
    _arrow(ax, c_corr["attn_high"], c_corr["mlp_high"], corrupt, lw=1.4, ls=(0, (1.2, 3.0)))
    _arrow(ax, c_corr["mlp_low"], c_corr["mlp_high"], corrupt, lw=1.4, ls=(0, (1.2, 3.0)))
    _arrow(ax, c_corr["mlp_high"], pC["logits"], both, lw=1.7, ls=(0, (4.0, 2.4)))
    _arrow(ax, pC["embed"], c_corr["mlp_high"], normal, lw=1.2)

    lx, ly = 5.02, 0.84
    _arrow(ax, (lx - 0.10, ly), (lx + 0.10, ly), normal, lw=1.7)
    ax.text(lx + 0.13, ly, "Normal Input", va="center", fontsize=11.5)
    _arrow(ax, (lx - 0.10, ly - 0.07), (lx + 0.10, ly - 0.07), corrupt, lw=1.7, ls=(0, (1.2, 3.0)))
    ax.text(lx + 0.13, ly - 0.07, "Corrupt Input", va="center", fontsize=11.5)
    _arrow(ax, (lx - 0.10, ly - 0.14), (lx + 0.10, ly - 0.14), both, lw=1.8, ls=(0, (4.0, 2.4)))
    ax.text(lx + 0.13, ly - 0.14, "Both Inputs", va="center", fontsize=11.5)

    cap_l1 = (
        "Reverse circuit (rich) for <tool_call> "
        f"(selected: L{circuit.head_key[0]}H{circuit.head_key[1]}, "
        f"L{circuit.head_support[0]}H{circuit.head_support[1]}, "
        f"MLP{circuit.support_mlp}, MLP{circuit.key_mlp})."
    )
    cap_l2 = (
        f"A: unpatched PD={pd_base:.4f}.  "
        f"B: patch (MLP{circuit.key_mlp}, logits) -> corrupt, PD={pd_mlp_key_patch:.4f}.  "
        f"C: head-key patch PD={pd_head_key_patch:.4f}, +restore MLP{circuit.key_mlp} -> {pd_head_key_restore_key:.4f}."
    )
    cap_l3 = (
        f"Support-head patch PD={pd_head_support_patch:.4f}, +restore MLP{circuit.support_mlp} -> {pd_head_support_restore_support:.4f}, "
        f"+restore chain (MLP{circuit.support_mlp}+MLP{circuit.key_mlp}) -> {pd_head_support_restore_chain:.4f}."
    )
    ax.text(0.02, -0.215, cap_l1 + "\n" + cap_l2 + "\n" + cap_l3, fontsize=10.0, ha="left", va="top")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


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
    parser.add_argument("--force-support-mlp", type=int, default=None)
    args = parser.parse_args()

    set_seed(args.seed)

    pair_dir = Path(args.pair_dir)
    table_dir = Path(args.table_dir)
    fig_dir = Path(args.fig_dir)
    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    head_df, mlp_df, base_df, top_heads_df = load_tables(table_dir)
    circuit = select_reverse_circuit(
        head_df=head_df,
        mlp_df=mlp_df,
        top_heads_df=top_heads_df,
        force_key_mlp=args.force_key_mlp,
        force_support_mlp=args.force_support_mlp,
    )

    print(
        f"[select] key_mlp=MLP{circuit.key_mlp}, support_mlp=MLP{circuit.support_mlp}, "
        f"head_key=L{circuit.head_key[0]}H{circuit.head_key[1]}, "
        f"head_support=L{circuit.head_support[0]}H{circuit.head_support[1]}"
    )

    samples = load_pair_samples(pair_dir)
    if len(samples) != len(base_df):
        raise RuntimeError(
            f"sample count mismatch: pair={len(samples)} baseline={len(base_df)}. "
            "Please use matched subset."
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

    num_layers = int(model.config.num_hidden_layers)
    num_heads = int(model.config.num_attention_heads)
    hidden_size = int(model.config.hidden_size)
    head_dim = hidden_size // num_heads

    tool_ids = tokenizer.encode("<tool_call>", add_special_tokens=False)
    if len(tool_ids) != 1:
        raise RuntimeError(f"<tool_call> not single token: {tool_ids}")
    tool_id = int(tool_ids[0])

    clean_texts = [s.clean_text for s in samples]
    corr_texts = [s.corrupt_text for s in samples]
    clean_tok, corr_tok = tokenize_all(tokenizer, clean_texts, corr_texts)

    p_corr_ref = base_df["p_corr"].to_numpy(dtype=np.float64)
    pd_base = float((base_df["p_clean"] - base_df["p_corr"]).mean())

    # targeted interventions
    pd_mlp_key_patch = compute_pd_intervention(
        model=model,
        clean_tok=clean_tok,
        corr_tok=corr_tok,
        p_corr_ref=p_corr_ref,
        patch_heads=[],
        patch_mlps=[circuit.key_mlp],
        restore_mlps=[],
        tool_id=tool_id,
        batch_size=args.batch_size,
        device=device,
        num_layers=num_layers,
        head_dim=head_dim,
    )
    pd_head_key_patch = compute_pd_intervention(
        model=model,
        clean_tok=clean_tok,
        corr_tok=corr_tok,
        p_corr_ref=p_corr_ref,
        patch_heads=[circuit.head_key],
        patch_mlps=[],
        restore_mlps=[],
        tool_id=tool_id,
        batch_size=args.batch_size,
        device=device,
        num_layers=num_layers,
        head_dim=head_dim,
    )
    pd_head_key_restore_key = compute_pd_intervention(
        model=model,
        clean_tok=clean_tok,
        corr_tok=corr_tok,
        p_corr_ref=p_corr_ref,
        patch_heads=[circuit.head_key],
        patch_mlps=[],
        restore_mlps=[circuit.key_mlp],
        tool_id=tool_id,
        batch_size=args.batch_size,
        device=device,
        num_layers=num_layers,
        head_dim=head_dim,
    )
    pd_head_support_patch = compute_pd_intervention(
        model=model,
        clean_tok=clean_tok,
        corr_tok=corr_tok,
        p_corr_ref=p_corr_ref,
        patch_heads=[circuit.head_support],
        patch_mlps=[],
        restore_mlps=[],
        tool_id=tool_id,
        batch_size=args.batch_size,
        device=device,
        num_layers=num_layers,
        head_dim=head_dim,
    )
    pd_head_support_restore_support = compute_pd_intervention(
        model=model,
        clean_tok=clean_tok,
        corr_tok=corr_tok,
        p_corr_ref=p_corr_ref,
        patch_heads=[circuit.head_support],
        patch_mlps=[],
        restore_mlps=[circuit.support_mlp],
        tool_id=tool_id,
        batch_size=args.batch_size,
        device=device,
        num_layers=num_layers,
        head_dim=head_dim,
    )
    pd_head_support_restore_chain = compute_pd_intervention(
        model=model,
        clean_tok=clean_tok,
        corr_tok=corr_tok,
        p_corr_ref=p_corr_ref,
        patch_heads=[circuit.head_support],
        patch_mlps=[],
        restore_mlps=[circuit.support_mlp, circuit.key_mlp],
        tool_id=tool_id,
        batch_size=args.batch_size,
        device=device,
        num_layers=num_layers,
        head_dim=head_dim,
    )

    rows = [
        {"metric": "pd_base", "value": pd_base},
        {"metric": f"pd_patch_MLP{circuit.key_mlp}_to_logits", "value": pd_mlp_key_patch},
        {"metric": f"pd_patch_L{circuit.head_key[0]}H{circuit.head_key[1]}", "value": pd_head_key_patch},
        {
            "metric": f"pd_patch_L{circuit.head_key[0]}H{circuit.head_key[1]}_restore_MLP{circuit.key_mlp}",
            "value": pd_head_key_restore_key,
        },
        {"metric": f"pd_patch_L{circuit.head_support[0]}H{circuit.head_support[1]}", "value": pd_head_support_patch},
        {
            "metric": f"pd_patch_L{circuit.head_support[0]}H{circuit.head_support[1]}_restore_MLP{circuit.support_mlp}",
            "value": pd_head_support_restore_support,
        },
        {
            "metric": (
                f"pd_patch_L{circuit.head_support[0]}H{circuit.head_support[1]}_"
                f"restore_MLP{circuit.support_mlp}_MLP{circuit.key_mlp}"
            ),
            "value": pd_head_support_restore_chain,
        },
    ]
    metric_df = pd.DataFrame(rows)
    metric_df.to_csv(table_dir / "reverse_rich_path_metrics.csv", index=False)

    # compact interpreted deltas
    delta_rows = [
        {"quantity": "ap_drop_key_mlp", "value": pd_base - pd_mlp_key_patch},
        {"quantity": "ap_drop_head_key", "value": pd_base - pd_head_key_patch},
        {"quantity": "ct_rescue_head_key_via_key_mlp", "value": pd_head_key_restore_key - pd_head_key_patch},
        {"quantity": "ap_drop_head_support", "value": pd_base - pd_head_support_patch},
        {
            "quantity": "ct_rescue_head_support_via_support_mlp",
            "value": pd_head_support_restore_support - pd_head_support_patch,
        },
        {
            "quantity": "ct_rescue_head_support_via_support_plus_key",
            "value": pd_head_support_restore_chain - pd_head_support_patch,
        },
    ]
    delta_df = pd.DataFrame(delta_rows)
    delta_df.to_csv(table_dir / "reverse_rich_path_deltas.csv", index=False)

    sel_df = pd.DataFrame(
        [
            {
                "key_mlp": circuit.key_mlp,
                "support_mlp": circuit.support_mlp,
                "head_key_layer": circuit.head_key[0],
                "head_key_idx": circuit.head_key[1],
                "head_support_layer": circuit.head_support[0],
                "head_support_idx": circuit.head_support[1],
                "table_key_mlp_ap": circuit.key_mlp_ap,
                "table_support_mlp_ap": circuit.support_mlp_ap,
                "table_support_mlp_ct_to_key": circuit.support_mlp_ct_to_key,
                "table_head_key_ap": circuit.head_key_ap,
                "table_head_key_ct_to_key": circuit.head_key_ct_to_key,
                "table_head_support_ct_to_support": circuit.head_support_ct_to_support,
            }
        ]
    )
    sel_df.to_csv(table_dir / "reverse_rich_selected_circuit.csv", index=False)

    draw_triptych_rich(
        out_path=fig_dir / "reverse_circuit_triptych_rich.png",
        circuit=circuit,
        pd_base=pd_base,
        pd_mlp_key_patch=pd_mlp_key_patch,
        pd_head_key_patch=pd_head_key_patch,
        pd_head_key_restore_key=pd_head_key_restore_key,
        pd_head_support_patch=pd_head_support_patch,
        pd_head_support_restore_support=pd_head_support_restore_support,
        pd_head_support_restore_chain=pd_head_support_restore_chain,
    )

    print("[done] rich reverse outputs:")
    print(f"- {table_dir / 'reverse_rich_selected_circuit.csv'}")
    print(f"- {table_dir / 'reverse_rich_path_metrics.csv'}")
    print(f"- {table_dir / 'reverse_rich_path_deltas.csv'}")
    print(f"- {fig_dir / 'reverse_circuit_triptych_rich.png'}")


if __name__ == "__main__":
    main()
