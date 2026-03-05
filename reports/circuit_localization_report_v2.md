# Circuit Localization Report v2 (Significance + Paper-Style Figures)

## 1. Statistical Robustness (Bootstrap)
- clean hit rate: 87.84% [82.32%, 92.68%]
- corrupt non-tool rate: 82.99% [76.83%, 88.41%]
- pair success rate: 70.83% [64.02%, 78.05%]
- PD_base: 0.605861 [0.562437, 0.647070]
- PD_suff (best circuit): 0.464956 [0.427995, 0.500310]
- PD_nec (best circuit): -0.028764 [-0.035652, -0.022582]
- Recovery: 0.767585 [0.725885, 0.808674]
- Necessity drop: 0.634625 [0.591617, 0.674443]

## 2. Best Circuit (from iterative search)
- heads=32, mlps=4, recovery=0.7671, nec_drop=0.6342, score=0.8965

## 3. Top Component Significance
- L17H8: mean=0.217771, 95%CI=[0.193361, 0.241828], P_boot(<=0)=0.0000
- MLP27: mean=0.192467, 95%CI=[0.167632, 0.217299], P_boot(<=0)=0.0000
- L24H6: mean=0.191923, 95%CI=[0.166738, 0.218128], P_boot(<=0)=0.0000
- L21H12: mean=0.191338, 95%CI=[0.170191, 0.212531], P_boot(<=0)=0.0000
- L21H1: mean=0.189789, 95%CI=[0.166953, 0.211393], P_boot(<=0)=0.0000
- L20H5: mean=0.134580, 95%CI=[0.119366, 0.150826], P_boot(<=0)=0.0000
- L16H13: mean=0.109229, 95%CI=[0.093282, 0.126339], P_boot(<=0)=0.0000
- L17H2: mean=0.097329, 95%CI=[0.084861, 0.109699], P_boot(<=0)=0.0000
- L19H8: mean=0.082232, 95%CI=[0.071900, 0.093056], P_boot(<=0)=0.0000
- L23H6: mean=0.079800, 95%CI=[0.069299, 0.090168], P_boot(<=0)=0.0000
- L20H14: mean=0.077898, 95%CI=[0.069160, 0.086634], P_boot(<=0)=0.0000
- L18H15: mean=0.073108, 95%CI=[0.062502, 0.084462], P_boot(<=0)=0.0000
- MLP26: mean=0.068284, 95%CI=[0.059630, 0.077603], P_boot(<=0)=0.0000
- L16H5: mean=0.055791, 95%CI=[0.047354, 0.064584], P_boot(<=0)=0.0000
- MLP11: mean=0.054923, 95%CI=[0.045325, 0.064926], P_boot(<=0)=0.0000
- MLP22: mean=0.034924, 95%CI=[0.027315, 0.042669], P_boot(<=0)=0.0000
- MLP4: mean=0.029557, 95%CI=[0.024986, 0.034108], P_boot(<=0)=0.0000
- MLP3: mean=0.025919, 95%CI=[0.022297, 0.029584], P_boot(<=0)=0.0000

## 4. Figure Mapping to Paper
- `figs/probability_heatmap.png` aligns to paper `probability-heatmap` (organized by sorted sample axis and probability scale).
- `figs/probability_slice.png` aligns to paper `probability-slice` (line slice across ordered samples).
- `figs/logits_ipp.png` aligns to paper `logits-ipp` (layer x [heads+mlp], diverging centered at 0).
- `figs/patched_probability_heatmap.png` aligns to paper `patched-probability-heatmap` (intervention-conditioned probability matrix).
- `figs/top_components_ci.png` adds uncertainty quantification not shown in original paper but needed for statistical persuasiveness.

## 5. Further Iteration (v3)
- Multi-key-MLP panel figure: `figs/mlps_ipp_panels.png` (paper-style `mlps-ipp` organization).
- Upstream overlap matrix: `figs/mlps_ipp_overlap.png`.
- AP-vs-CT correlation figure: `figs/mlps_ipp_ap_ct_correlation.png`.
- Detailed note: `reports/circuit_localization_report_v3_note.md`.
