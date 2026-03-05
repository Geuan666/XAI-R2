# Circuit Localization Report: Qwen3-1.7B `<tool_call>` First Token

## 1. Experiment Setup
- Model: `/root/data/Qwen/Qwen3-1.7B`
- Sample size: 164 paired clean/corrupt prompts
- Batch size: 1
- Precision: `float16`
- Seed: 123
- Logits mode: `logits_to_keep=1` (memory-safe; exact last-token evaluation under batch=1)
- Decision position: assistant first-token prediction (logits at prompt last token)

## 2. Data Integrity
- Total pairs: 164
- Length-aligned pairs: 164 (100.0%)
- `corr_len - clean_len` unique values: [0]

## 3. Baseline Behavior
- Clean top-1 is `<tool_call>`: 87.8%
- Corrupt top-1 is non-`<tool_call>`: 82.9%
- Pair simultaneously satisfies both: 70.7%
- `PD_tool = mean(p_clean - p_corr)`: 0.605372
- `PairSignAcc`: 0.707317
- `MarginSep`: 4.898819
- Reference quick baseline hint (todo): clean=89.6%, corrupt_non=83.5%, pair=73.2%

## 4. AP (Direct to Logits) Findings
Top attention heads by AP impact (`Impact = PD_unpatched - PD_patched`):
- L17H8: impact=0.216635, ct_rescue=0.091230
- L24H6: impact=0.190929, ct_rescue=0.113986
- L21H12: impact=0.189301, ct_rescue=0.084102
- L21H1: impact=0.187903, ct_rescue=0.099647
- L20H5: impact=0.132976, ct_rescue=0.065410
- L16H13: impact=0.107389, ct_rescue=0.048572
- L17H2: impact=0.095159, ct_rescue=0.041062
- L19H8: impact=0.080153, ct_rescue=0.037212
- L23H6: impact=0.077631, ct_rescue=0.041604
- L20H14: impact=0.075424, ct_rescue=0.026127
- L18H15: impact=0.070722, ct_rescue=0.017678
- L16H5: impact=0.054290, ct_rescue=0.012116

Top MLPs by AP impact:
- MLP27: impact=0.192467, ct_rescue=0.192467
- MLP26: impact=0.068284, ct_rescue=-0.049124
- MLP11: impact=0.054923, ct_rescue=-0.005127
- MLP22: impact=0.034924, ct_rescue=0.021026
- MLP4: impact=0.029557, ct_rescue=0.018723
- MLP3: impact=0.025919, ct_rescue=0.012197
- MLP8: impact=0.021635, ct_rescue=0.025188
- MLP20: impact=0.019758, ct_rescue=-0.019543
- MLP5: impact=0.015628, ct_rescue=0.013287
- MLP0: impact=0.007646, ct_rescue=0.004305
- Key downstream MLP for conditioned tracing: `MLP27`

## 5. Sufficiency / Necessity Search
Candidate sets (iterative):

|   num_heads |   num_mlps |   pd_suff |      pd_nec |   recovery |   nec_drop |   pair_suff |   pair_nec |    score |
|------------:|-----------:|----------:|------------:|-----------:|-----------:|------------:|-----------:|---------:|
|          32 |          4 | 0.464407  | -0.0288118  |  0.767144  |   0.634184 |  0.615854   | 0          | 0.896468 |
|          32 |          6 | 0.464407  | -0.027087   |  0.767144  |   0.632459 |  0.615854   | 0          | 0.895248 |
|          32 |          3 | 0.427983  | -0.0329175  |  0.706976  |   0.638289 |  0.585366   | 0          | 0.863376 |
|          32 |          2 | 0.423576  | -0.0240346  |  0.699696  |   0.629406 |  0.573171   | 0          | 0.852921 |
|          24 |          4 | 0.341769  | -0.0256791  |  0.56456   |   0.631051 |  0.45122    | 0          | 0.767143 |
|          24 |          6 | 0.341769  | -0.0225168  |  0.56456   |   0.627889 |  0.45122    | 0          | 0.765218 |
|          24 |          3 | 0.30828   | -0.0286067  |  0.509241  |   0.633979 |  0.426829   | 0          | 0.730277 |
|          24 |          2 | 0.30828   | -0.0225223  |  0.509241  |   0.627894 |  0.426829   | 0          | 0.726764 |
|          16 |          4 | 0.185674  | -0.029219   |  0.306711  |   0.634591 |  0.237805   | 0          | 0.567023 |
|          16 |          6 | 0.185674  | -0.0280411  |  0.306711  |   0.633413 |  0.237805   | 0          | 0.566496 |
|          16 |          3 | 0.158874  | -0.0328359  |  0.26244   |   0.638208 |  0.20122    | 0          | 0.525999 |
|          16 |          2 | 0.158874  | -0.0285994  |  0.26244   |   0.633971 |  0.20122    | 0          | 0.524251 |
|          12 |          4 | 0.140148  | -0.0265564  |  0.231508  |   0.631928 |  0.176829   | 0          | 0.491593 |
|          12 |          6 | 0.140148  | -0.0250581  |  0.231508  |   0.63043  |  0.176829   | 0          | 0.49101  |
|          12 |          3 | 0.119321  | -0.0324292  |  0.197104  |   0.637801 |  0.140244   | 0          | 0.4557   |
|          12 |          2 | 0.119321  | -0.0263735  |  0.197104  |   0.631745 |  0.140244   | 0          | 0.453531 |
|           8 |          6 | 0.0982979 | -0.00293485 |  0.162376  |   0.608307 |  0.121951   | 0          | 0.403935 |
|           8 |          4 | 0.0982979 | -0.00202582 |  0.162376  |   0.607398 |  0.121951   | 0          | 0.403633 |
|           8 |          3 | 0.0821662 | -0.00660099 |  0.135728  |   0.611973 |  0.097561   | 0          | 0.370417 |
|           8 |          2 | 0.0821662 |  0.0020385  |  0.135728  |   0.603333 |  0.097561   | 0          | 0.367793 |
|           4 |          6 | 0.0135736 |  0.0252134  |  0.0224219 |   0.580158 |  0.00609756 | 0.0365854  | 0.146588 |
|           4 |          4 | 0.0135736 |  0.0280648  |  0.0224219 |   0.577307 |  0.00609756 | 0.0365854  | 0.146227 |
|           4 |          3 | 0.010593  |  0.0240789  |  0.0174984 |   0.581293 |  0.00609756 | 0.00609756 | 0.129624 |
|           4 |          2 | 0.010593  |  0.033759   |  0.0174984 |   0.571613 |  0.00609756 | 0.0609756  | 0.12854  |

Best set:
- heads=32, mlps=4, recovery=0.7671, nec_drop=0.6342, score=0.8965

## 6. Final Candidate Circuit
Selected nodes:
- L24H6
- L17H8
- L21H1
- L21H12
- L20H5
- L16H13
- L17H2
- L23H6
- L19H8
- L20H14
- L18H15
- L16H7
- L18H7
- L15H5
- L16H5
- L15H9
- L17H11
- L19H5
- L16H4
- L26H0
- L13H13
- L13H7
- L12H13
- L23H5
- L19H13
- L11H9
- L25H13
- L11H1
- L19H6
- L15H15
- L15H4
- L26H10
- MLP11
- MLP22
- MLP26
- MLP27

Selected edges (with causal evidence scores):
- L11H1 -> MLP27 (ct_rescue=0.013645)
- L11H9 -> MLP27 (ct_rescue=0.007842)
- L12H13 -> MLP27 (ct_rescue=0.010190)
- L13H7 -> MLP27 (ct_rescue=0.008726)
- L13H13 -> MLP27 (ct_rescue=0.011610)
- L15H4 -> MLP27 (ct_rescue=0.003748)
- L15H5 -> MLP27 (ct_rescue=0.019054)
- L15H9 -> MLP27 (ct_rescue=0.012246)
- L15H15 -> MLP27 (ct_rescue=0.008066)
- L16H4 -> MLP27 (ct_rescue=0.013870)
- L16H5 -> MLP27 (ct_rescue=0.012116)
- L16H7 -> MLP27 (ct_rescue=0.023380)
- L16H13 -> MLP27 (ct_rescue=0.048572)
- L17H2 -> MLP27 (ct_rescue=0.041062)
- L17H8 -> MLP27 (ct_rescue=0.091230)
- L17H11 -> MLP27 (ct_rescue=0.013632)
- L18H7 -> MLP27 (ct_rescue=0.017723)
- L18H15 -> MLP27 (ct_rescue=0.017678)
- L19H5 -> MLP27 (ct_rescue=0.016014)
- L19H6 -> MLP27 (ct_rescue=0.005764)
- L19H8 -> MLP27 (ct_rescue=0.037212)
- L19H13 -> MLP27 (ct_rescue=0.008868)
- L20H5 -> MLP27 (ct_rescue=0.065410)
- L20H14 -> MLP27 (ct_rescue=0.026127)
- L21H1 -> MLP27 (ct_rescue=0.099647)
- L21H12 -> MLP27 (ct_rescue=0.084102)
- L23H5 -> MLP27 (ct_rescue=0.016370)
- L23H6 -> MLP27 (ct_rescue=0.041604)
- L24H6 -> MLP27 (ct_rescue=0.113986)
- L25H13 -> MLP27 (ct_rescue=0.010560)
- L26H0 -> MLP27 (ct_rescue=0.020488)
- L26H10 -> MLP27 (ct_rescue=0.021697)
- MLP11 -> LOGITS(<tool_call>) (ap_direct=0.054923)
- MLP22 -> LOGITS(<tool_call>) (ap_direct=0.034924)
- MLP26 -> LOGITS(<tool_call>) (ap_direct=0.068284)
- MLP27 -> LOGITS(<tool_call>) (ap_direct=0.192467)
- L17H8 -> LOGITS(<tool_call>) (ap_direct_weak=0.216635)
- L24H6 -> LOGITS(<tool_call>) (ap_direct_weak=0.190929)
- L21H12 -> LOGITS(<tool_call>) (ap_direct_weak=0.189301)
- L21H1 -> LOGITS(<tool_call>) (ap_direct_weak=0.187903)

## 7. Conclusion (Localization only)
- The `<tool_call>` first-token decision is localized to a sparse set of late-layer heads and MLPs.
- Conditioned tracing through the key MLP identifies upstream heads with mediated causal contribution.
- Sufficiency and necessity both show measurable evidence, supporting a compact causal circuit explanation.

## 8. Figure Notes (Path / Metric / N / Color)
- `figs/ap_head_heatmap.png`: path `(L{layer}H{head} -> logits@first-token)`, metric `Impact = PD_unpatched - PD_patched`, `N=164`, colormap `RdBu_r` centered at 0.
- `figs/ap_mlp_heatmap.png`: path `(MLP{layer} -> logits@first-token)`, same metric, `N=164`, colormap `RdBu_r` centered at 0.
- `figs/ct_head_heatmap.png`: conditioned path score via `MLP27`, metric `Rescue = PD(C->corr + restore MLP27->clean) - PD(C->corr)`, `N=164`, colormap `RdBu_r` centered at 0.
- `figs/ct_mlp_heatmap.png`: conditioned MLP rescue vs `MLP27`, same metric family, `N=164`, colormap `RdBu_r` centered at 0.
- `figs/final_circuit.png`: final node-edge diagram with direction and signed/weighted edge styling based on AP/CT scores.

## 9. Update Note
- For stronger statistical evidence and paper-style figure organization, see `reports/circuit_localization_report_v2.md`.
- For multi-key-MLP panel tracing and AP-CT agreement stats, see `reports/circuit_localization_report_v3_note.md` and `reports/mlps_ipp_ap_ct_correlation_stats.csv`.
