# MLP-IPP Panel Refinement Note (v3)

## 1. Key MLP Panels
- Multiple key downstream MLPs were traced with conditioned path patching to test whether upstream head structure is stable or MLP-specific.
- Figure: `figs/mlps_ipp_panels.png`

## 2. AP-CT Agreement
- MLP11: Pearson r(AP head impact, CT rescue) = 0.0883
- MLP22: Pearson r(AP head impact, CT rescue) = 0.6784
- MLP26: Pearson r(AP head impact, CT rescue) = 0.1337
- MLP27: Pearson r(AP head impact, CT rescue) = 0.9499

## 3. Top Upstream Heads per Key MLP
- MLP11:
  - L10H5: rescue=0.043394
  - L10H2: rescue=0.038835
  - L9H5: rescue=0.037942
  - L11H6: rescue=0.033954
  - L11H5: rescue=0.017445
  - L2H15: rescue=0.012133
  - L7H9: rescue=0.010926
  - L11H10: rescue=0.010833
- MLP22:
  - L20H5: rescue=0.027232
  - L21H12: rescue=0.026900
  - L21H1: rescue=0.025657
  - L17H8: rescue=0.022248
  - L17H2: rescue=0.019010
  - L16H13: rescue=0.015551
  - L15H4: rescue=0.011934
  - L16H8: rescue=0.011262
- MLP26:
  - L24H6: rescue=0.040230
  - L22H7: rescue=0.030432
  - L20H14: rescue=0.023409
  - L24H14: rescue=0.020793
  - L26H15: rescue=0.020673
  - L26H10: rescue=0.018368
  - L17H1: rescue=0.016340
  - L17H12: rescue=0.014892
- MLP27:
  - L24H6: rescue=0.113986
  - L21H1: rescue=0.099647
  - L17H8: rescue=0.091230
  - L21H12: rescue=0.084102
  - L20H5: rescue=0.065410
  - L16H13: rescue=0.048572
  - L23H6: rescue=0.041604
  - L17H2: rescue=0.041062

## 4. Overlap Matrix
- Top-head set overlap (Jaccard, top-16 each panel):

|       |     MLP11 |     MLP22 |    MLP26 |    MLP27 |
|:------|----------:|----------:|---------:|---------:|
| MLP11 | 1         | 0.0322581 | 0        | 0        |
| MLP22 | 0.0322581 | 1         | 0.230769 | 0.333333 |
| MLP26 | 0         | 0.230769  | 1        | 0.333333 |
| MLP27 | 0         | 0.333333  | 0.333333 | 1        |