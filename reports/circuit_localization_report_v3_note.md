# MLP-IPP Panel Refinement Note (v3)

## 1. Key MLP Panels
- Multiple key downstream MLPs were traced with conditioned path patching to test whether upstream head structure is stable or MLP-specific.
- Figure: `figs/mlps_ipp_panels.png`

## 2. AP-CT Agreement (with Significance, n=448 heads)
- MLP11: r=0.0883, p=6.18e-02, 95%CI=[-0.0044, 0.1795]
- MLP22: r=0.6784, p=1.04e-61, 95%CI=[0.6251, 0.7255]
- MLP26: r=0.1337, p=4.58e-03, 95%CI=[0.0416, 0.2236]
- MLP27: r=0.9499, p=2.15e-227, 95%CI=[0.9400, 0.9582]

## 3. Top Upstream Heads per Key MLP
- MLP11 top8: L10H5(0.0434), L10H2(0.0388), L9H5(0.0379), L11H6(0.0340), L11H5(0.0174), L2H15(0.0121), L7H9(0.0109), L11H10(0.0108)
- MLP22 top8: L20H5(0.0272), L21H12(0.0269), L21H1(0.0257), L17H8(0.0222), L17H2(0.0190), L16H13(0.0156), L15H4(0.0119), L16H8(0.0113)
- MLP26 top8: L24H6(0.0402), L22H7(0.0304), L20H14(0.0234), L24H14(0.0208), L26H15(0.0207), L26H10(0.0184), L17H1(0.0163), L17H12(0.0149)
- MLP27 top8: L24H6(0.1140), L21H1(0.0996), L17H8(0.0912), L21H12(0.0841), L20H5(0.0654), L16H13(0.0486), L23H6(0.0416), L17H2(0.0411)

## 4. Overlap Matrix
- Top-head set overlap (Jaccard, top-16 each panel):

|       |     MLP11 |     MLP22 |    MLP26 |    MLP27 |
|:------|----------:|----------:|---------:|---------:|
| MLP11 | 1         | 0.0322581 | 0        | 0        |
| MLP22 | 0.0322581 | 1         | 0.230769 | 0.333333 |
| MLP26 | 0         | 0.230769  | 1        | 0.333333 |
| MLP27 | 0         | 0.333333  | 0.333333 | 1        |
