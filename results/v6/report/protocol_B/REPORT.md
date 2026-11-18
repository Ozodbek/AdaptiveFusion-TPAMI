# AdaptiveFusion results -- protocol B

Seeds per dataset: BDGP=10, Caltech101-20=10, LandUse21=10, MFeat=10, Reuters1200=10, Scene15=10

## Overview: AdaptiveFusion vs each method over all conditions

| Competitor | AF wins | AF losses | Holm-significant AF wins | Holm-significant AF losses |
|---|---|---|---|---|
| Concat-SVM | 42 | 0 | 42 | 0 |
| Const-beta | 14 | 26 | 0 | 1 |
| DCCA-style | 10 | 32 | 2 | 0 |
| DIMvLN-unified | 42 | 0 | 42 | 0 |
| MAXVAR-GCCA | 41 | 1 | 40 | 0 |
| SP-RR | 27 | 14 | 6 | 3 |
| Uniform | 23 | 19 | 3 | 0 |

(scarcity conditions: Missing30/50/70 and the four joint conditions)

## AdaptiveFusion vs SP-RR (scarcity conditions)

| Dataset | Condition | Diff (pp) | t p | Holm p | dz |
|---|---|---|---|---|---|
| BDGP | Missing30 | -0.03 | 0.8144 | 1.0000 | -0.08 |
| BDGP | Missing50 | -0.08 | 0.7198 | 1.0000 | -0.12 |
| BDGP | Missing50+Conflict50 | -0.36 | 0.0932 | 1.0000 | -0.59 |
| BDGP | Missing50+Noise50 | +0.07 | 0.7184 | 1.0000 | 0.12 |
| BDGP | Missing70 | +0.36 | 0.0442 | 1.0000 | 0.74 |
| BDGP | Missing70+Conflict50 | +0.08 | 0.5462 | 1.0000 | 0.20 |
| BDGP | Missing70+Noise50 | +0.07 | 0.7377 | 1.0000 | 0.11 |
| Caltech101-20 | Missing30 | +5.25 | 0.0000 | 0.0001 | 3.18 |
| Caltech101-20 | Missing50 | +6.31 | 0.0000 | 0.0003 | 2.84 |
| Caltech101-20 | Missing50+Conflict50 | +8.38 | 0.0000 | 0.0013 | 2.39 |
| Caltech101-20 | Missing50+Noise50 | +2.98 | 0.0170 | 0.5262 | 0.92 |
| Caltech101-20 | Missing70 | +5.40 | 0.0002 | 0.0054 | 1.97 |
| Caltech101-20 | Missing70+Conflict50 | +6.36 | 0.0000 | 0.0015 | 2.35 |
| Caltech101-20 | Missing70+Noise50 | +3.28 | 0.0011 | 0.0369 | 1.49 |
| LandUse21 | Missing30 | +0.30 | 0.2899 | 1.0000 | 0.36 |
| LandUse21 | Missing50 | +0.56 | 0.0734 | 1.0000 | 0.64 |
| LandUse21 | Missing50+Conflict50 | +0.22 | 0.4861 | 1.0000 | 0.23 |
| LandUse21 | Missing50+Noise50 | -2.33 | 0.0001 | 0.0030 | -2.13 |
| LandUse21 | Missing70 | +0.33 | 0.0269 | 0.7791 | 0.84 |
| LandUse21 | Missing70+Conflict50 | +0.22 | 0.2380 | 1.0000 | 0.40 |
| LandUse21 | Missing70+Noise50 | -0.54 | 0.1879 | 1.0000 | -0.45 |
| MFeat | Missing30 | +0.00 | 1.0000 | 1.0000 | 0.00 |
| MFeat | Missing50 | +0.03 | 0.7895 | 1.0000 | 0.09 |
| MFeat | Missing50+Conflict50 | +0.30 | 0.1180 | 1.0000 | 0.55 |
| MFeat | Missing50+Noise50 | -0.52 | 0.0282 | 0.7907 | -0.83 |
| MFeat | Missing70 | +0.03 | 0.6926 | 1.0000 | 0.13 |
| MFeat | Missing70+Conflict50 | +0.73 | 0.0174 | 0.5262 | 0.92 |
| MFeat | Missing70+Noise50 | -0.83 | 0.0054 | 0.1788 | -1.15 |
| Reuters1200 | Missing30 | -0.06 | 0.8173 | 1.0000 | -0.08 |
| Reuters1200 | Missing50 | -0.17 | 0.1114 | 1.0000 | -0.56 |
| Reuters1200 | Missing50+Conflict50 | -0.19 | 0.3726 | 1.0000 | -0.30 |
| Reuters1200 | Missing50+Noise50 | -0.92 | 0.0640 | 1.0000 | -0.67 |
| Reuters1200 | Missing70 | +0.00 | 1.0000 | 1.0000 | 0.00 |
| Reuters1200 | Missing70+Conflict50 | +0.08 | 0.8350 | 1.0000 | 0.07 |
| Reuters1200 | Missing70+Noise50 | -1.33 | 0.0124 | 0.3969 | -0.99 |
| Scene15 | Missing30 | +0.23 | 0.0543 | 1.0000 | 0.70 |
| Scene15 | Missing50 | +0.10 | 0.4923 | 1.0000 | 0.23 |
| Scene15 | Missing50+Conflict50 | +0.33 | 0.1594 | 1.0000 | 0.49 |
| Scene15 | Missing50+Noise50 | -2.48 | 0.0000 | 0.0000 | -3.63 |
| Scene15 | Missing70 | +0.20 | 0.0499 | 1.0000 | 0.72 |
| Scene15 | Missing70+Conflict50 | +0.22 | 0.2630 | 1.0000 | 0.38 |
| Scene15 | Missing70+Noise50 | -1.07 | 0.0000 | 0.0013 | -2.40 |
