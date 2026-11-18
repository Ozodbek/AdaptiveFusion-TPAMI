# AdaptiveFusion results -- protocol A

Seeds per dataset: BDGP=10, Caltech101-20=10, LandUse21=10, MFeat=10, Reuters1200=10, Scene15=10

## Overview: AdaptiveFusion vs each method over all conditions

| Competitor | AF wins | AF losses | Holm-significant AF wins | Holm-significant AF losses |
|---|---|---|---|---|
| Concat-SVM | 42 | 0 | 42 | 0 |
| Const-beta | 18 | 23 | 0 | 0 |
| DCCA-style | 7 | 35 | 0 | 7 |
| DIMvLN-unified | 29 | 13 | 22 | 11 |
| DIMvLN-unified [own head] | 29 | 13 | 20 | 9 |
| MAXVAR-GCCA | 42 | 0 | 35 | 0 |
| SP-RR | 26 | 16 | 7 | 0 |
| Uniform | 29 | 13 | 0 | 0 |

(scarcity conditions: Missing30/50/70 and the four joint conditions)

## AdaptiveFusion vs SP-RR (scarcity conditions)

| Dataset | Condition | Diff (pp) | t p | Holm p | dz |
|---|---|---|---|---|---|
| BDGP | Missing30 | +0.48 | 0.0023 | 0.0818 | 1.33 |
| BDGP | Missing50 | +0.25 | 0.0852 | 1.0000 | 0.61 |
| BDGP | Missing50+Conflict50 | +0.12 | 0.5018 | 1.0000 | 0.22 |
| BDGP | Missing50+Noise50 | +0.31 | 0.0338 | 0.9124 | 0.79 |
| BDGP | Missing70 | +0.15 | 0.0932 | 1.0000 | 0.59 |
| BDGP | Missing70+Conflict50 | +0.09 | 0.2259 | 1.0000 | 0.41 |
| BDGP | Missing70+Noise50 | +0.31 | 0.0218 | 0.6320 | 0.88 |
| Caltech101-20 | Missing30 | +6.52 | 0.0000 | 0.0005 | 2.72 |
| Caltech101-20 | Missing50 | +7.12 | 0.0000 | 0.0000 | 4.59 |
| Caltech101-20 | Missing50+Conflict50 | +7.53 | 0.0000 | 0.0000 | 3.74 |
| Caltech101-20 | Missing50+Noise50 | +6.11 | 0.0000 | 0.0000 | 4.65 |
| Caltech101-20 | Missing70 | +4.70 | 0.0002 | 0.0067 | 1.92 |
| Caltech101-20 | Missing70+Conflict50 | +4.70 | 0.0004 | 0.0138 | 1.74 |
| Caltech101-20 | Missing70+Noise50 | +4.09 | 0.0000 | 0.0000 | 4.00 |
| LandUse21 | Missing30 | -0.40 | 0.0406 | 1.0000 | -0.76 |
| LandUse21 | Missing50 | -0.29 | 0.1559 | 1.0000 | -0.49 |
| LandUse21 | Missing50+Conflict50 | -0.30 | 0.0550 | 1.0000 | -0.70 |
| LandUse21 | Missing50+Noise50 | -0.21 | 0.5001 | 1.0000 | -0.22 |
| LandUse21 | Missing70 | -0.05 | 0.6376 | 1.0000 | -0.15 |
| LandUse21 | Missing70+Conflict50 | -0.08 | 0.5505 | 1.0000 | -0.20 |
| LandUse21 | Missing70+Noise50 | +0.08 | 0.5437 | 1.0000 | 0.20 |
| MFeat | Missing30 | -0.15 | 0.0543 | 1.0000 | -0.70 |
| MFeat | Missing50 | -0.05 | 0.5599 | 1.0000 | -0.19 |
| MFeat | Missing50+Conflict50 | +0.28 | 0.1553 | 1.0000 | 0.49 |
| MFeat | Missing50+Noise50 | -0.67 | 0.0103 | 0.3199 | -1.02 |
| MFeat | Missing70 | -0.03 | 0.6193 | 1.0000 | -0.16 |
| MFeat | Missing70+Conflict50 | +0.17 | 0.4090 | 1.0000 | 0.27 |
| MFeat | Missing70+Noise50 | -0.47 | 0.0137 | 0.4104 | -0.97 |
| Reuters1200 | Missing30 | +0.03 | 0.8227 | 1.0000 | 0.07 |
| Reuters1200 | Missing50 | -0.25 | 0.3168 | 1.0000 | -0.34 |
| Reuters1200 | Missing50+Conflict50 | +0.75 | 0.0736 | 1.0000 | 0.64 |
| Reuters1200 | Missing50+Noise50 | -0.94 | 0.0089 | 0.2862 | -1.05 |
| Reuters1200 | Missing70 | +0.08 | 0.5203 | 1.0000 | 0.21 |
| Reuters1200 | Missing70+Conflict50 | +0.36 | 0.1461 | 1.0000 | 0.50 |
| Reuters1200 | Missing70+Noise50 | -0.78 | 0.0306 | 0.8581 | -0.81 |
| Scene15 | Missing30 | +0.19 | 0.1059 | 1.0000 | 0.57 |
| Scene15 | Missing50 | +0.03 | 0.7454 | 1.0000 | 0.11 |
| Scene15 | Missing50+Conflict50 | +0.20 | 0.0796 | 1.0000 | 0.62 |
| Scene15 | Missing50+Noise50 | -0.48 | 0.0025 | 0.0849 | -1.31 |
| Scene15 | Missing70 | +0.14 | 0.0070 | 0.2307 | 1.10 |
| Scene15 | Missing70+Conflict50 | +0.01 | 0.8853 | 1.0000 | 0.05 |
| Scene15 | Missing70+Noise50 | -0.22 | 0.2037 | 1.0000 | -0.43 |
