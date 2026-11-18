# v5 results: capacity-matched BML and BML + controller

## Parameter counts

| Dataset | AdaptiveFusion | BML-small (width) | BML-full |
|---|---|---|---|
| MFeat | 91,114 | 90,353 (47) | 14,300,482 |
| Caltech101-20 | 390,836 | 388,320 (69) | 17,500,030 |
| Scene15 | 26,319 | 26,334 (39) | 6,941,744 |
| BDGP | 182,405 | 181,833 (69) | 8,608,018 |
| LandUse21 | 26,613 | 27,054 (39) | 6,944,066 |
| Reuters1200 | 983,894 | 989,555 (84) | 21,600,675 |

## Comparisons over the 7 scarcity conditions x 6 datasets

| Comparison | Protocol | Head | Mean diff (pp) | Wins | Losses | Holm-sig. wins | Holm-sig. losses |
|---|---|---|---|---|---|---|---|
| BML + controller (trained) vs BML | A | svm | +1.19 | 42 | 0 | 10 | 0 |
| BML + controller (trained) vs BML | A | own | +0.07 | 24 | 18 | 1 | 0 |
| BML + controller (trained) vs BML | B | svm | +0.11 | 21 | 21 | 1 | 0 |
| BML + controller (test time only) vs BML | A | svm | +0.76 | 39 | 2 | 16 | 0 |
| BML + controller (test time only) vs BML | A | own | +0.06 | 28 | 11 | 2 | 0 |
| BML + controller (test time only) vs BML | B | svm | +0.04 | 24 | 17 | 2 | 0 |
| BML router vs ignoring it (beta=0) | A | svm | -0.44 | 10 | 32 | 0 | 7 |
| BML router vs ignoring it (beta=0) | A | own | -0.04 | 18 | 23 | 1 | 4 |
| BML router vs ignoring it (beta=0) | B | svm | +0.00 | 21 | 20 | 0 | 2 |
| capacity-matched BML vs AdaptiveFusion | A | svm | -5.78 | 11 | 31 | 0 | 23 |
| capacity-matched BML vs AdaptiveFusion | B | svm | -3.15 | 16 | 26 | 3 | 18 |
| capacity-matched BML + controller vs capacity-matched BML | A | svm | +0.48 | 33 | 9 | 1 | 0 |
| capacity-matched BML + controller vs capacity-matched BML | A | own | +0.01 | 22 | 20 | 0 | 0 |
| capacity-matched BML + controller vs capacity-matched BML | B | svm | +0.05 | 26 | 16 | 0 | 0 |
| capacity-matched BML + controller vs AdaptiveFusion | A | svm | -5.30 | 10 | 32 | 1 | 23 |
| capacity-matched BML + controller vs AdaptiveFusion | B | svm | -3.10 | 16 | 26 | 1 | 16 |
| full BML vs capacity-matched BML | A | svm | +10.55 | 34 | 8 | 26 | 2 |
| full BML vs capacity-matched BML | A | own | +7.54 | 36 | 6 | 27 | 0 |
| full BML vs capacity-matched BML | B | svm | +7.90 | 36 | 6 | 27 | 0 |
