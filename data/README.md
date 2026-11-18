# Datasets

| File | Dataset | Samples | Classes | Views | Source |
|---|---|---|---|---|---|
| `multiple+features.zip` | MFeat (UCI Multiple Features) | 2000 | 10 | 6 | R. P. W. Duin, UCI Machine Learning Repository, doi:10.24432/C5HC70 |
| `Caltech101-20_balanced_33perclass.mat` | Caltech101-20, class-balanced subset | 660 | 20 | 6 | L. Fei-Fei, R. Fergus, P. Perona, CVPR Workshops 2004 |
| `Scene15.mat` | Scene15 | 4485 | 15 | 3 | S. Lazebnik, C. Schmid, J. Ponce, CVPR 2006 |
| `BDGP_fea.mat` | BDGP | 2500 | 5 | 3 | X. Cai, H. Wang, H. Huang, C. Ding, Bioinformatics 2012 |
| `LandUse-21.mat` | LandUse-21 | 2100 | 21 | 3 | Y. Yang, S. Newsam, ACM SIGSPATIAL 2010 |
| `Reuters-1200.mat` | Reuters-1200 | 1200 | 6 | 5 | M.-R. Amini, N. Usunier, C. Goutte, NeurIPS 2009 |

These are the multi-view feature versions commonly used in the multi-view learning
literature; they are redistributed here only to make the experiments reproducible.
Please cite the original sources.

## Caltech101-20 subset

From the 2,386 instances of Caltech101-20, a class-balanced subset was drawn once, before
any train/test split: 33 instances per class, sampled uniformly without replacement with
NumPy's `default_rng(42)` (660 instances; all six views and the labels are kept).
The original sample indices are listed in `Caltech101-20_subset_indices.csv`.
