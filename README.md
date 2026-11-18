# Decoupling View Availability and Reliability for Robust Multi-View Classification

Code, data, and complete results for the paper

> O. Yusupov, Y. Yuldoshev, Y. Khandamov, and E. Eshonqulov, "Decoupling View Availability and Reliability for Robust
> Multi-View Classification," submitted to *IEEE Transactions on Pattern Analysis and
> Machine Intelligence*, 2026.

The paper proposes an **availability controller** for reliability-aware multi-view fusion:
the influence of an estimated view reliability grows with the number of observed views,
$\beta_i = (K_i-1)/(V-1)$, $w_i^{(v)} = m_i^{(v)}\,[(1-\beta_i) + \beta_i r_i^{(v)}]$.
It is evaluated in a feature-level model (AdaptiveFusion) and as a plug-in for a
decision-level method (BML).

## Repository layout

| Path | Content |
|---|---|
| `benchmark/` | All methods and the common evaluation protocol (Python, PyTorch) |
| `data/` | The six benchmark datasets used in the paper (see `data/README.md`) |
| `run1.py`, `run1.bat`, `run1_quick.bat` | One-command reproduction of every experiment |
| `results/v6/` | The complete results reported in the paper: per-seed CSVs, epoch selection, trained AdaptiveFusion checkpoints, AUROC/localization, reports, logs, and the software environment |
| `paper/make_tables.py` | Builds every table and every number quoted in the paper from `results/v6` |
| `paper/make_figure.py` | Draws Fig. 1 (`paper/perturbation.pdf`) from `results/v6` |
| `paper/tables/`, `paper/numbers.json` | Output of `make_tables.py` |

## Methods (`benchmark/`)

| File | Content |
|---|---|
| `adaptivefusion.py` | AdaptiveFusion: shared/private encoders, residual reliability, availability controller |
| `unified.py` | Common protocol: splits, 12 test conditions, protocols A and B, inner-CV epoch selection; Concat-SVM, MAXVAR-GCCA, DCCA-style, DIMvLN, and the fusion-rule controls (Uniform, Const-β, SP-RR) |
| `v5_bml.py` | BML (full width and capacity-matched), each with its own fusion rule and with the controller |
| `gcca.py`, `dimvln_matched.py`, `bml_matched.py` | Baseline components |
| `auroc_from_models.py` | Corruption detection AUROC and localization accuracy from saved checkpoints |
| `report.py`, `v5_report.py`, `summarize.py` | Summary tables and statistics |

## Reproducing the experiments

Requirements: Python 3.11–3.13; packages in `requirements.txt` (PyTorch ≥ 2.1, CUDA optional).

**Windows:** run `run1_quick.bat` first (installs the packages into `.venv`, then a short
check with 2 seeds), then `run1.bat` (full run: 6 datasets × 10 seeds; about 1–2 days on
one RTX 4090). An interrupted run resumes where it stopped.

**Linux/macOS:**
```bash
pip install -r requirements.txt
python run1.py --quick     # 2 seeds, tiny epoch grid: checks the pipeline
python run1.py             # full run, writes results/v6
```

All learned methods select their training budget from the same grid
{25, 50, 100, 200, 400} epochs by 3-fold inner cross-validation. Seeds:
3, 7, 11, 17, 23, 29, 31, 37, 41, 43.

## Reproducing the tables

```bash
python paper/make_tables.py
```
reads `results/v6` and writes `paper/tables/*.tex` and `paper/numbers.json`; these are the
tables and numbers of the paper. `python paper/make_figure.py` draws Fig. 1.

## Determinism

Training is deterministic for a given seed and device. Methods without training epochs
(Concat-SVM, MAXVAR-GCCA) and BML (whose selected budgets never fell on the new grid point)
reproduce earlier runs bit for bit; see `results/v6/environment.txt` for the software
versions used.

## Citation

```bibtex
@article{yusupov2026decoupling,
  author  = {Yusupov, Ozod and Yuldoshev, Yusuf and Khandamov, Yigitali and Eshonqulov, Erali},
  title   = {Decoupling View Availability and Reliability for Robust Multi-View Classification},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  note    = {Submitted},
  year    = {2026}
}
```

## Funding

This research was supported by the Agency for Innovative Development under the Ministry of
Education and Science of the Republic of Uzbekistan, project No. AL-9224104661.

## License

Code: MIT License (see `LICENSE`). The datasets in `data/` remain under the terms of their
original sources (see `data/README.md`).
