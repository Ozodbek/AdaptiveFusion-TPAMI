"""Report for the v5 experiments (capacity-matched BML, BML + controller).

Reads  <results>/v5/*.csv  and, if present, AdaptiveFusion rows from the v4 run
<results>/unified/*.csv (same seeds, splits and test sets, so paired tests are valid).

Writes <results>/report_v5/:
  accuracy_<protocol>_<head>.csv   mean accuracy of every variant
  comparisons.csv                  paired tests for every comparison (t, Wilcoxon, dz, Holm)
  comparison_summary.csv           wins / losses / Holm-significant per comparison
  latex_tables.tex                 ready-to-paste LaTeX tables
  REPORT.md                        short overview
"""
import argparse, glob, json, os
import numpy as np, pandas as pd
from scipy.stats import ttest_rel, wilcoxon

DATASETS = ["MFeat", "Caltech101-20", "Scene15", "BDGP", "LandUse21", "Reuters1200"]
PRETTY = {"MFeat": "MFeat", "Caltech101-20": "Caltech101-20", "Scene15": "Scene15",
          "BDGP": "BDGP", "LandUse21": "LandUse-21", "Reuters1200": "Reuters-1200"}
JOINT = ["Missing50+Noise50", "Missing70+Noise50", "Missing50+Conflict50", "Missing70+Conflict50"]
SCARCE = ["Missing30", "Missing50", "Missing70"] + JOINT
ALL = ["Clean", "Missing30", "Missing50", "Missing70", "Conflict30", "Conflict50",
       "Noise30", "Noise50"] + JOINT
SHORT = lambda c: c.replace("Missing", "M").replace("Conflict", "C").replace("Noise", "N")

V = lambda size, tr, te: f"BML-{size}[train={tr},test={te}]"
AF = "AdaptiveFusion"
COMPARISONS = [  # (key, description, treatment, control, heads)
    ("plugin_trained", "BML + controller (trained) vs BML", V("full", "ctrl", "ctrl"), V("full", "fixed", "fixed"), ("svm", "own")),
    ("plugin_testtime", "BML + controller (test time only) vs BML", V("full", "fixed", "ctrl"), V("full", "fixed", "fixed"), ("svm", "own")),
    ("router_value", "BML router vs ignoring it (beta=0)", V("full", "fixed", "fixed"), V("full", "fixed", "uniform"), ("svm", "own")),
    ("capacity", "capacity-matched BML vs AdaptiveFusion", V("small", "fixed", "fixed"), AF, ("svm",)),
    ("plugin_small", "capacity-matched BML + controller vs capacity-matched BML", V("small", "ctrl", "ctrl"), V("small", "fixed", "fixed"), ("svm", "own")),
    ("small_ctrl_vs_af", "capacity-matched BML + controller vs AdaptiveFusion", V("small", "ctrl", "ctrl"), AF, ("svm",)),
    ("capacity_cost", "full BML vs capacity-matched BML", V("full", "fixed", "fixed"), V("small", "fixed", "fixed"), ("svm", "own")),
]
MAIN = [V("full", "fixed", "fixed"), V("full", "fixed", "ctrl"), V("full", "ctrl", "ctrl"),
        V("full", "fixed", "uniform"), V("small", "fixed", "fixed"), V("small", "ctrl", "ctrl"), AF]


def holm(p):
    p = np.asarray(p, float); out = np.full_like(p, np.nan); ok = ~np.isnan(p)
    if not ok.any(): return out
    idx = np.argsort(p[ok]); m = ok.sum()
    adj = np.minimum(1, np.maximum.accumulate((m - np.arange(m)) * p[ok][idx]))
    tmp = np.empty(m); tmp[idx] = adj; out[ok] = tmp
    return out


def paired(a, b):
    q = a.merge(b, on="Seed", suffixes=("_t", "_c"))
    if len(q) < 2: return None
    diff = q.Accuracy_t - q.Accuracy_c
    if np.allclose(diff, 0): return len(q), 0.0, 1.0, 1.0, 0.0
    tp = ttest_rel(q.Accuracy_t, q.Accuracy_c).pvalue
    try: wp = wilcoxon(diff).pvalue
    except ValueError: wp = np.nan
    sd = diff.std(ddof=1)
    return len(q), diff.mean(), tp, wp, (diff.mean() / sd if sd > 0 else np.nan)


def latex(caption, label, header, rows, colspec):
    s = ["\\begin{table*}[!t]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
         "\\setlength{\\tabcolsep}{3pt}", f"\\begin{{tabular}}{{{colspec}}}", "\\toprule",
         " & ".join(header) + " \\\\", "\\midrule"] + [" & ".join(r) + " \\\\" for r in rows]
    return "\n".join(s + ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--results", default="./results")
    a = ap.parse_args()
    files = [f for f in glob.glob(os.path.join(a.results, "v5", "*.csv")) if not f.endswith("_epoch_selection.csv")]
    if not files:
        print("No v5 results found."); return
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    uf = [f for f in glob.glob(os.path.join(a.results, "unified", "*.csv"))
          if not f.endswith(("_K.csv", "_epoch_selection.csv"))]
    has_af = False
    if uf:
        u = pd.concat([pd.read_csv(f) for f in uf], ignore_index=True)
        u = u[(u.Method == AF) & u.Condition.isin(ALL)].copy()
        if len(u):
            u["Head"] = "svm"; has_af = True
            d = pd.concat([d, u[["Dataset", "Method", "Seed", "Condition", "Protocol", "Head",
                                 "Accuracy", "MacroF1", "BalancedAccuracy"]]], ignore_index=True)
    odir = os.path.join(a.results, "report_v5"); os.makedirs(odir, exist_ok=True)
    dsets = [x for x in DATASETS if x in set(d.Dataset)]

    # ---------- accuracy tables
    for (p, h), g in d.groupby(["Protocol", "Head"]):
        t = g.pivot_table(index=["Dataset", "Method"], columns="Condition", values="Accuracy", aggfunc="mean")
        t = t[[c for c in ALL if c in t.columns]].round(2)
        t.to_csv(os.path.join(odir, f"accuracy_{p}_{h}.csv"))

    # ---------- paired comparisons
    out = []
    for key, desc, trt, ctl, heads in COMPARISONS:
        for p in sorted(d.Protocol.unique()):
            for h in heads:
                for ds in dsets:
                    for c in ALL:
                        sel = (d.Dataset == ds) & (d.Condition == c) & (d.Protocol == p)
                        A = d[sel & (d.Method == trt) & (d.Head == h)][["Seed", "Accuracy"]]
                        # AdaptiveFusion exists only with the SVM head
                        hc = "svm" if ctl == AF else h
                        B = d[sel & (d.Method == ctl) & (d.Head == hc)][["Seed", "Accuracy"]]
                        r = paired(A, B)
                        if r is None: continue
                        n, md, tp, wp, dz = r
                        out.append(dict(Comparison=key, Description=desc, Protocol=p, Head=h, Dataset=ds,
                                        Condition=c, n=n, MeanDiff_pp=md, t_p=tp, Wilcoxon_p=wp, Cohen_dz=dz))
    st = pd.DataFrame(out)
    if len(st):
        st["Holm_t_p"] = np.nan
        for _, g in st[st.Condition.isin(SCARCE)].groupby(["Comparison", "Protocol", "Head"]):
            st.loc[g.index, "Holm_t_p"] = holm(g.t_p.values)
        st.to_csv(os.path.join(odir, "comparisons.csv"), index=False)
        summ = []
        for (k, p, h), g in st[st.Condition.isin(SCARCE)].groupby(["Comparison", "Protocol", "Head"], sort=False):
            sig = g.Holm_t_p < .05
            summ.append(dict(Comparison=k, Description=g.Description.iloc[0], Protocol=p, Head=h, Cells=len(g),
                             MeanDiff_pp=round(g.MeanDiff_pp.mean(), 2), Wins=int((g.MeanDiff_pp > 0).sum()),
                             Losses=int((g.MeanDiff_pp < 0).sum()), SigWins=int((sig & (g.MeanDiff_pp > 0)).sum()),
                             SigLosses=int((sig & (g.MeanDiff_pp < 0)).sum())))
        summ = pd.DataFrame(summ)
        summ.to_csv(os.path.join(odir, "comparison_summary.csv"), index=False)
    else:
        summ = pd.DataFrame()

    # ---------- LaTeX
    T = []
    for p in sorted(d.Protocol.unique()):
        for h in ["svm", "own"]:
            g = d[(d.Protocol == p) & (d.Head == h)]
            meths = [m for m in MAIN if m in set(g.Method)]
            if not meths: continue
            rows = []
            for ds in dsets:
                for i, m in enumerate(meths):
                    r = [PRETTY[ds] if i == 0 else "", m.replace("BML-", "")]
                    for c in ["Clean", "Missing50", "Missing70"] + JOINT:
                        q = g[(g.Dataset == ds) & (g.Method == m) & (g.Condition == c)].Accuracy
                        r.append(f"{q.mean():.2f}\\,$\\pm$\\,{q.std(ddof=1):.2f}" if len(q) else "--")
                    rows.append(r)
            T.append(latex(f"v5 accuracy (\\%), protocol {p}, {'linear SVM' if h == 'svm' else 'own output layer'}.",
                           f"tab:v5_{p}_{h}", ["Dataset", "Variant", "Clean", "M50", "M70"] + [SHORT(c) for c in JOINT],
                           rows, "ll" + "c" * 7))
    if len(st):
        for key in ["plugin_trained", "capacity"]:
            g = st[(st.Comparison == key)]
            if not len(g): continue
            rows = []
            for (p, h), q in g.groupby(["Protocol", "Head"]):
                for ds in dsets:
                    r = [f"{p}/{h}" if ds == dsets[0] else "", PRETTY[ds]]
                    for c in SCARCE:
                        x = q[(q.Dataset == ds) & (q.Condition == c)]
                        if not len(x): r.append("--"); continue
                        x = x.iloc[0]; star = "$^{*}$" if x.Holm_t_p < .05 else ""
                        r.append(f"${x.MeanDiff_pp:+.2f}${star}")
                    rows.append(r)
            T.append(latex(f"{g.Description.iloc[0]}: accuracy difference (pp). $^{{*}}$ Holm-adjusted $p<0.05$.",
                           f"tab:v5_{key}", ["Prot./head", "Dataset"] + [SHORT(c) for c in SCARCE], rows,
                           "ll" + "c" * len(SCARCE)))
    with open(os.path.join(odir, "latex_tables.tex"), "w", encoding="utf-8") as f:
        f.write("% Auto-generated by benchmark/v5_report.py -- requires \\usepackage{booktabs}\n\n" + "\n".join(T))

    # ---------- markdown
    cfg = {}
    for f in glob.glob(os.path.join(a.results, "v5", "*_config.json")):
        c = json.load(open(f)); cfg[c["dataset"]] = c
    L = ["# v5 results: capacity-matched BML and BML + controller", ""]
    if not has_af:
        L += ["**Note:** v4 results (results/unified) not found, so comparisons with AdaptiveFusion are missing.", ""]
    L += ["## Parameter counts", "", "| Dataset | AdaptiveFusion | BML-small (width) | BML-full |", "|---|---|---|---|"]
    for ds in dsets:
        if ds in cfg:
            c = cfg[ds]
            L.append(f"| {ds} | {c['AdaptiveFusion_params']:,} | {c['BML_small_params']:,} ({c['BML_small_config']['width']}) | {c['BML_full_params']:,} |")
    L += ["", "## Comparisons over the 7 scarcity conditions x 6 datasets", "",
          "| Comparison | Protocol | Head | Mean diff (pp) | Wins | Losses | Holm-sig. wins | Holm-sig. losses |",
          "|---|---|---|---|---|---|---|---|"]
    for _, r in summ.iterrows():
        L.append(f"| {r.Description} | {r.Protocol} | {r.Head} | {r.MeanDiff_pp:+.2f} | {r.Wins} | {r.Losses} | {r.SigWins} | {r.SigLosses} |")
    with open(os.path.join(odir, "REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("v5 report written to", odir)


if __name__ == "__main__":
    main()
