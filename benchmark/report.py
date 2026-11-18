"""Builds every table, statistic and figure for the paper from the CSV files
written by unified.py (all methods under one protocol).

Outputs (in <results>/report/):
  summary.csv               mean/std of Accuracy, MacroF1, BalancedAccuracy
  paired_stats.csv          AdaptiveFusion vs every other method, t-test, Wilcoxon, dz, Holm
  K_distribution.csv        realized observed-view counts
  perturbation.csv          accuracy vs reliability-perturbation magnitude
  perturbation.pdf / .png   figure for the paper
  latex_tables.tex          ready-to-paste LaTeX tables
  REPORT.md                 short human-readable overview
"""
import argparse, glob, os
import numpy as np, pandas as pd
from scipy.stats import ttest_rel, wilcoxon

DATASETS = ["MFeat", "Caltech101-20", "Scene15", "BDGP", "LandUse21", "Reuters1200"]
PRETTY = {"MFeat": "MFeat", "Caltech101-20": "Caltech101-20", "Scene15": "Scene15",
          "BDGP": "BDGP", "LandUse21": "LandUse-21", "Reuters1200": "Reuters-1200"}
JOINT = ["Missing50+Noise50", "Missing70+Noise50", "Missing50+Conflict50", "Missing70+Conflict50"]
SCARCE = ["Missing30", "Missing50", "Missing70"] + JOINT
ALL = ["Clean", "Missing30", "Missing50", "Missing70", "Conflict30", "Conflict50", "Noise30", "Noise50"] + JOINT


def holm(p):
    p = np.asarray(p, float); out = np.full_like(p, np.nan)
    ok = ~np.isnan(p); idx = np.argsort(p[ok]); m = ok.sum()
    adj = np.minimum(1, np.maximum.accumulate((m - np.arange(m)) * p[ok][idx]))
    tmp = np.empty(m); tmp[idx] = adj; out[ok] = tmp
    return out


def ms(g, col="Accuracy"):
    return f"{g[col].mean():.2f}\\,$\\pm$\\,{g[col].std(ddof=1):.2f}"


def cell(d, ds, meth, cond):
    g = d[(d.Dataset == ds) & (d.Method == meth) & (d.Condition == cond)]
    return ms(g) if len(g) else "--"


def mean(d, ds, meth, cond):
    g = d[(d.Dataset == ds) & (d.Method == meth) & (d.Condition == cond)]
    return g.Accuracy.mean() if len(g) else np.nan


def latex_table(caption, label, header, rows, star=False, colspec=None):
    env = "table*" if star else "table"
    colspec = colspec or ("l" + "c" * (len(header) - 1))
    s = [f"\\begin{{{env}}}[!t]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
         "\\setlength{\\tabcolsep}{3pt}", f"\\begin{{tabular}}{{{colspec}}}", "\\toprule",
         " & ".join(header) + " \\\\", "\\midrule"]
    s += [" & ".join(r) + " \\\\" for r in rows]
    s += ["\\bottomrule", "\\end{tabular}", f"\\end{{{env}}}", ""]
    return "\n".join(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="./results")
    a = ap.parse_args()
    rdir = os.path.join(a.results, "unified")
    files = [f for f in glob.glob(os.path.join(rdir, "*.csv"))
             if not f.endswith(("_K.csv", "_epoch_selection.csv"))]
    if not files:
        print("No results found in", rdir); return
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if "Protocol" not in d: d["Protocol"] = "A"
    root = os.path.join(a.results, "report"); os.makedirs(root, exist_ok=True)
    sf = glob.glob(os.path.join(rdir, "*_epoch_selection.csv"))
    if sf:
        es = pd.concat([pd.read_csv(f).assign(Dataset=os.path.basename(f).replace("_epoch_selection.csv", ""))
                        for f in sf], ignore_index=True)
        es.to_csv(os.path.join(root, "epoch_selection_all.csv"), index=False)
        es[es.Selected].groupby(["Dataset", "Group"]).Epochs.agg(
            lambda x: ", ".join(map(str, x))).reset_index().to_csv(
            os.path.join(root, "selected_epochs.csv"), index=False)
    for proto in sorted(d.Protocol.unique()):
        build(d[d.Protocol == proto].copy(), rdir, os.path.join(root, f"protocol_{proto}"), proto)
    print("Report written to", root)


def build(d, rdir, odir, proto):
    os.makedirs(odir, exist_ok=True)
    dsets = [x for x in DATASETS if x in set(d.Dataset)]

    # ---------------- summary ----------------
    summ = d.groupby(["Dataset", "Method", "Condition"]).agg(
        n=("Seed", "count"),
        Accuracy_mean=("Accuracy", "mean"), Accuracy_std=("Accuracy", "std"),
        MacroF1_mean=("MacroF1", "mean"), MacroF1_std=("MacroF1", "std"),
        BAcc_mean=("BalancedAccuracy", "mean"), BAcc_std=("BalancedAccuracy", "std")).reset_index()
    summ.to_csv(os.path.join(odir, "summary.csv"), index=False)

    # ---------------- paired statistics ----------------
    st = []
    for (ds, cond), g in d.groupby(["Dataset", "Condition"]):
        ref = g[g.Method == "AdaptiveFusion"][["Seed", "Accuracy"]]
        for meth in sorted(set(g.Method) - {"AdaptiveFusion"}):
            q = ref.merge(g[g.Method == meth][["Seed", "Accuracy"]], on="Seed", suffixes=("_af", "_m"))
            if len(q) < 2: continue
            diff = q.Accuracy_af - q.Accuracy_m
            if np.allclose(diff, 0):
                tp = wp = 1.0; dz = 0.0
            else:
                tp = ttest_rel(q.Accuracy_af, q.Accuracy_m).pvalue
                try: wp = wilcoxon(diff).pvalue
                except ValueError: wp = np.nan
                sd = diff.std(ddof=1); dz = diff.mean() / sd if sd > 0 else np.nan
            st.append(dict(Dataset=ds, Condition=cond, Competitor=meth, n=len(q),
                           MeanDiff_pp=diff.mean(), t_p=tp, Wilcoxon_p=wp, Cohen_dz=dz))
    st = pd.DataFrame(st, columns=["Dataset", "Condition", "Competitor", "n", "MeanDiff_pp",
                                   "t_p", "Wilcoxon_p", "Cohen_dz"])
    st["Holm_t_p"] = np.nan
    for comp in st.Competitor.unique():           # one family per competitor, scarcity conditions
        fam = (st.Competitor == comp) & st.Condition.isin(SCARCE)
        st.loc[fam, "Holm_t_p"] = holm(st.loc[fam, "t_p"].values)
    st.to_csv(os.path.join(odir, "paired_stats.csv"), index=False)

    # ---------------- K distribution ----------------
    kf = [f for f in glob.glob(os.path.join(rdir, "*_K.csv"))]
    kd = pd.concat([pd.read_csv(f) for f in kf], ignore_index=True) if kf else pd.DataFrame()
    if len(kd):
        kd = kd.groupby(["Dataset", "Condition", "K"]).Fraction.mean().reset_index()
        kd.to_csv(os.path.join(odir, "K_distribution.csv"), index=False)

    # ---------------- perturbation ----------------
    pert = []
    for ds in dsets:
        for meth in ["AdaptiveFusion", "SP-RR", "Const-beta", "Uniform"]:
            base = d[(d.Dataset == ds) & (d.Method == meth) & (d.Condition == "Missing70")]
            if len(base): pert.append(dict(Dataset=ds, Method=meth, Sigma=0.0,
                                           Acc=base.Accuracy.mean(), Std=base.Accuracy.std(ddof=1)))
            g = d[(d.Dataset == ds) & (d.Method == meth) & d.Condition.str.contains("RelPert", na=False)]
            for sig, h in g.groupby("Sigma"):
                pert.append(dict(Dataset=ds, Method=meth, Sigma=sig, Acc=h.Accuracy.mean(),
                                 Std=h.Accuracy.std(ddof=1)))
    pert = pd.DataFrame(pert) if proto == "A" else pd.DataFrame()
    pert.to_csv(os.path.join(odir, "perturbation.csv"), index=False)
    try:
        if not len(pert): raise RuntimeError("no perturbation results for this protocol")
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        n = len(dsets); cols = 3; rows_ = int(np.ceil(n / cols))
        fig, axs = plt.subplots(rows_, cols, figsize=(10, 3 * rows_), squeeze=False)
        style = {"AdaptiveFusion": ("#1f5fa8", "o", "-"), "SP-RR": ("#c0392b", "s", "--"),
                 "Const-beta": ("#7f8c8d", "^", ":"), "Uniform": ("#27ae60", "d", "-.")}
        for ax, ds in zip(axs.ravel(), dsets):
            for meth, (c, mk, ls) in style.items():
                h = pert[(pert.Dataset == ds) & (pert.Method == meth)].sort_values("Sigma")
                if len(h): ax.plot(h.Sigma, h.Acc, color=c, marker=mk, ls=ls, label=meth, ms=4)
            ax.set_title(PRETTY[ds], fontsize=10); ax.set_xlabel(r"perturbation $\sigma$")
            ax.set_ylabel("accuracy (%)"); ax.grid(alpha=.3)
        for ax in axs.ravel()[n:]: ax.axis("off")
        axs.ravel()[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(odir, "perturbation.pdf")); fig.savefig(os.path.join(odir, "perturbation.png"), dpi=200)
        plt.close(fig)
    except Exception as e:
        print("Figure skipped:", e)

    # ---------------- LaTeX tables ----------------
    T = []
    T.append(latex_table("AdaptiveFusion accuracy (\\%) under complete and missing views (ten seeds).",
                         "tab:missing", ["Dataset", "Clean", "Missing30", "Missing50", "Missing70"],
                         [[PRETTY[ds]] + [cell(d, ds, "AdaptiveFusion", c) for c in
                                          ["Clean", "Missing30", "Missing50", "Missing70"]] for ds in dsets]))
    BASE = ["Concat-SVM", "MAXVAR-GCCA", "DCCA-style", "BML-unified", "DIMvLN-unified"]
    MAIN = ["AdaptiveFusion", "SP-RR", "Uniform"] + BASE
    present = set(d.Method)
    MAIN = [m for m in MAIN if m in present]; BASE = [m for m in BASE if m in present]

    def best_cells(ds, cond, methods):
        vals = {m: mean(d, ds, m, cond) for m in methods}
        top = max(v for v in vals.values() if not np.isnan(v)) if vals else np.nan
        return [("\\textbf{" + cell(d, ds, m, cond) + "}") if vals[m] == top else cell(d, ds, m, cond)
                for m in methods]

    def method_table(conds, caption, label, header_conds):
        rows = []
        for ds in dsets:
            cols = {c: best_cells(ds, c, MAIN) for c in conds}
            for i, m in enumerate(MAIN):
                rows.append([PRETTY[ds] if i == 0 else "", m] + [cols[c][i] for c in conds])
        return latex_table(caption, label, ["Dataset", "Method"] + header_conds, rows,
                           star=True, colspec="ll" + "c" * len(conds))

    T.append(method_table(["Clean", "Missing30", "Missing50", "Missing70"],
             "Accuracy (\\%) of all methods under one protocol: complete and missing views (ten seeds). "
             "Best per dataset and condition in bold.", "tab:matched_missing",
             ["Clean", "Missing30", "Missing50", "Missing70"]))
    T.append(method_table(["Conflict30", "Conflict50", "Noise30", "Noise50"],
             "Accuracy (\\%) of all methods under one protocol: conflicting and noisy views, all views observed.",
             "tab:matched_corrupt", ["Conflict30", "Conflict50", "Noise30", "Noise50"]))
    T.append(method_table(JOINT,
             "Accuracy (\\%) under joint missingness and corruption (ten seeds). Half of the samples "
             "with at least two observed views have one observed view corrupted.", "tab:joint",
             ["M50+N50", "M70+N50", "M50+C50", "M70+C50"]))
    rows = []
    for ds in dsets:
        r = [PRETTY[ds]]
        for c in ["Missing50", "Missing70"]:
            for comp in ["SP-RR", "Uniform"]:
                r.append(f"${mean(d, ds, 'AdaptiveFusion', c) - mean(d, ds, comp, c):+.2f}$")
            bb = max(BASE, key=lambda m: -np.inf if np.isnan(mean(d, ds, m, c)) else mean(d, ds, m, c)) if BASE else None
            r.append(f"${mean(d, ds, 'AdaptiveFusion', c) - mean(d, ds, bb, c):+.2f}$" if bb else "--")
        rows.append(r)
    T.append(latex_table("Accuracy difference (pp) of AdaptiveFusion relative to SP-RR, Uniform, and the best "
                         "external baseline.", "tab:gains",
                         ["Dataset", "M50 SP-RR", "M50 Unif.", "M50 best", "M70 SP-RR", "M70 Unif.", "M70 best"], rows,
                         star=True))
    rows = []
    for ds in dsets:
        for c in ["Missing50", "Missing70"]:
            rows.append([PRETTY[ds] if c == "Missing50" else "", c.replace("Missing", "M")] +
                        [cell(d, ds, m, c) for m in ["Uniform", "Const-beta", "SP-RR", "AdaptiveFusion"]])
    T.append(latex_table("Controller controls (ten seeds, accuracy \\%). Uniform: $\\beta=0$; Const-$\\beta$: $\\beta=0.5$; SP-RR: $\\beta=1$.",
                         "tab:controls", ["Dataset", "", "Uniform", "Const-$\\beta$", "SP-RR", "AdaptiveFusion"], rows,
                         star=True, colspec="llcccc"))
    # mean rank of each method over datasets, per condition
    rk = []
    for c in ALL:
        g = d[d.Condition == c].groupby(["Dataset", "Method"]).Accuracy.mean().reset_index()
        g = g[g.Method.isin(MAIN)]
        g["Rank"] = g.groupby("Dataset").Accuracy.rank(ascending=False)
        rk.append(g.groupby("Method").Rank.mean().rename(c))
    rk = pd.concat(rk, axis=1).reindex(MAIN)
    rk.round(2).to_csv(os.path.join(odir, "mean_rank.csv"))
    T.append(latex_table("Mean rank over the six datasets (lower is better).", "tab:rank",
                         ["Method"] + [c.replace("Missing", "M").replace("Conflict", "C").replace("Noise", "N") for c in ALL],
                         [[m] + [f"{rk.loc[m, c]:.1f}" for c in ALL] for m in MAIN], star=True,
                         colspec="l" + "c" * len(ALL)))
    T.append(latex_table("AdaptiveFusion accuracy (\\%) under semantic replacement and Gaussian noise, all views observed.",
                         "tab:corruption", ["Dataset", "Conflict30", "Conflict50", "Noise30", "Noise50"],
                         [[PRETTY[ds]] + [cell(d, ds, "AdaptiveFusion", c) for c in
                                          ["Conflict30", "Conflict50", "Noise30", "Noise50"]] for ds in dsets]))
    if len(pert):
        smax = pert.Sigma.max(); rows = []
        for ds in dsets:
            r = [PRETTY[ds]]
            for m in ["SP-RR", "Const-beta", "AdaptiveFusion"]:
                h = pert[(pert.Dataset == ds) & (pert.Method == m)]
                v = h[h.Sigma == smax].Acc.mean() - h[h.Sigma == 0].Acc.mean()
                r.append(f"${v:+.2f}$")
            rows.append(r)
        T.append(latex_table(f"Accuracy change (pp) under reliability perturbation $\\sigma={smax}$, Missing70.",
                             "tab:perturb", ["Dataset", "Fixed ($\\beta=1$)", "Const-$\\beta$", "AdaptiveFusion"], rows))
    if len(kd):
        rows = []
        Vmax = int(kd.K.max())
        for ds in dsets:
            for c in ["Missing30", "Missing50", "Missing70"]:
                h = kd[(kd.Dataset == ds) & (kd.Condition == c)].set_index("K").Fraction
                rows.append([PRETTY[ds] if c == "Missing30" else "", c] +
                            [f"{100 * h.get(k, 0):.1f}" if k in h.index else "--" for k in range(1, Vmax + 1)])
        T.append(latex_table("Realized distribution of the observed-view count $K_i$ on the test split (\\% of samples).",
                             "tab:kdist", ["Dataset", "Cond."] + [f"$K={k}$" for k in range(1, Vmax + 1)], rows,
                             star=True, colspec="ll" + "c" * Vmax))
    with open(os.path.join(odir, "latex_tables.tex"), "w", encoding="utf-8") as f:
        f.write("% Auto-generated by benchmark/report.py -- requires \\usepackage{booktabs}\n\n" + "\n".join(T))

    # ---------------- short markdown report ----------------
    lines = [f"# AdaptiveFusion results -- protocol {proto}", ""]
    nseed = d.groupby("Dataset").Seed.nunique()
    lines += ["Seeds per dataset: " + ", ".join(f"{k}={v}" for k, v in nseed.items()), ""]
    lines += ["## Overview: AdaptiveFusion vs each method over all conditions", "",
              "| Competitor | AF wins | AF losses | Holm-significant AF wins | Holm-significant AF losses |",
              "|---|---|---|---|---|"]
    for comp in sorted(st.Competitor.unique()):
        q = st[(st.Competitor == comp) & st.Condition.isin(SCARCE)]
        sig = q.Holm_t_p < 0.05
        lines.append(f"| {comp} | {(q.MeanDiff_pp > 0).sum()} | {(q.MeanDiff_pp < 0).sum()} | "
                     f"{(sig & (q.MeanDiff_pp > 0)).sum()} | {(sig & (q.MeanDiff_pp < 0)).sum()} |")
    lines += ["", "(scarcity conditions: Missing30/50/70 and the four joint conditions)", ""]
    key = st[(st.Competitor == "SP-RR") & st.Condition.isin(SCARCE)].copy()
    lines += ["## AdaptiveFusion vs SP-RR (scarcity conditions)", "",
              "| Dataset | Condition | Diff (pp) | t p | Holm p | dz |", "|---|---|---|---|---|---|"]
    for _, r in key.sort_values(["Dataset", "Condition"]).iterrows():
        lines.append(f"| {r.Dataset} | {r.Condition} | {r.MeanDiff_pp:+.2f} | {r.t_p:.4f} | {r.Holm_t_p:.4f} | {r.Cohen_dz:.2f} |")
    with open(os.path.join(odir, "REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
