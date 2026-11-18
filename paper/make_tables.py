"""Builds every LaTeX table and every number quoted in the paper from the v6 result CSVs in ../results/v6."""
import glob, json, os
import numpy as np, pandas as pd
from scipy.stats import ttest_rel, wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results", "v6")
V4 = os.path.join(RES, "unified")
V5 = os.path.join(RES, "v5")
OUT = os.path.join(HERE, "tables"); os.makedirs(OUT, exist_ok=True)
DS = ["MFeat", "Caltech101-20", "Scene15", "BDGP", "LandUse21", "Reuters1200"]
PR = {"MFeat": "MFeat", "Caltech101-20": "Caltech101-20", "Scene15": "Scene15", "BDGP": "BDGP",
      "LandUse21": "LandUse-21", "Reuters1200": "Reuters-1200"}
JOINT = ["Missing50+Noise50", "Missing70+Noise50", "Missing50+Conflict50", "Missing70+Conflict50"]
SCARCE = ["Missing30", "Missing50", "Missing70"] + JOINT
ALL = ["Clean", "Missing30", "Missing50", "Missing70", "Conflict30", "Conflict50", "Noise30", "Noise50"] + JOINT
SH = lambda c: (c.replace("Missing", "M").replace("Conflict", "C").replace("Noise", "N"))

# ---------------------------------------------------------------- load
v4 = pd.concat([pd.read_csv(f) for f in glob.glob(f"{V4}/*.csv") if not f.endswith(("_K.csv", "_selection.csv"))])
v5 = pd.concat([pd.read_csv(f) for f in glob.glob(f"{V5}/*.csv") if not f.endswith("_selection.csv")])
NAME5 = {"BML-full[train=fixed,test=fixed]": "BML", "BML-full[train=ctrl,test=ctrl]": "BML+ctrl",
         "BML-full[train=fixed,test=ctrl]": "BML+ctrl (test)", "BML-full[train=fixed,test=uniform]": "BML-uniform",
         "BML-small[train=fixed,test=fixed]": "BML-small", "BML-small[train=ctrl,test=ctrl]": "BML-small+ctrl",
         "BML-small[train=fixed,test=ctrl]": "BML-small+ctrl (test)", "BML-small[train=fixed,test=uniform]": "BML-small-uniform"}
v5s = v5[v5.Method.isin(NAME5)].copy(); v5s["Method"] = v5s.Method.map(NAME5)
v5s.loc[v5s.Head == "own", "Method"] += " [own]"
v4 = v4[~v4.Method.str.contains("own head")]
v4b = v4[v4.Method.isin(["BML-unified [own head]"])]
cols = ["Dataset", "Method", "Seed", "Condition", "Protocol", "Accuracy"]
D = pd.concat([v4[cols], v5s[cols]], ignore_index=True)
D = D[~D.Condition.str.contains("RelPert")]
NUM = {}


def m(ds, meth, c, p="A"):
    g = D[(D.Dataset == ds) & (D.Method == meth) & (D.Condition == c) & (D.Protocol == p)].Accuracy
    return g.mean() if len(g) else np.nan


def ms(ds, meth, c, p="A", bold=False):
    g = D[(D.Dataset == ds) & (D.Method == meth) & (D.Condition == c) & (D.Protocol == p)].Accuracy
    if not len(g): return "--"
    s = f"{g.mean():.2f}\\,{{\\scriptsize$\\pm${g.std(ddof=1):.2f}}}"
    return f"\\textbf{{{g.mean():.2f}}}\\,{{\\scriptsize$\\pm${g.std(ddof=1):.2f}}}" if bold else s


def mean_cell(ds, meth, c, p="A", bold=False):
    v = m(ds, meth, c, p)
    if np.isnan(v): return "--"
    return f"\\textbf{{{v:.2f}}}" if bold else f"{v:.2f}"


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); n = len(p)
    a = np.minimum(1, np.maximum.accumulate((n - np.arange(n)) * p[o])); r = np.empty(n); r[o] = a
    return r


def paired_cells(t, c_, p, conds=ALL):
    rows = []
    for ds in DS:
        for c in conds:
            A = D[(D.Dataset == ds) & (D.Method == t) & (D.Condition == c) & (D.Protocol == p)][["Seed", "Accuracy"]]
            B = D[(D.Dataset == ds) & (D.Method == c_) & (D.Condition == c) & (D.Protocol == p)][["Seed", "Accuracy"]]
            q = A.merge(B, on="Seed")
            if len(q) < 2: continue
            d = q.Accuracy_x - q.Accuracy_y
            pv = 1.0 if np.allclose(d, 0) else ttest_rel(q.Accuracy_x, q.Accuracy_y).pvalue
            try: wp = 1.0 if np.allclose(d, 0) else wilcoxon(d).pvalue
            except ValueError: wp = np.nan
            rows.append(dict(Dataset=ds, Condition=c, n=len(q), diff=d.mean(), p=pv, wp=wp))
    r = pd.DataFrame(rows)
    if len(r): r["holm"] = holm(r.p.values)
    return r


TITLES = {  # IEEE style: short title-case captions; the explanation goes below the table
    "tab:kdist": "Realized Distribution of the Number of Observed Views",
    "tab:epochs": "Training Epochs Selected by Inner Cross-Validation",
    "tab:mainA": "Accuracy (\\%) under Protocol~A",
    "tab:mainB": "Accuracy (\\%) under Protocol~B",
    "tab:joint": "Accuracy (\\%) under Joint Missingness and Corruption",
    "tab:paired": "Paired Comparison of AdaptiveFusion with Each Method",
    "tab:controls": "Fusion-Rule Controls on the Same Trained Model",
    "tab:perturb": "Accuracy Change under Perturbed Reliability Scores",
    "tab:auroc": "Corruption-Detection AUROC of Reliability Scores",
    "tab:localization": "Localization Accuracy for a Single Corrupted View",
    "tab:capacity": "Capacity Control",
    "tab:plugin": "The Availability Controller as a Plug-In for BML",
}


def table(env, cap, lab, colspec, header, rows, size="\\footnotesize", pre=None, colsep="3pt"):
    title = TITLES.get(lab, cap)
    s = [f"\\begin{{{env}}}[!t]", f"\\caption{{{title}}}", f"\\label{{{lab}}}", "\\centering", size,
         f"\\setlength{{\\tabcolsep}}{{{colsep}}}", f"\\begin{{tabular}}{{{colspec}}}", "\\toprule"]
    s += (pre or []) + [" & ".join(header) + " \\\\", "\\midrule"]
    for r in rows:
        s.append(r if isinstance(r, str) else " & ".join(r) + " \\\\")
    s += ["\\bottomrule", "\\end{tabular}"]
    if title != cap:
        s += ["\\par\\smallskip", f"\\parbox{{\\linewidth}}{{\\scriptsize {cap}}}"]
    s += [f"\\end{{{env}}}", ""]
    return "\n".join(s)


def save(name, txt):
    open(f"{OUT}/{name}.tex", "w").write(txt)


# ---------------------------------------------------------------- K distribution
K = pd.concat([pd.read_csv(f) for f in glob.glob(f"{V4}/*_K.csv")])
K = K[K.Condition.isin(["Missing30", "Missing50", "Missing70"])].groupby(["Dataset", "Condition", "K"]).Fraction.mean()
rows = []
for ds in DS:
    for i, c in enumerate(["Missing30", "Missing50", "Missing70"]):
        h = K.loc[(ds, c)]; V = int(h.index.max())
        mid = sum(h.get(k, 0) for k in range(2, V))
        cells = [f"{100 * h.get(k, 0):.1f}" if k <= V else "--" for k in range(1, 7)]
        rows.append([PR[ds] if i == 0 else "", c.replace("Missing", "M")] + cells + [f"\\textbf{{{100 * mid:.1f}}}"])
        NUM[f"kmid_{ds}_{c}"] = 100 * mid
    if ds != DS[-1]: rows.append("\\addlinespace[2pt]")
save("kdist", table("table", "Realized distribution of the observed-view count $K_i$ on the test split "
     "(\\% of samples, mean over ten seeds). The last column is the share of samples with $2\\le K_i\\le V-1$, "
     "the only regime in which AdaptiveFusion and fixed reliability weighting differ.", "tab:kdist",
     "llccccccc", ["Dataset", "Cond.", "$K{=}1$", "$2$", "$3$", "$4$", "$5$", "$6$", "$2{\\le}K{<}V$"], rows))

# ---------------------------------------------------------------- main tables
FEAT = ["Concat-SVM", "MAXVAR-GCCA", "DIMvLN-unified", "DCCA-style", "AdaptiveFusion"]
DEC = ["BML-small", "BML"]
LBL = {"Concat-SVM": "Concat-SVM", "MAXVAR-GCCA": "MAXVAR-GCCA", "DIMvLN-unified": "DIMvLN$^\\dagger$",
       "DCCA-style": "DCCA-style", "AdaptiveFusion": "\\textbf{AdaptiveFusion}",
       "BML-small": "BML-CM$^\\dagger$", "BML": "BML$^\\dagger$"}


def main_table(p, conds, cap, lab, env="table*", std=True, colsep="3pt"):
    rows = []
    for ds in DS:
        best = {c: max(FEAT, key=lambda mm: -np.inf if np.isnan(m(ds, mm, c, p)) else m(ds, mm, c, p)) for c in conds}
        for i, meth in enumerate(FEAT + DEC):
            if meth == DEC[0]: rows.append("\\cmidrule(lr){2-%d}" % (2 + len(conds)))
            rows.append([f"\\multirow{{{len(FEAT) + len(DEC)}}}{{*}}{{{PR[ds]}}}" if i == 0 else "", LBL[meth]] +
                        [(ms(ds, meth, c, p, bold=(meth in FEAT and best[c] == meth)) if std else
                          mean_cell(ds, meth, c, p, bold=(meth in FEAT and best[c] == meth))) for c in conds])
        if ds != DS[-1]: rows.append("\\midrule")
    return table(env, cap, lab, "ll" + "c" * len(conds), ["Dataset", "Method"] + conds, rows, colsep=colsep)


save("mainA", main_table("A", ["Clean", "Missing30", "Missing50", "Missing70", "Conflict50", "Noise50"],
     "Accuracy (\\%, mean\\,$\\pm$\\,std over ten seeds) under Protocol~A (one classifier fitted on clean training "
     "representations). Bold: best feature-level method. Below the inner rule: decision-level BML at matched "
     "capacity (BML-CM) and at full width (BML), for reference. $^\\dagger$Re-implemented under the common protocol.",
     "tab:mainA"))
save("mainB", main_table("B", ["Missing30", "Missing50", "Missing70", "Conflict50", "Noise50"],
     "Mean accuracy (\\%) over ten seeds under Protocol~B (classifier fitted on training data subjected to the test "
     "condition). Clean results coincide with Protocol~A and are omitted. Bold: best feature-level method. "
     "Standard deviations are released with the per-seed results.", "tab:mainB", env="table", std=False,
     colsep="2.5pt").replace(" & ".join(["Dataset", "Method", "Missing30", "Missing50", "Missing70", "Conflict50", "Noise50"]),
     " & ".join(["Dataset", "Method", "M30", "M50", "M70", "C50", "N50"])))
save("joint", main_table("A", JOINT,
     "Mean accuracy (\\%) over ten seeds under joint missingness and corruption, Protocol~A. Half of the samples "
     "that keep at least two observed views have one observed view corrupted. Bold: best feature-level method. "
     "Standard deviations are released with the per-seed results.", "tab:joint", env="table", std=False,
     colsep="2.5pt").replace(
     " & ".join(["Dataset", "Method"] + JOINT), "& & \\multicolumn{2}{c}{+Noise50} & \\multicolumn{2}{c}{+Conflict50} \\\\\n"
     "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\n"
     "Dataset & Method & M50 & M70 & M50 & M70"))

# mean ranks (feature-level, by condition group)
GR = {"Clean": ["Clean"], "Missing": ["Missing30", "Missing50", "Missing70"],
      "Corrupted": ["Conflict30", "Conflict50", "Noise30", "Noise50"], "Joint": JOINT}
for p in "AB":
    g = D[(D.Protocol == p) & D.Method.isin(FEAT)].groupby(["Dataset", "Condition", "Method"]).Accuracy.mean().reset_index()
    g["R"] = g.groupby(["Dataset", "Condition"]).Accuracy.rank(ascending=False)
    for k, v in GR.items():
        for meth in FEAT:
            NUM[f"rank_{p}_{k}_{meth}"] = g[g.Condition.isin(v) & (g.Method == meth)].R.mean()
    first = g[g.R == 1].Method.value_counts()
    for meth in FEAT: NUM[f"first_{p}_{meth}"] = int(first.get(meth, 0))

# ---------------------------------------------------------------- paired summary
COMPS = [("Concat-SVM", "Concat"), ("MAXVAR-GCCA", "GCCA"), ("DIMvLN-unified", "DIMvLN$^\\dagger$"),
         ("DCCA-style", "DCCA-st."), ("BML-small", "BML-CM$^\\dagger$"), ("BML", "BML$^\\dagger$")]
rows = []
for comp, lab in COMPS:
    for p in "AB":
        r = paired_cells("AdaptiveFusion", comp, p)
        cells = []
        for k, v in GR.items():
            q = r[r.Condition.isin(v)]
            if not len(q): cells.append("--"); continue
            sw = int(((q.holm < .05) & (q["diff"] > 0)).sum()); sl = int(((q.holm < .05) & (q["diff"] < 0)).sum())
            cells.append(f"${q['diff'].mean():+.1f}$\\,({sw}/{sl})")
            NUM[f"pair_{comp}_{p}_{k}"] = (q["diff"].mean(), sw, sl, len(q))
        rows.append([lab if p == "A" else "", p] + cells)
save("paired", table("table", "AdaptiveFusion versus each method: mean accuracy difference (pp) over the "
     "dataset--condition cells of each group, and the number of cells with a Holm-significant win/loss "
     "for AdaptiveFusion (paired $t$-test over ten seeds; Holm correction over all 72 cells of each comparison). "
     "Cells per group: Clean 6, Missing 18, Corrupted 24, Joint 24. Counts quoted for the 42 scarcity cells use a separate Holm correction over those cells.", "tab:paired", "llcccc",
     ["Competitor", "P.", "Clean", "Missing", "Corrupt.", "Joint"], rows, colsep="2pt"))

# ---------------------------------------------------------------- controller controls
RULES = [("Uniform", "Uniform"), ("Const-beta", "Const-$\\beta$"), ("SP-RR", "SP-RR"), ("AdaptiveFusion", "AF")]
CC = ["Missing50", "Missing70", "Missing70+Conflict50"]
SHORT_DS = {"MFeat": "MFeat", "Caltech101-20": "Caltech", "Scene15": "Scene15", "BDGP": "BDGP",
            "LandUse21": "LandUse", "Reuters1200": "Reuters"}
rows = []
for ds in DS:
    for k, c in enumerate(CC):
        r = [f"\\multirow{{{len(CC)}}}{{*}}{{{SHORT_DS[ds]}}}" if k == 0 else "", SH(c)]
        for p in "AB":
            vals = [m(ds, key, c, p) for key, _ in RULES]; top = max(vals)
            r += [f"\\textbf{{{v:.1f}}}" if np.isclose(v, top) else f"{v:.1f}" for v in vals]
        rows.append(r)
    if ds != DS[-1]: rows.append("\\midrule")
pre = ["& & \\multicolumn{4}{c}{Protocol A} & \\multicolumn{4}{c}{Protocol B} \\\\",
       "\\cmidrule(lr){3-6}\\cmidrule(lr){7-10}"]
save("controls", table("table", "Fusion-rule controls on the same trained AdaptiveFusion model (accuracy \\%). "
     "U: Uniform, $\\beta_i=0$ (reliability ignored); C$\\beta$: Const-$\\beta$, $\\beta_i=0.5$; SP: SP-RR, "
     "$\\beta_i=1$ (fixed reliability weighting); AF: AdaptiveFusion, $\\beta_i=(K_i-1)/(V-1)$. Bold: best rule.",
     "tab:controls", "ll" + "cccc" * 2, ["Dataset", "Cond.", "U", "C$\\beta$", "SP", "AF", "U", "C$\\beta$", "SP", "AF"],
     rows, pre=pre, colsep="2.2pt"))
for comp in ["SP-RR", "Uniform", "Const-beta"]:
    for p in "AB":
        r = paired_cells("AdaptiveFusion", comp, p)
        q = r[r.Condition.isin(SCARCE)]
        NUM[f"ctrl_{comp}_{p}"] = (q["diff"].mean(), int((q["diff"] > 0).sum()), int((q["diff"] < 0).sum()),
                                   int(((q.holm < .05) & (q["diff"] > 0)).sum()), int(((q.holm < .05) & (q["diff"] < 0)).sum()), len(q))
        if comp == "SP-RR":
            NUM[f"spr_clean_{p}"] = r[r.Condition == "Clean"]["diff"].mean()
NUM["caltech_clean_uniform"] = m("Caltech101-20", "Uniform", "Clean")
NUM["caltech_clean_af"] = m("Caltech101-20", "AdaptiveFusion", "Clean")

# ---------------------------------------------------------------- perturbation
P = v4[v4.Condition.str.contains("RelPert") | (v4.Condition == "Missing70")]
P = P[(P.Protocol == "A") & P.Method.isin(["AdaptiveFusion", "SP-RR", "Const-beta", "Uniform"])]
P = P.assign(Sigma=P.Sigma.fillna(0.0))
P.loc[P.Condition == "Missing70", "Sigma"] = 0.0
pt = P.groupby(["Dataset", "Method", "Sigma"]).Accuracy.mean()
smax = P.Sigma.max(); rows = []
for ds in DS:
    r = [PR[ds]]
    for meth in ["SP-RR", "Const-beta", "Uniform", "AdaptiveFusion"]:
        dlt = pt[(ds, meth, smax)] - pt[(ds, meth, 0.0)]; r.append(f"${dlt:+.2f}$"); NUM[f"pert_{ds}_{meth}"] = dlt
    rows.append(r)
save("perturb", table("table", f"Accuracy change (pp) when the test-time reliability scores are perturbed by "
     f"i.i.d.\\ $\\mathcal U[-\\sigma,\\sigma]$ noise ($\\sigma={smax}$), Missing70, Protocol~A, same trained model.",
     "tab:perturb", "lcccc", ["Dataset", "SP-RR", "Const-$\\beta$", "Uniform", "AF"], rows))

# ---------------------------------------------------------------- AUROC + localization
A = pd.read_csv(os.path.join(RES, "auroc.csv")); L = pd.read_csv(os.path.join(RES, "auroc_localization.csv"))
a = A[A.Group == "all"].groupby(["Dataset", "Condition"])[["AUROC_residual", "AUROC_disagreement"]].mean()
rows = []
for ds in DS:
    r = [PR[ds]]
    for c in ["Noise50", "Conflict50"]:
        r += [f"{a.loc[(ds, c), 'AUROC_residual']:.3f}", f"{a.loc[(ds, c), 'AUROC_disagreement']:.3f}"]
    rows.append(r)
    NUM[f"auc_{ds}"] = {c: tuple(a.loc[(ds, c)]) for c in ["Noise50", "Conflict50"]}
pre = ["& \\multicolumn{2}{c}{Noise50} & \\multicolumn{2}{c}{Conflict50} \\\\", "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}"]
save("auroc", table("table", "Corruption-detection AUROC of the residual score used by AdaptiveFusion and of "
     "cross-view disagreement (diagnostic only), computed from the trained models over all observed "
     "(sample, view) pairs; mean over ten seeds.", "tab:auroc", "lcccc",
     ["Dataset", "Resid.", "Disagr.", "Resid.", "Disagr."], rows, pre=pre))
lc = L[L.Condition.str.contains("Conflict")].groupby(["Dataset", "K", "Score"]).Localization.mean()
rows = []
for sc, lab in [("disagreement", "Disagreement"), ("residual", "Residual")]:
    rows.append(f"\\multicolumn{{6}}{{l}}{{\\emph{{{lab}}}}} \\\\")
    for ds in DS:
        r = [PR[ds]]
        for k in range(2, 7):
            r.append(f"{lc[(ds, k, sc)]:.3f}" if (ds, k, sc) in lc.index else "--")
        rows.append(r)
rows.append("\\midrule")
rows.append(["Chance $1/K$"] + [f"{1 / k:.3f}" for k in range(2, 7)])
save("localization", table("table", "Localization accuracy $q_K$: among samples with exactly one corrupted (conflicting) "
     "observed view under Missing50/70+Conflict50, the probability that the corrupted view receives the highest "
     "unreliability score, by the number $K$ of observed views (ties share credit). Mean over ten seeds.",
     "tab:localization", "lccccc", ["Dataset", "$K{=}2$", "$3$", "$4$", "$5$", "$6$"], rows))
NUM["loc"] = {f"{d}_{k}_{s}": v for (d, k, s), v in lc.items()}

# ---------------------------------------------------------------- capacity (v5)
cfg = {json.load(open(f))["dataset"]: json.load(open(f)) for f in glob.glob(f"{V5}/*_config.json")}
rows = []
for ds in DS:
    c = cfg[ds]
    params = [f"{c['AdaptiveFusion_params'] / 1e3:.0f}k", f"{c['BML_small_params'] / 1e3:.0f}k",
              f"{c['BML_full_params'] / 1e6:.1f}M"]
    for k, p in enumerate("AB"):
        r = ([f"\\multirow{{2}}{{*}}{{{SHORT_DS[ds]}}}"] + [f"\\multirow{{2}}{{*}}{{{x}}}" for x in params]) if k == 0 \
            else ["", "", "", ""]
        r.append(p)
        for meth in ["AdaptiveFusion", "BML-small", "BML"]:
            r.append(f"{np.mean([m(ds, meth, cc, p) for cc in SCARCE]):.1f}")
        q = paired_cells("AdaptiveFusion", "BML-small", p, SCARCE)
        q = q[q.Dataset == ds]
        r.append(f"{int(((q.holm < .05) & (q['diff'] > 0)).sum())}/{int(((q.holm < .05) & (q['diff'] < 0)).sum())}")
        rows.append(r)
    if ds != DS[-1]: rows.append("\\addlinespace[2pt]")
pre = ["& \\multicolumn{3}{c}{Parameters} & & \\multicolumn{3}{c}{Accuracy} & \\\\",
       "\\cmidrule(lr){2-4}\\cmidrule(lr){6-8}"]
save("capacity", table("table", "Capacity control. Parameter counts of AdaptiveFusion (AF), of BML with its width "
     "reduced to match AdaptiveFusion (CM), and of full-width BML; mean accuracy (\\%) over the seven scarcity "
     "conditions (Missing30/50/70 and the four joint conditions) under Protocols A and B (P.); and the number of "
     "these conditions in which AdaptiveFusion is Holm-significantly better/worse than capacity-matched BML (W/L; Holm "
     "correction over the 42 scarcity cells).", "tab:capacity",
     "lccccccccc"[:9], ["Dataset", "AF", "CM", "BML", "P.", "AF", "CM", "BML", "W/L"], rows, pre=pre, colsep="2.2pt"))
for p in "AB":
    q = paired_cells("AdaptiveFusion", "BML-small", p, SCARCE)
    NUM[f"cap_{p}"] = (q["diff"].mean(), int(((q.holm < .05) & (q["diff"] > 0)).sum()), int(((q.holm < .05) & (q["diff"] < 0)).sum()), len(q))
    q2 = paired_cells("BML", "BML-small", p, SCARCE)
    NUM[f"capgap_{p}"] = q2["diff"].mean()

# ---------------------------------------------------------------- plug-in (v5)
rows = []
for ds in DS:
    r = [PR[ds]]
    for t in ["BML+ctrl", "BML+ctrl (test)"]:
        q = paired_cells(t, "BML", "A", SCARCE); q = q[q.Dataset == ds]
        r.append(f"${q['diff'].mean():+.2f}$\\,({int(((q.holm < .05) & (q['diff'] > 0)).sum())}/{int(((q.holm < .05) & (q['diff'] < 0)).sum())})")
    for t, b, p in [("BML+ctrl", "BML", "B"), ("BML+ctrl [own]", "BML [own]", "A")]:
        q = paired_cells(t, b, p, SCARCE); q = q[q.Dataset == ds]; r.append(f"${q['diff'].mean():+.2f}$")
    rows.append(r)
pre = ["& \\multicolumn{2}{c}{Protocol A: $\\Delta$ (sig.\\ W/L)} & \\multicolumn{2}{c}{Other read-outs: $\\Delta$} \\\\",
       "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}"]
save("plugin", table("table", "The availability controller as a plug-in for BML (full width). $\\Delta$: mean accuracy "
     "difference (pp) to BML over the seven scarcity conditions, with the number of Holm-significant wins/losses. "
     "``Trained'': BML trained and tested with $w_i^{(v)}=m_i^{(v)}[(1-\\beta_i)+\\beta_iR_i^{(v)}]$; ``test-time'': original BML with "
     "the controller applied only at inference. Right: the trained variant under Protocol~B and with BML's own "
     "output layer.", "tab:plugin",
     "lcccc", ["Dataset", "Trained", "Test-time", "Prot.\\ B", "Own head"], rows, pre=pre, colsep="2.5pt"))
for t, b, p, key in [("BML+ctrl", "BML", "A", "plug_tr_A"), ("BML+ctrl (test)", "BML", "A", "plug_te_A"),
                     ("BML+ctrl", "BML", "B", "plug_tr_B"), ("BML+ctrl [own]", "BML [own]", "A", "plug_own"),
                     ("BML", "BML-uniform", "A", "router_A"), ("BML", "BML-uniform", "B", "router_B"),
                     ("BML-small+ctrl", "BML-small", "A", "plug_small_A"), ("BML+ctrl", "BML", "A", "plug_clean")]:
    conds = ["Clean"] if key == "plug_clean" else SCARCE
    q = paired_cells(t, b, p, conds)
    NUM[key] = (q["diff"].mean(), int((q["diff"] > 0).sum()), int((q["diff"] < 0).sum()),
                int(((q.holm < .05) & (q["diff"] > 0)).sum()), int(((q.holm < .05) & (q["diff"] < 0)).sum()), len(q))
for ds in DS:
    NUM[f"plug_m70c50_{ds}"] = (m(ds, "BML", "Missing70+Conflict50"), m(ds, "BML+ctrl", "Missing70+Conflict50"))

# ---------------------------------------------------------------- selected epochs
E4 = pd.concat([pd.read_csv(f).assign(Dataset=os.path.basename(f).replace("_epoch_selection.csv", ""))
                for f in glob.glob(f"{V4}/*_epoch_selection.csv")])
E5 = pd.concat([pd.read_csv(f).assign(Dataset=os.path.basename(f).replace("_epoch_selection.csv", ""))
                for f in glob.glob(f"{V5}/*_epoch_selection.csv")])
E = pd.concat([E4[E4.Selected & E4.Group.isin(["AF", "DCCA", "DIMvLN"])], E5[E5.Selected & E5.Group.isin(["FULL_FIXED", "SMALL_FIXED"])]])
lab = {"AF": "AF", "DCCA": "DCCA-style", "DIMvLN": "DIMvLN$^\\dagger$", "SMALL_FIXED": "BML-CM$^\\dagger$", "FULL_FIXED": "BML$^\\dagger$"}
grids = {g: "25--400" for g in ["AF", "DCCA", "DIMvLN", "SMALL_FIXED", "FULL_FIXED"]}
rows = []
for g in ["AF", "DCCA", "DIMvLN", "SMALL_FIXED", "FULL_FIXED"]:
    r = [lab[g]]
    for ds in DS:
        x = E[(E.Group == g) & (E.Dataset == ds)].Epochs
        r.append(f"{int(np.median(x))}")
    top = 400
    r.append(f"{100 * (E[E.Group == g].Epochs == top).mean():.0f}\\%")
    rows.append(r)
GRIDNOTE = "; ".join(f"{lab[g]}: {grids[g]}" for g in ["AF", "DCCA", "DIMvLN", "SMALL_FIXED", "FULL_FIXED"])
save("epochs", table("table", "Training epochs selected by three-fold inner cross-validation on the training split "
     "(median over ten seeds) and the share of selections at the upper end of the grid. AF: AdaptiveFusion. "
     "All learned methods select from the same grid $\\{25,50,100,200,400\\}$.", "tab:epochs",
     "lccccccc", ["Method", "MFeat", "Calt.", "Sc15", "BDGP", "LU21", "Reut.", "At max."], rows, colsep="2.5pt"))

json.dump({k: (v if not isinstance(v, (np.floating, np.integer)) else float(v)) for k, v in NUM.items()},
          open(os.path.join(HERE, "numbers.json"), "w"), indent=1, default=float)
print("tables:", sorted(os.listdir(OUT)))
