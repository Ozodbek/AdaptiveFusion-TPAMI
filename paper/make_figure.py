"""Draws Fig. 1 (perturbation.pdf) from results/v6/report/protocol_A/perturbation.csv.

Identical to the figure written by benchmark/report.py, except that the legend uses the
method names of the paper (Const-$\\beta$ instead of Const-beta)."""
import argparse, os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# TrueType (Type 42) fonts: IEEE PDF eXpress rejects Type 3 fonts
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42,
                            "font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans"})
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT = os.path.join(HERE, "..", "results", "v6", "report", "protocol_A", "perturbation.csv")
DS = ["MFeat", "Caltech101-20", "Scene15", "BDGP", "LandUse21", "Reuters1200"]
PRETTY = {"MFeat": "MFeat", "Caltech101-20": "Caltech101-20", "Scene15": "Scene15",
          "BDGP": "BDGP", "LandUse21": "LandUse-21", "Reuters1200": "Reuters-1200"}
STYLE = {"AdaptiveFusion": ("#1f5fa8", "o", "-"), "SP-RR": ("#c0392b", "s", "--"),
         "Const-beta": ("#7f8c8d", "^", ":"), "Uniform": ("#27ae60", "d", "-.")}
LABEL = {"AdaptiveFusion": "AdaptiveFusion", "SP-RR": "SP-RR",
         "Const-beta": r"Const-$\beta$", "Uniform": "Uniform"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DEFAULT)
    ap.add_argument("--out", default=os.path.join(HERE, "perturbation.pdf"))
    a = ap.parse_args()
    pert = pd.read_csv(a.csv)
    dsets = [d for d in DS if d in set(pert.Dataset)]
    n = len(dsets); cols = 3; rows = int(np.ceil(n / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(10, 3 * rows), squeeze=False)
    for ax, ds in zip(axs.ravel(), dsets):
        for meth, (c, mk, ls) in STYLE.items():
            h = pert[(pert.Dataset == ds) & (pert.Method == meth)].sort_values("Sigma")
            if len(h):
                ax.plot(h.Sigma, h.Acc, color=c, marker=mk, ls=ls, label=LABEL[meth], ms=4)
        ax.set_title(PRETTY[ds], fontsize=10); ax.set_xlabel(r"perturbation $\sigma$")
        ax.set_ylabel("accuracy (%)"); ax.grid(alpha=.3)
    for ax in axs.ravel()[n:]:
        ax.axis("off")
    axs.ravel()[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(a.out)
    plt.close(fig)
    print("written:", a.out)


if __name__ == "__main__":
    main()
