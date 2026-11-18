"""Corruption-detection AUROC computed from the saved v4 AdaptiveFusion checkpoints.

For every dataset and seed the checkpoint results/unified/models/<ds>_AF_seed<s>.pt is
loaded (no retraining). The same test sets as in unified.py are regenerated, and for
every observed (sample, view) pair two scores are computed:
  residual      e_v (projected reconstruction RMSE)            -- the score AdaptiveFusion uses
  disagreement  1 - cos(s_v, mean of the other observed s_u)   -- diagnostic only
A pair is "corrupted" when its input differs from the clean test input.
AUROC is reported per condition and, for the joint conditions, per observed-view count K.
Localization accuracy q_K: among samples with exactly one corrupted observed view, the
probability that the corrupted view receives the highest score (chance level 1/K).
"""
import argparse, os, glob
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from .common import load_dataset, split_indices, standardize_train_only
from .adaptivefusion import AdaptiveFusionNet
from .revision_experiments import encode
from .unified import apply_condition, cond_seed

CONDS = ["Conflict30", "Conflict50", "Noise30", "Noise50",
         "Missing50+Noise50", "Missing70+Noise50", "Missing50+Conflict50", "Missing70+Conflict50"]


def scores(model, Xc, m):
    S, P, E = encode(model, Xc)
    M = torch.tensor(m, dtype=torch.float32)
    V = S.shape[1]; dis = torch.zeros_like(E)
    for v in range(V):
        others = torch.cat([M[:, :v], M[:, v + 1:]], 1)
        So = torch.cat([S[:, :v], S[:, v + 1:]], 1)
        cnt = others.sum(1, keepdim=True)
        mean_o = (So * others.unsqueeze(-1)).sum(1) / cnt.clamp_min(1)
        d = 1 - F.cosine_similarity(S[:, v], mean_o, dim=1)
        dis[:, v] = torch.where(cnt.squeeze(1) > 0, d, torch.full_like(d, float("nan")))
    return E.numpy(), dis.numpy()


def auc(y, s):
    ok = ~np.isnan(s)
    y, s = y[ok], s[ok]
    if len(np.unique(y)) < 2: return np.nan
    return roc_auc_score(y, s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--models", default="./results/unified/models")
    ap.add_argument("--out", default="./results/auroc.csv")
    a = ap.parse_args()
    rows = []; loc = []
    for path in sorted(glob.glob(os.path.join(a.models, "*_AF_seed*.pt"))):
        base = os.path.basename(path)[:-3]; ds, seed = base.split("_AF_seed"); seed = int(seed)
        X, y = load_dataset(ds, a.data_dir); C = len(np.unique(y)); V = len(X)
        tr, te = split_indices(y, seed, .30); Xtr, Xte = standardize_train_only(X, tr, te)
        model = AdaptiveFusionNet([x.shape[1] for x in X], C)
        model.load_state_dict(torch.load(path, map_location="cpu")); model.eval()
        for c in CONDS:
            Xc, m = apply_condition(Xte, c, cond_seed(c, seed, (10007, 97)))
            corrupt = np.stack([np.any(np.abs(Xc[v] - Xte[v]) > 1e-6, axis=1) for v in range(V)], 1)
            corrupt &= m.astype(bool)                       # missing views are zero-filled, not corrupted
            E, D = scores(model, Xc, m)
            obs = m.astype(bool); K = m.sum(1)
            groups = [("all", obs)] + ([(f"K={k}", obs & (K[:, None] == k)) for k in range(1, V + 1)]
                                       if c.startswith("Missing") else [])
            # within-sample localization: samples with exactly one corrupted observed view;
            # success = that view has the strictly highest score (ties share the credit)
            one = (corrupt.sum(1) == 1) & (K >= 2)
            for k in sorted(set(K[one].tolist())):
                idx = np.where(one & (K == k))[0]
                for sname, Sc in [("residual", E), ("disagreement", D)]:
                    acc = []
                    for i in idx:
                        o = np.where(obs[i])[0]; sc = Sc[i, o]; cv = np.where(corrupt[i, o])[0][0]
                        top = np.flatnonzero(np.isclose(sc, sc.max(), rtol=0, atol=1e-7))
                        acc.append(1.0 / len(top) if cv in top else 0.0)
                    loc.append(dict(Dataset=ds, Seed=seed, Condition=c, K=k, Samples=len(idx),
                                    Score=sname, Localization=float(np.mean(acc)), Chance=1.0 / k))
            for gname, sel in groups:
                if sel.sum() == 0: continue
                rows.append(dict(Dataset=ds, Seed=seed, Condition=c, Group=gname, Pairs=int(sel.sum()),
                                 Corrupted=int((corrupt & sel).sum()),
                                 AUROC_residual=auc(corrupt[sel].astype(int), E[sel]),
                                 AUROC_disagreement=auc(corrupt[sel].astype(int), D[sel])))
        print(ds, seed, "done", flush=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)
    pd.DataFrame(loc).to_csv(a.out.replace(".csv", "_localization.csv"), index=False)
    print("written", a.out)


if __name__ == "__main__":
    main()
