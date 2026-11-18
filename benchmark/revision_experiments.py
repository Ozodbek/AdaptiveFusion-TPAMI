"""Revision experiments for the TPAMI AdaptiveFusion paper.

Reuses the original training code (adaptivefusion.train_model) and the
original test-condition generator (common.condition_data) unchanged, so the
base conditions reproduce runner.py exactly. Adds:
  * controller controls: Uniform (beta=0) and Const-beta (beta=0.5)
  * joint scarcity + corruption conditions
  * test-time reliability perturbation under Missing70
  * realized K distribution
"""
import argparse, os
import numpy as np, pandas as pd, torch
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score

from .common import (load_dataset, split_indices, standardize_train_only,
                     condition_data, missing_mask, CONDITIONS, SEEDS)
from .adaptivefusion import train_model, _robust_rel
from .gcca import fit_gcca, transform_gcca

COND_CODE = {c: i + 1 for i, c in enumerate(CONDITIONS)}   # same as runner.py
JOINT = ["Missing50+Noise50", "Missing70+Noise50",
         "Missing50+Conflict50", "Missing70+Conflict50"]
JOINT_CODE = {c: 50 + i for i, c in enumerate(JOINT)}
SIGMAS = [0.1, 0.2, 0.3, 0.5]
RULES = {"AdaptiveFusion": "adaptive", "SP-RR": "fixed",
         "Uniform": "zero", "Const-beta": "const"}


def beta_of(mode, K, V):
    if mode == "adaptive": return (K - 1) / max(V - 1, 1)
    if mode == "fixed": return torch.ones_like(K)
    if mode == "zero": return torch.zeros_like(K)
    if mode == "const": return torch.full_like(K, 0.5)
    raise ValueError(mode)


def fuse(S, P, M, R, mode):
    """Identical to adaptivefusion._fuse, with an extra 'const' mode."""
    V = S.shape[1]
    K = M.sum(1, keepdim=True).clamp_min(1)
    beta = beta_of(mode, K, V)
    w = M * ((1 - beta) + beta * R)
    sf = (w.unsqueeze(-1) * S).sum(1) / (w.sum(1, keepdim=True) + 1e-8)
    pf = (M.unsqueeze(-1) * P).sum(1) / (K + 1e-8)
    return torch.cat([sf, pf], 1)


@torch.no_grad()
def encode(model, X):
    dev = next(model.parameters()).device
    xs = [torch.tensor(x, device=dev) for x in X]
    o = model.encode_views(xs)
    S = torch.stack([q[0] for q in o], 1).cpu()
    P = torch.stack([q[1] for q in o], 1).cpu()
    E = torch.stack([torch.sqrt(((q[2] - q[3]) ** 2).mean(1) + 1e-8) for q in o], 1).cpu()
    return S, P, E


def metrics(y, p):
    return dict(Accuracy=100 * accuracy_score(y, p),
                MacroF1=100 * f1_score(y, p, average="macro"),
                BalancedAccuracy=100 * balanced_accuracy_score(y, p))


def joint_condition(X, cond, seed):
    """Missing mask first, then corrupt one observed view of 50% of the
    samples that keep at least two observed views."""
    miss, corr = cond.split("+")
    p = float(miss.replace("Missing", "")) / 100
    frac = float(''.join(c for c in corr if c.isdigit())) / 100
    n, V = len(X[0]), len(X)
    m = missing_mask(n, V, p, seed)
    Xc = [a.copy() for a in X]
    rng = np.random.default_rng(seed + 1)
    elig = np.where(m.sum(1) >= 2)[0]
    ids = rng.choice(elig, size=int(round(frac * len(elig))), replace=False) if len(elig) else []
    for i in ids:
        v = int(rng.choice(np.where(m[i] == 1)[0]))
        if corr.startswith("Noise"):
            Xc[v][i] = Xc[v][i] + rng.normal(0, 2.0, size=Xc[v][i].shape).astype(np.float32)
        else:
            j = int(rng.integers(n - 1)); j += j >= i
            Xc[v][i] = X[v][j]
    for v in range(V):
        Xc[v][m[:, v] == 0] = 0
    return Xc, m, len(ids) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--out-dir", default="./results/revision")
    ap.add_argument("--device", default="auto", help="auto, cpu or cuda")
    a = ap.parse_args()
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    os.makedirs(a.out_dir, exist_ok=True)
    out = os.path.join(a.out_dir, f"{a.dataset}.csv")
    kout = os.path.join(a.out_dir, f"{a.dataset}_K.csv")
    rows, krows = [], []
    if os.path.exists(out):
        rows = pd.read_csv(out).to_dict("records")
        krows = pd.read_csv(kout).to_dict("records") if os.path.exists(kout) else []
    done = {int(r["Seed"]) for r in rows}

    X, y = load_dataset(a.dataset, a.data_dir)
    C = len(np.unique(y)); V = len(X)
    for seed in [int(s) for s in a.seeds.split(",")]:
        if seed in done:
            continue
        tr, te = split_indices(y, seed, .30)
        Xtr, Xte = standardize_train_only(X, tr, te)
        ytr, yte = y[tr], y[te]
        model = train_model(Xtr, ytr, C, seed, a.epochs, device=dev)
        model.eval()
        mdir = os.path.join(a.out_dir, "models")
        os.makedirs(mdir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(mdir, f"{a.dataset}_seed{seed}.pt"))

        # training representations and one SVM per fusion rule
        Str, Ptr, Etr = encode(model, Xtr)
        med = Etr.median(0).values
        mad = (Etr - med).abs().median(0).values.clamp_min(1e-4)
        Mtr = torch.ones(len(ytr), V)
        Rtr = _robust_rel(Etr, med, mad) * Mtr
        clfs = {}
        for name, mode in RULES.items():
            ztr = fuse(Str, Ptr, Mtr, Rtr, mode).numpy()
            sc = StandardScaler().fit(ztr)
            clfs[name] = (sc, LinearSVC(C=1, dual="auto", max_iter=10000).fit(sc.transform(ztr), ytr))
        gpre, gA = fit_gcca(Xtr)
        gztr = transform_gcca(Xtr, Mtr.numpy(), gpre, gA)
        gsc = StandardScaler().fit(gztr)
        gclf = LinearSVC(C=1, dual="auto", max_iter=10000).fit(gsc.transform(gztr), ytr)

        def run_rules(Xc, m, cond, names=RULES, rel_noise=None, gcca=True, extra=None):
            S, P, E = encode(model, Xc)
            M = torch.tensor(m, dtype=torch.float32)
            R = _robust_rel(E, med, mad) * M
            if rel_noise is not None:
                R = (R + torch.tensor(rel_noise, dtype=torch.float32)).clamp(0, 1) * M
            for name in names:
                sc, clf = clfs[name]
                z = fuse(S, P, M, R, RULES[name]).numpy()
                rows.append(dict(Dataset=a.dataset, Method=name, Seed=seed, Condition=cond,
                                 **(extra or {}), **metrics(yte, clf.predict(sc.transform(z)))))
            if gcca:
                gz = transform_gcca(Xc, m, gpre, gA)
                rows.append(dict(Dataset=a.dataset, Method="MAXVAR-GCCA", Seed=seed, Condition=cond,
                                 **(extra or {}), **metrics(yte, gclf.predict(gsc.transform(gz)))))

        def log_K(cond, m, frac=None):
            K = m.sum(1)
            for k in range(1, V + 1):
                krows.append(dict(Dataset=a.dataset, Seed=seed, Condition=cond, K=k,
                                  Fraction=float((K == k).mean()), CorruptedFraction=frac))

        for cond in CONDITIONS:
            Xc, m = condition_data(Xte, cond, seed * 10007 + COND_CODE[cond] * 97)
            run_rules(Xc, m, cond)
            if cond.startswith("Missing"):
                log_K(cond, m)
        for cond in JOINT:
            Xc, m, frac = joint_condition(Xte, cond, seed * 10007 + JOINT_CODE[cond] * 97)
            run_rules(Xc, m, cond, extra=dict(CorruptedFraction=frac))
            log_K(cond, m, frac)
        # reliability perturbation under the Missing70 mask
        Xc, m = condition_data(Xte, "Missing70", seed * 10007 + COND_CODE["Missing70"] * 97)
        for sig in SIGMAS:
            rng = np.random.default_rng(seed * 31 + int(sig * 100))
            noise = rng.uniform(-sig, sig, size=m.shape)
            run_rules(Xc, m, f"Missing70+RelPert{sig}", rel_noise=noise, gcca=False,
                      extra=dict(Sigma=sig))

        pd.DataFrame(rows).to_csv(out, index=False)
        pd.DataFrame(krows).to_csv(kout, index=False)
        print(a.dataset, seed, "done", flush=True)


if __name__ == "__main__":
    main()
