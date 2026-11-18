"""Unified benchmark: every method, every dataset, every condition, ONE protocol.

Protocol (identical for all methods)
  * outer split: stratified 70/30, seeds 3,7,11,17,23,29,31,37,41,43;
    train-only standardization.
  * training data: the same simulated view removal for every learned method
    (nominal p in {0,.3,.5,.7}, at least one view kept, missing inputs zero-filled).
    Method-specific self-supervision (BML's synthetic conflict injection) is part
    of that method.
  * epochs: chosen per (method, seed) by 3-fold stratified inner cross-validation
    on the TRAINING split only, from a fixed grid; criterion = mean validation
    accuracy over Clean and Missing50. The final model is retrained on the whole
    training split with the selected number of epochs. Test data are never used.
  * test: the SAME 12 corrupted test sets for every method, generated once per seed.
  * classifier: a linear SVM (C=1) on standardized representations, for every
    method (end-to-end methods: BML -> routed fused logits, DIMvLN -> concatenated
    latents). Their own output layers are reported additionally as "[own head]".
  * Protocol A: one SVM fitted on CLEAN training representations, applied to every
    test condition (test-time condition unknown).
  * Protocol B: for each test condition, the SVM is fitted on training
    representations subjected to the SAME condition (condition-aware classifier).
    The trained network is the same in A and B; only the SVM differs.

Method groups
  AF      AdaptiveFusion, SP-RR (beta=1), Uniform (beta=0), Const-beta (beta=.5);
          one trained model, + reliability perturbation + K distribution
  CONCAT  per-view PCA (<=64, as in GCCA) + concatenation, no representation learning
  GCCA    regularized MAXVAR-GCCA
  DCCA    DCCA-style: view encoders + pairwise correlation regularizer
  BML     BML-unified: BML with an availability mask in routing and agreement
  DIMvLN  DIMvLN-unified: kNN completion + multi-view AE with ICL/CCL losses
"""
import argparse, copy, json, os, time, warnings
import numpy as np, pandas as pd, torch
import torch.nn as nn, torch.nn.functional as F
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

from .common import (load_dataset, split_indices, standardize_train_only,
                     condition_data, missing_mask, CONDITIONS, SEEDS, seed_all)
from .adaptivefusion import AdaptiveFusionNet, _robust_rel, _orth, _align, _proto_loss
from .gcca import fit_gcca, transform_gcca
from .revision_experiments import (fuse, encode, metrics, joint_condition,
                                   RULES, SIGMAS, JOINT, JOINT_CODE, COND_CODE)
from .bml_matched import BML, _inject
from .dimvln_matched import Model as DIMvLNModel, _icl, _ccl

from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Reproducible GPU runs: deterministic kernels.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)

ALL_CONDS = CONDITIONS + JOINT
TRAIN_RATES = [0., .3, .5, .7]
GROUPS = ["AF", "CONCAT", "GCCA", "DCCA", "BML", "DIMvLN"]
GRID = {"AF": [10, 18, 30, 50, 75, 100, 150],
        "DCCA": [25, 50, 100, 150, 200],
        "BML": [50, 100, 200],
        "DIMvLN": [50, 100, 200]}
QUICK_GRID = {"AF": [1, 2], "DCCA": [1, 2], "BML": [1, 2], "DIMvLN": [1, 2]}
PRIMARY = {"AF": "AdaptiveFusion", "CONCAT": "Concat-SVM", "GCCA": "MAXVAR-GCCA",
           "DCCA": "DCCA-style", "BML": "BML-unified", "DIMvLN": "DIMvLN-unified"}
N_FOLDS = 3


# ================================================================== helpers
def train_mask(n, V, ep, step, seed):
    """Simulated view removal shared by all learned methods (AdaptiveFusion's scheme)."""
    r = TRAIN_RATES[(ep + step) % len(TRAIN_RATES)]
    rng = np.random.default_rng(seed * 100000 + ep * 1000 + step)
    if r == 0:
        return np.ones((n, V), np.float32)
    m = (rng.random((n, V)) >= r).astype(np.float32)
    zero = np.where(m.sum(1) == 0)[0]
    if len(zero):
        m[zero, rng.integers(0, V, len(zero))] = 1
    return m


def apply_condition(X, cond, seed):
    """Base or joint condition; returns zero-filled views, mask."""
    if cond in JOINT:
        Xc, m, _ = joint_condition(X, cond, seed)
    else:
        Xc, m = condition_data(X, cond, seed)
    return [x.astype(np.float32) for x in Xc], m


def cond_seed(cond, seed, salt):
    code = JOINT_CODE[cond] if cond in JOINT else COND_CODE[cond]
    return seed * salt[0] + code * salt[1]


class _Projected:
    """Standardize, then (when dimension > samples) rotate onto the row space of the
    training matrix. For an L2-regularized linear SVM this gives exactly the same
    decision function (the weight vector lies in that row space) but trains much faster."""
    def __init__(self, ztr):
        self.sc = StandardScaler().fit(ztr)
        Z = self.sc.transform(ztr)
        self.Vt = None
        if Z.shape[1] > Z.shape[0]:
            _, S, Vt = np.linalg.svd(Z, full_matrices=False)
            self.Vt = Vt[S > S.max() * 1e-10]

    def transform(self, z):
        z = self.sc.transform(z)
        return z if self.Vt is None else z @ self.Vt.T


def svm_fit(ztr, ytr):
    """Linear SVM (C=1). Both liblinear solvers optimize the same objective; the primal
    solver is used for low-dimensional inputs and the dual one for high-dimensional
    inputs, purely for speed."""
    sc = _Projected(ztr)
    Z = sc.transform(ztr)
    return sc, LinearSVC(C=1, dual=Z.shape[1] > 256, max_iter=10000).fit(Z, ytr)


def svm_acc(ztr, ytr, zte, yte):
    sc, clf = svm_fit(ztr, ytr)
    return 100 * (clf.predict(sc.transform(zte)) == yte).mean()


def knn_complete(Xdon, X, m, k=10, self_index=False):
    """Vectorized equivalent of dimvln_matched._knn_complete with fully observed donors.
    Each missing view is filled with the mean of the k donors closest on the observed
    views (mean squared distance of L2-normalized features, averaged over views).
    self_index=True: rows of X are the donors themselves -> a row never uses itself."""
    V = len(X); n = len(X[0]); out = [x.copy() for x in X]
    Dn = [d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-8) for d in Xdon]
    Xn = [x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8) for x in X]
    dsq = [(d ** 2).sum(1) for d in Dn]
    patterns = {}
    for i in range(n):
        patterns.setdefault(tuple(int(t) for t in m[i]), []).append(i)
    for pat, idx in patterns.items():
        miss = [v for v in range(V) if pat[v] == 0]
        obs = [v for v in range(V) if pat[v] == 1]
        if not miss or not obs:
            continue
        idx = np.array(idx)
        for s in range(0, len(idx), 1024):
            q = idx[s:s + 1024]
            dist = np.zeros((len(q), len(Xdon[0])))
            for u in obs:
                a = Xn[u][q]
                dist += ((a ** 2).sum(1)[:, None] + dsq[u][None, :] - 2 * a @ Dn[u].T) / Xdon[u].shape[1]
            dist /= len(obs)
            if self_index:
                dist[np.arange(len(q)), q] = np.inf
            kk = min(k, dist.shape[1] - int(self_index))
            nn_ = np.argpartition(dist, kk - 1, axis=1)[:, :kk]
            for v in miss:
                out[v][q] = Xdon[v][nn_].mean(1)
    return [o.astype(np.float32) for o in out]


# ================================================================== method fitters
# Each fitter trains on (X, y) and yields (epochs, Rep) at every checkpoint, where
# Rep(Xc, m, train=False) -> {variant_name: representation ndarray}
# and optionally Rep.head(Xc, m) -> {variant_name: predictions}.

class Rep:
    def __init__(self, fn, head=None):
        self.fn = fn; self.head = head

    def __call__(self, Xc, m, train=False, **kw):
        return self.fn(Xc, m, train, **kw)


def _masked_median_mad(E, M):
    """Per-view median/MAD of residuals over OBSERVED entries only."""
    V = E.shape[1]; med = torch.ones(V, device=E.device); mad = torch.ones(V, device=E.device)
    for v in range(V):
        e = E[M[:, v] > 0, v]
        if len(e) >= 2:
            med[v] = e.median(); mad[v] = (e - med[v]).abs().median().clamp_min(1e-4)
    return med, mad


def fit_af(X, y, C, seed, checkpoints, dev):
    """AdaptiveFusion training loop (adaptivefusion.train_model), unchanged except:
    checkpoints are yielded, and the batch reliability calibration ignores missing
    (zero-filled) views."""
    seed_all(seed)
    model = AdaptiveFusionNet([x.shape[1] for x in X], C).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    TX = [torch.tensor(x, device=dev) for x in X]; TY = torch.tensor(y, dtype=torch.long, device=dev)
    N = len(y); V = len(X); bs = min(256, N)
    for ep in range(max(checkpoints)):
        model.train()
        perm = torch.randperm(N, device=dev)
        for st in range(0, N, bs):
            ids = perm[st:st + bs]
            if len(ids) < 2: continue
            xs = [x[ids] for x in TX]; yy = TY[ids]
            mt = torch.tensor(train_mask(len(ids), V, ep, st // bs, seed), device=dev)
            xmask = [x * mt[:, v:v + 1] for v, x in enumerate(xs)]
            outs = model.encode_views(xmask)
            S = torch.stack([o[0] for o in outs], 1); P = torch.stack([o[1] for o in outs], 1)
            E = torch.stack([torch.sqrt(((o[2] - o[3]) ** 2).mean(1) + 1e-8) for o in outs], 1)
            with torch.no_grad():
                medb, madb = _masked_median_mad(E, mt)
            rel = _robust_rel(E, medb, madb) * mt
            z = fuse(S, P, mt, rel, "adaptive"); logits = model.head(z)
            oc = model.encode_views(xs)
            Sc = torch.stack([o[0] for o in oc], 1); Pc = torch.stack([o[1] for o in oc], 1)
            Ec = torch.stack([torch.sqrt(((o[2] - o[3]) ** 2).mean(1) + 1e-8) for o in oc], 1)
            ones = torch.ones_like(mt)
            with torch.no_grad():
                medc, madc = _masked_median_mad(Ec, ones)
            zc = fuse(Sc, Pc, ones, _robust_rel(Ec, medc, madc), "adaptive"); lc = model.head(zc)
            loss = (F.cross_entropy(logits, yy) + F.cross_entropy(lc, yy)
                    + .22 * sum(F.mse_loss(o[3], o[2]) for o in outs) / V
                    + .06 * sum(_orth(o[0], o[1]) for o in outs) / V
                    + .08 * _align(S, mt) + .12 * _proto_loss(z, yy)
                    + .20 * (1 - F.cosine_similarity(z, zc).mean()))
            opt.zero_grad(); loss.backward(); opt.step()
        if ep + 1 in checkpoints:
            snap = copy.deepcopy(model).eval()
            yield ep + 1, af_rep(snap, X), snap


def af_rep(model, Xfit):
    Str, Ptr, Etr = encode(model, Xfit)
    med = Etr.median(0).values; mad = (Etr - med).abs().median(0).values.clamp_min(1e-4)

    def fn(Xc, m, train=False, rel_noise=None):
        S, P, E = encode(model, Xc); M = torch.tensor(m, dtype=torch.float32)
        R = _robust_rel(E, med, mad) * M
        if rel_noise is not None:
            R = (R + torch.tensor(rel_noise, dtype=torch.float32)).clamp(0, 1) * M
        return {name: fuse(S, P, M, R, mode).numpy() for name, mode in RULES.items()}
    return Rep(fn)


def fit_concat(X, y, C, seed, checkpoints, dev):
    """Concatenation baseline with the same per-view preprocessing as MAXVAR-GCCA:
    per-view PCA to at most 64 components (fitted on training data), concatenated.
    A missing (zero-filled) view maps to ~0 because inputs are standardized."""
    from sklearn.decomposition import PCA
    pcas = [PCA(min(64, x.shape[1], x.shape[0] - 1), random_state=0).fit(x) for x in X]
    return_fn = lambda Xc, m, train=False: {"Concat-SVM": np.hstack(
        [p.transform(x) * m[:, v:v + 1] for v, (p, x) in enumerate(zip(pcas, Xc))])}
    yield 0, Rep(return_fn), None


def fit_gcca_(X, y, C, seed, checkpoints, dev):
    pre, A = fit_gcca(X)
    yield 0, Rep(lambda Xc, m, train=False: {"MAXVAR-GCCA": transform_gcca(Xc, m, pre, A)}), None


class DCCANet(nn.Module):
    def __init__(self, dims, C, hidden=64, latent=32):
        super().__init__()
        self.enc = nn.ModuleList([nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, latent))
                                  for d in dims])
        self.head = nn.Linear(latent, C)

    def embed(self, xs, m):
        Z = torch.stack([e(x) for e, x in zip(self.enc, xs)], 1)
        return Z, (Z * m.unsqueeze(-1)).sum(1) / m.sum(1, keepdim=True).clamp_min(1)


def _corr_loss(Z, m):
    V = Z.shape[1]; loss = 0.; cnt = 0
    for u in range(V):
        for v in range(u + 1, V):
            ok = (m[:, u] * m[:, v]) > 0
            if ok.sum() < 3: continue
            a, b = Z[ok, u], Z[ok, v]
            a = (a - a.mean(0)) / (a.std(0) + 1e-6); b = (b - b.mean(0)) / (b.std(0) + 1e-6)
            loss = loss + (1 - (a * b).mean(0).mean()); cnt += 1
    return loss / max(cnt, 1)


def fit_dcca(X, y, C, seed, checkpoints, dev, lam=0.1):
    seed_all(seed); V = len(X)
    net = DCCANet([x.shape[1] for x in X], C).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    TX = [torch.tensor(x, device=dev) for x in X]; TY = torch.tensor(y, dtype=torch.long, device=dev)
    N = len(y); bs = min(256, N)
    for ep in range(max(checkpoints)):
        net.train(); perm = torch.randperm(N, device=dev)
        for st in range(0, N, bs):
            ids = perm[st:st + bs]
            if len(ids) < 3: continue
            m = torch.tensor(train_mask(len(ids), V, ep, st // bs, seed), device=dev)
            Z, f = net.embed([x[ids] * m[:, v:v + 1] for v, x in enumerate(TX)], m)
            loss = F.cross_entropy(net.head(f), TY[ids]) + lam * _corr_loss(Z, m)
            opt.zero_grad(); loss.backward(); opt.step()
        if ep + 1 in checkpoints:
            snap = copy.deepcopy(net).eval()

            @torch.no_grad()
            def fn(Xc, m, train=False, snap=snap):
                return {"DCCA-style": snap.embed([torch.tensor(x, device=dev) for x in Xc],
                                                 torch.tensor(m, dtype=torch.float32, device=dev))[1].cpu().numpy()}
            yield ep + 1, Rep(fn), None


def bml_forward(model, xs, m):
    """BML forward pass with an availability mask: missing views are excluded from
    cross-view agreement and from the routed fusion. Identical to the original BML
    forward pass when every view is observed."""
    Fv = [e(x) for e, x in zip(model.enc, xs)]
    L = torch.stack([c(f) for c, f in zip(model.cls, Fv)], 1)
    P = torch.softmax(L, -1).clamp_min(1e-8); logP = P.log()
    ent = (-(P * logP).sum(-1) / np.log(model.C)).clamp(0, 1)
    D = ((P.unsqueeze(2) - P.unsqueeze(1)) * (logP.unsqueeze(2) - logP.unsqueeze(1))).sum(-1)
    V = L.shape[1]; mj = m.unsqueeze(1) * (1 - torch.eye(V, device=L.device))
    agr = (D * mj).sum(-1) / mj.sum(-1).clamp_min(1)
    R = torch.cat([r(torch.cat([f, -torch.log((1 - ent[:, v:v + 1]).clamp_min(1e-8)), agr[:, v:v + 1]], 1))
                   for v, (r, f) in enumerate(zip(model.route, Fv))], 1)
    return ((R * m).unsqueeze(-1) * L).sum(1), R


def fit_bml(X, y, C, seed, checkpoints, dev):
    """BML uses a cosine schedule tied to the total number of epochs, so each grid
    value is a separate training run."""
    V = len(X); N = len(y); bs = min(2048, N)
    for epochs in sorted(checkpoints):
        seed_all(seed)
        model = BML([x.shape[1] for x in X], C).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=.002)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=1e-8)
        for ep in range(epochs):
            rng = np.random.default_rng(seed * 1000 + ep)
            xx, ind = _inject(X, y, .5, rng, one_view=True)
            ids = np.arange(N); rng.shuffle(ids); model.train()
            for k, st in enumerate(range(0, N, bs)):
                q = ids[st:st + bs]
                if len(q) < 2: continue
                m = torch.tensor(train_mask(len(q), V, ep, k, seed), device=dev)
                xs = [torch.tensor(x[q], device=dev) * m[:, v:v + 1] for v, x in enumerate(xx)]
                fused, r = bml_forward(model, xs, m)
                bce = (F.binary_cross_entropy(r, torch.tensor(ind[q], device=dev), reduction="none") * m).sum() / m.sum()
                loss = F.cross_entropy(fused, torch.tensor(y[q], dtype=torch.long, device=dev)) + bce
                opt.zero_grad(); loss.backward(); opt.step()
            sch.step()
        model.eval()

        @torch.no_grad()
        def fused_logits(Xc, m, model=model):
            return bml_forward(model, [torch.tensor(x, device=dev) for x in Xc],
                               torch.tensor(m, dtype=torch.float32, device=dev))[0].cpu().numpy()
        rep = Rep(lambda Xc, m, train=False, f=fused_logits: {"BML-unified": f(Xc, m)},
                  head=lambda Xc, m, f=fused_logits: {"BML-unified": f(Xc, m).argmax(1)})
        yield epochs, rep, None


def fit_dimvln(X, y, C, seed, checkpoints, dev):
    seed_all(seed); V = len(X); N = len(y)
    Xa = [[x] for x in X]; ya = [y]
    for k, p in enumerate(TRAIN_RATES[1:]):
        mk = missing_mask(N, V, p, seed * 7919 + k)
        Xk = knn_complete(X, [x * mk[:, v:v + 1] for v, x in enumerate(X)], mk, self_index=True)
        for v in range(V): Xa[v].append(Xk[v])
        ya.append(y)
    Xa = [np.concatenate(a).astype(np.float32) for a in Xa]; ya = np.concatenate(ya)
    model = DIMvLNModel([x.shape[1] for x in X], C).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    TX = [torch.tensor(x, device=dev) for x in Xa]; TY = torch.tensor(ya, dtype=torch.long, device=dev)
    bs = min(256, len(TY))
    for ep in range(max(checkpoints)):
        g = torch.Generator(device=dev); g.manual_seed(seed * 100000 + ep)
        perm = torch.randperm(len(TY), generator=g, device=dev); model.train()
        for st in range(0, len(TY), bs):
            ids = perm[st:st + bs]
            if len(ids) < 2: continue
            xs = [x[ids] for x in TX]; yy = TY[ids]
            R, Z, L = model(xs)
            loss = sum(F.mse_loss(r, x) for r, x in zip(R, xs)) / V
            loss = loss + sum(_icl(Z[i], Z[j]) for i in range(V) for j in range(i + 1, V))
            loss = loss + _ccl(torch.cat(Z, 1), yy, C) + F.cross_entropy(L, yy)
            opt.zero_grad(); loss.backward(); opt.step()
        if ep + 1 in checkpoints:
            snap = copy.deepcopy(model).eval()

            @torch.no_grad()
            def run(Xc, m, train=False, snap=snap):
                Xf = knn_complete(X, Xc, m, self_index=train)
                _, Z, L = snap([torch.tensor(x, device=dev) for x in Xf])
                return torch.cat(Z, 1).cpu().numpy(), L.argmax(1).cpu().numpy()
            rep = Rep(lambda Xc, m, train=False, r=run: {"DIMvLN-unified": r(Xc, m, train)[0]},
                      head=lambda Xc, m, r=run: {"DIMvLN-unified": r(Xc, m)[1]})
            yield ep + 1, rep, None


FITTERS = {"AF": fit_af, "CONCAT": fit_concat, "GCCA": fit_gcca_, "DCCA": fit_dcca,
           "BML": fit_bml, "DIMvLN": fit_dimvln}


# ================================================================== epoch selection
def select_epochs(group, Xtr, ytr, C, seed, grid, dev, log):
    """3-fold stratified inner CV on the training split. Score = mean validation
    accuracy (linear SVM on clean inner-train representations) over Clean and Missing50."""
    if group in ("CONCAT", "GCCA"):
        return 0
    V = len(Xtr); prim = PRIMARY[group]
    scores = {e: [] for e in grid}
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed)
    for f, (a, b) in enumerate(skf.split(np.zeros(len(ytr)), ytr)):
        Xa = [x[a] for x in Xtr]; Xb = [x[b] for x in Xtr]
        mb = missing_mask(len(b), V, .5, seed * 131 + f).astype(np.float32)
        Xb50 = [(x * mb[:, v:v + 1]).astype(np.float32) for v, x in enumerate(Xb)]
        ones_a = np.ones((len(a), V), np.float32); ones_b = np.ones((len(b), V), np.float32)
        for e, rep, _ in FITTERS[group](Xa, ytr[a], C, seed + 1000 * (f + 1), grid, dev):
            za = rep(Xa, ones_a, train=True)[prim]
            s = (svm_acc(za, ytr[a], rep(Xb, ones_b)[prim], ytr[b]) +
                 svm_acc(za, ytr[a], rep(Xb50, mb)[prim], ytr[b])) / 2
            scores[e].append(s)
    mean = {e: float(np.mean(v)) for e, v in scores.items()}
    best = max(sorted(mean), key=lambda e: mean[e])          # smallest epoch wins ties
    for e in grid:
        log.append(dict(Group=group, Seed=seed, Epochs=e, ValScore=mean[e], Selected=e == best))
    return best


# ================================================================== main
def run_group(group, ds, seed, Xtr, ytr, yte, T, TB, C, epochs, dev, out_dir):
    rows, krows = [], []
    V = len(Xtr); ones_tr = np.ones((len(ytr), V), np.float32)
    ck = [epochs] if epochs else [0]
    _, rep, model = next(iter(FITTERS[group](Xtr, ytr, C, seed, ck, dev)))
    if model is not None and group == "AF":
        os.makedirs(os.path.join(out_dir, "models"), exist_ok=True)
        torch.save(model.state_dict(), os.path.join(out_dir, "models", f"{ds}_AF_seed{seed}.pt"))
    extra = dict(SelectedEpochs=epochs)
    # Protocol A: one classifier from clean training representations
    ztr = rep(Xtr, ones_tr, train=True)
    clfA = {n: svm_fit(z, ytr) for n, z in ztr.items()}
    for cond, (Xc, m, frac) in T.items():
        for n, z in rep(Xc, m).items():
            sc, clf = clfA[n]
            rows.append(dict(Dataset=ds, Method=n, Seed=seed, Condition=cond, Protocol="A",
                             CorruptedFraction=frac, **extra, **metrics(yte, clf.predict(sc.transform(z)))))
        if rep.head is not None:
            for n, pred in rep.head(Xc, m).items():
                rows.append(dict(Dataset=ds, Method=f"{n} [own head]", Seed=seed, Condition=cond,
                                 Protocol="A", CorruptedFraction=frac, **extra, **metrics(yte, pred)))
    # Protocol B: classifier fitted on training data under the same condition
    for cond, (Xc, m, frac) in T.items():
        XtrB, mtrB = TB[cond]
        zB = rep(XtrB, mtrB.astype(np.float32), train=True)
        for n, z in rep(Xc, m).items():
            sc, clf = svm_fit(zB[n], ytr)
            rows.append(dict(Dataset=ds, Method=n, Seed=seed, Condition=cond, Protocol="B",
                             CorruptedFraction=frac, **extra, **metrics(yte, clf.predict(sc.transform(z)))))
    if group == "AF":
        Xc, m, _ = T["Missing70"]
        for sig in SIGMAS:
            noise = np.random.default_rng(seed * 31 + int(sig * 100)).uniform(-sig, sig, size=m.shape)
            for n, z in rep(Xc, m, rel_noise=noise).items():
                sc, clf = clfA[n]
                rows.append(dict(Dataset=ds, Method=n, Seed=seed, Condition=f"Missing70+RelPert{sig}",
                                 Protocol="A", Sigma=sig, **extra, **metrics(yte, clf.predict(sc.transform(z)))))
        for cond, (Xc, m, frac) in T.items():
            if cond.startswith("Missing"):
                K = m.sum(1)
                for k in range(1, V + 1):
                    krows.append(dict(Dataset=ds, Seed=seed, Condition=cond, K=k,
                                      Fraction=float((K == k).mean()), CorruptedFraction=frac))
    return rows, krows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--groups", default=",".join(GROUPS))
    ap.add_argument("--quick", action="store_true", help="tiny epoch grid, for testing only")
    ap.add_argument("--out-dir", default="./results/unified")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--grid", default=None,
                    help="comma-separated epoch grid used for EVERY learned method (overrides GRID)")
    a = ap.parse_args()
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    grid = QUICK_GRID if a.quick else GRID
    if a.grid and not a.quick:
        g = [int(x) for x in a.grid.split(",") if x]
        grid = {k: g for k in GRID}
    groups = [g for g in a.groups.split(",") if g]
    os.makedirs(a.out_dir, exist_ok=True)
    paths = {k: os.path.join(a.out_dir, f"{a.dataset}{s}") for k, s in
             [("rows", ".csv"), ("K", "_K.csv"), ("sel", "_epoch_selection.csv"), ("done", "_done.json")]}
    rows = pd.read_csv(paths["rows"]).to_dict("records") if os.path.exists(paths["rows"]) else []
    krows = pd.read_csv(paths["K"]).to_dict("records") if os.path.exists(paths["K"]) else []
    sel = pd.read_csv(paths["sel"]).to_dict("records") if os.path.exists(paths["sel"]) else []
    done = set(json.load(open(paths["done"]))) if os.path.exists(paths["done"]) else set()

    def save():
        pd.DataFrame(rows).to_csv(paths["rows"], index=False)
        if krows: pd.DataFrame(krows).to_csv(paths["K"], index=False)
        if sel: pd.DataFrame(sel).to_csv(paths["sel"], index=False)
        json.dump(sorted(done), open(paths["done"], "w"), indent=1)

    X, y = load_dataset(a.dataset, a.data_dir); C = len(np.unique(y))
    for seed in [int(s) for s in a.seeds.split(",")]:
        todo = [g for g in groups if f"{g}:{seed}" not in done]
        if not todo: continue
        tr, te = split_indices(y, seed, .30)
        Xtr, Xte = standardize_train_only(X, tr, te); ytr, yte = y[tr], y[te]
        T = {}
        for c in ALL_CONDS:
            Xc, m = apply_condition(Xte, c, cond_seed(c, seed, (10007, 97)))
            frac = joint_condition(Xte, c, cond_seed(c, seed, (10007, 97)))[2] if c in JOINT else None
            T[c] = (Xc, m.astype(np.float32), frac)
        TB = {c: apply_condition(Xtr, c, cond_seed(c, seed, (20011, 193))) for c in ALL_CONDS}
        for g in todo:
            t0 = time.time()
            ep = select_epochs(g, Xtr, ytr, C, seed, grid.get(g, [0]), dev, sel)
            r, k = run_group(g, a.dataset, seed, Xtr, ytr, yte, T, TB, C, ep, dev, a.out_dir)
            rows += r; krows += k; done.add(f"{g}:{seed}"); save()
            print(f"{a.dataset} seed={seed} {g} epochs={ep} done ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
