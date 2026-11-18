"""v5 experiments: (1) capacity-matched BML and (2) BML + availability controller.

Everything else is identical to the v4 unified protocol (benchmark/unified.py):
same splits, seeds, preprocessing, simulated view removal during training,
3-fold inner-CV epoch selection, the same 12 test sets per seed, linear-SVM
protocols A and B, and the methods' own output layer reported as "own".

Trained models (one per seed each)
  FULL_FIXED   BML-unified as in v4 (width 1024), fusion  sum_v m_v R_v L_v
  FULL_CTRL    same network, trained with the availability controller:
               fusion  sum_v m_v [(1-beta_K) + beta_K R_v] L_v,  beta_K=(K-1)/(V-1)
  SMALL_FIXED  BML with its width scaled so that the parameter count matches
               AdaptiveFusion on that dataset (capacity-matched BML)
  SMALL_CTRL   capacity-matched BML trained with the controller

Every trained model is evaluated with three test-time fusion rules
  fixed (beta=1, original BML), ctrl (beta_K), uniform (beta=0: router ignored).

Method names in the CSV:  "<model>[train=<rule>,test=<rule>]",
e.g. "BML-full[train=fixed,test=fixed]" is the original BML-unified.
"""
import argparse, copy, json, os, time, warnings
import numpy as np, pandas as pd, torch
import torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.exceptions import ConvergenceWarning

from .common import (load_dataset, split_indices, standardize_train_only,
                     missing_mask, SEEDS, seed_all)
from .adaptivefusion import AdaptiveFusionNet
from .bml_matched import _inject
from .revision_experiments import metrics, joint_condition, JOINT
from .unified import (ALL_CONDS, apply_condition, cond_seed, train_mask, svm_fit, svm_acc,
                      N_FOLDS)

warnings.filterwarnings("ignore", category=ConvergenceWarning)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)

GRID = [50, 100, 200, 400]
QUICK_GRID = [1, 2]
MODELS = {"FULL_FIXED": ("full", "fixed"), "FULL_CTRL": ("full", "ctrl"),
          "SMALL_FIXED": ("small", "fixed"), "SMALL_CTRL": ("small", "ctrl")}
TEST_RULES = ["fixed", "ctrl", "uniform"]


# ================================================================== network
class BMLNet(nn.Module):
    """Structurally identical to bml_matched.BML when width=1024, feat=128, route=256
    (same modules in the same order, so the same seed gives the same initialization)."""
    def __init__(self, dims, C, width=1024, feat=128, route=256, drop=.5):
        super().__init__(); self.C = C
        self.enc = nn.ModuleList([nn.Sequential(
            nn.Linear(d, width), nn.BatchNorm1d(width), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(width, width), nn.BatchNorm1d(width), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(width, width), nn.BatchNorm1d(width), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(width, feat)) for d in dims])
        self.cls = nn.ModuleList([nn.Linear(feat, C) for _ in dims])
        self.route = nn.ModuleList([nn.Sequential(nn.Linear(feat + 2, route), nn.ReLU(),
                                                  nn.Linear(route, 1), nn.Sigmoid()) for _ in dims])


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def small_config(dims, C, target):
    """Width (feat=min(128,w), route=min(256,w)) whose parameter count is closest to target."""
    best = None
    for w in range(4, 1025):
        n = n_params(BMLNet(dims, C, w, min(128, w), min(256, w)))
        if best is None or abs(n - target) < abs(best[1] - target):
            best = (w, n)
        if n > 2 * target:
            break
    w = best[0]
    return dict(width=w, feat=min(128, w), route=min(256, w)), best[1]


def beta_of(rule, m):
    K = m.sum(1, keepdim=True).clamp_min(1); V = m.shape[1]
    if rule == "fixed": return torch.ones_like(K)
    if rule == "ctrl": return (K - 1) / max(V - 1, 1)
    if rule == "uniform": return torch.zeros_like(K)
    raise ValueError(rule)


def forward(model, xs, m, rule):
    """BML forward pass with an availability mask (as unified.bml_forward) and a
    selectable fusion rule. rule='fixed' reproduces unified.bml_forward exactly."""
    Fv = [e(x) for e, x in zip(model.enc, xs)]
    L = torch.stack([c(f) for c, f in zip(model.cls, Fv)], 1)
    P = torch.softmax(L, -1).clamp_min(1e-8); logP = P.log()
    ent = (-(P * logP).sum(-1) / np.log(model.C)).clamp(0, 1)
    D = ((P.unsqueeze(2) - P.unsqueeze(1)) * (logP.unsqueeze(2) - logP.unsqueeze(1))).sum(-1)
    V = L.shape[1]; mj = m.unsqueeze(1) * (1 - torch.eye(V, device=L.device))
    agr = (D * mj).sum(-1) / mj.sum(-1).clamp_min(1)
    R = torch.cat([r(torch.cat([f, -torch.log((1 - ent[:, v:v + 1]).clamp_min(1e-8)), agr[:, v:v + 1]], 1))
                   for v, (r, f) in enumerate(zip(model.route, Fv))], 1)
    b = beta_of(rule, m)
    w = m * ((1 - b) + b * R)
    return (w.unsqueeze(-1) * L).sum(1), R


# ================================================================== training
def fit(X, y, C, seed, checkpoints, dev, cfg, train_rule):
    """BML training loop of unified.fit_bml; only the fusion rule and width differ.
    Cosine schedule tied to the total epochs -> one run per grid value."""
    V = len(X); N = len(y); bs = min(2048, N)
    for epochs in sorted(checkpoints):
        seed_all(seed)
        model = BMLNet([x.shape[1] for x in X], C, **cfg).to(dev)
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
                fused, r = forward(model, xs, m, train_rule)
                bce = (F.binary_cross_entropy(r, torch.tensor(ind[q], device=dev), reduction="none") * m).sum() / m.sum()
                loss = F.cross_entropy(fused, torch.tensor(y[q], dtype=torch.long, device=dev)) + bce
                opt.zero_grad(); loss.backward(); opt.step()
            sch.step()
        model.eval()
        yield epochs, model


@torch.no_grad()
def logits(model, Xc, m, rule, dev):
    return forward(model, [torch.tensor(x, device=dev) for x in Xc],
                   torch.tensor(m, dtype=torch.float32, device=dev), rule)[0].cpu().numpy()


def select_epochs(Xtr, ytr, C, seed, grid, dev, cfg, train_rule, log, tag):
    """3-fold inner CV as in unified.select_epochs; score = mean SVM validation accuracy
    over Clean and Missing50, using the model's own training fusion rule."""
    V = len(Xtr); scores = {e: [] for e in grid}
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed)
    for f, (a, b) in enumerate(skf.split(np.zeros(len(ytr)), ytr)):
        Xa = [x[a] for x in Xtr]; Xb = [x[b] for x in Xtr]
        mb = missing_mask(len(b), V, .5, seed * 131 + f).astype(np.float32)
        Xb50 = [(x * mb[:, v:v + 1]).astype(np.float32) for v, x in enumerate(Xb)]
        oa = np.ones((len(a), V), np.float32); ob = np.ones((len(b), V), np.float32)
        for e, model in fit(Xa, ytr[a], C, seed + 1000 * (f + 1), grid, dev, cfg, train_rule):
            za = logits(model, Xa, oa, train_rule, dev)
            scores[e].append((svm_acc(za, ytr[a], logits(model, Xb, ob, train_rule, dev), ytr[b]) +
                              svm_acc(za, ytr[a], logits(model, Xb50, mb, train_rule, dev), ytr[b])) / 2)
    mean = {e: float(np.mean(v)) for e, v in scores.items()}
    best = max(sorted(mean), key=lambda e: mean[e])
    for e in grid:
        log.append(dict(Group=tag, Seed=seed, Epochs=e, ValScore=mean[e], Selected=e == best))
    return best


# ================================================================== evaluation
def evaluate(model, tag, ds, seed, Xtr, ytr, yte, T, TB, dev, extra):
    size, train_rule = MODELS[tag]
    V = len(Xtr); ones = np.ones((len(ytr), V), np.float32); rows = []
    for rule in TEST_RULES:
        name = f"BML-{size}[train={train_rule},test={rule}]"
        info = dict(Dataset=ds, Method=name, Model=f"BML-{size}", TrainRule=train_rule,
                    TestRule=rule, Seed=seed, **extra)
        scA, clfA = svm_fit(logits(model, Xtr, ones, rule, dev), ytr)
        for cond, (Xc, m, frac) in T.items():
            z = logits(model, Xc, m, rule, dev)
            rows.append(dict(info, Condition=cond, Protocol="A", Head="svm", CorruptedFraction=frac,
                             **metrics(yte, clfA.predict(scA.transform(z)))))
            rows.append(dict(info, Condition=cond, Protocol="A", Head="own", CorruptedFraction=frac,
                             **metrics(yte, z.argmax(1))))
            XtrB, mtrB = TB[cond]
            scB, clfB = svm_fit(logits(model, XtrB, mtrB.astype(np.float32), rule, dev), ytr)
            rows.append(dict(info, Condition=cond, Protocol="B", Head="svm", CorruptedFraction=frac,
                             **metrics(yte, clfB.predict(scB.transform(z)))))
    return rows


# ================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out-dir", default="./results/v5")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--grid", default=None, help="comma-separated epoch grid (overrides GRID)")
    a = ap.parse_args()
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    grid = QUICK_GRID if a.quick else GRID
    if a.grid and not a.quick:
        grid = [int(x) for x in a.grid.split(",") if x]
    tags = [t for t in a.models.split(",") if t]
    os.makedirs(a.out_dir, exist_ok=True)
    P = {k: os.path.join(a.out_dir, f"{a.dataset}{s}") for k, s in
         [("rows", ".csv"), ("sel", "_epoch_selection.csv"), ("done", "_done.json"), ("cfg", "_config.json")]}
    rows = pd.read_csv(P["rows"]).to_dict("records") if os.path.exists(P["rows"]) else []
    sel = pd.read_csv(P["sel"]).to_dict("records") if os.path.exists(P["sel"]) else []
    done = set(json.load(open(P["done"]))) if os.path.exists(P["done"]) else set()

    X, y = load_dataset(a.dataset, a.data_dir); C = len(np.unique(y)); dims = [x.shape[1] for x in X]
    af_params = n_params(AdaptiveFusionNet(dims, C))
    small, small_params = small_config(dims, C, af_params)
    full = dict(width=1024, feat=128, route=256)
    cfgs = {"full": full, "small": small}
    json.dump(dict(dataset=a.dataset, AdaptiveFusion_params=af_params,
                   BML_full_params=n_params(BMLNet(dims, C, **full)),
                   BML_small_params=small_params, BML_small_config=small),
              open(P["cfg"], "w"), indent=1)
    print(f"{a.dataset}: AdaptiveFusion {af_params} params; BML-small width={small['width']} "
          f"({small_params} params); BML-full {n_params(BMLNet(dims, C, **full))} params", flush=True)

    def save():
        pd.DataFrame(rows).to_csv(P["rows"], index=False)
        if sel: pd.DataFrame(sel).to_csv(P["sel"], index=False)
        json.dump(sorted(done), open(P["done"], "w"), indent=1)

    for seed in [int(s) for s in a.seeds.split(",")]:
        todo = [t for t in tags if f"{t}:{seed}" not in done]
        if not todo: continue
        tr, te = split_indices(y, seed, .30)
        Xtr, Xte = standardize_train_only(X, tr, te); ytr, yte = y[tr], y[te]
        # identical test / protocol-B training sets to unified.py
        T = {}
        for c in ALL_CONDS:
            Xc, m = apply_condition(Xte, c, cond_seed(c, seed, (10007, 97)))
            frac = joint_condition(Xte, c, cond_seed(c, seed, (10007, 97)))[2] if c in JOINT else None
            T[c] = (Xc, m.astype(np.float32), frac)
        TB = {c: apply_condition(Xtr, c, cond_seed(c, seed, (20011, 193))) for c in ALL_CONDS}
        for tag in todo:
            t0 = time.time(); size, train_rule = MODELS[tag]; cfg = cfgs[size]
            ep = select_epochs(Xtr, ytr, C, seed, grid, dev, cfg, train_rule, sel, tag)
            _, model = next(fit(Xtr, ytr, C, seed, [ep], dev, cfg, train_rule))
            rows += evaluate(model, tag, a.dataset, seed, Xtr, ytr, yte, T, TB, dev,
                             dict(SelectedEpochs=ep, Params=n_params(model)))
            done.add(f"{tag}:{seed}"); save()
            print(f"{a.dataset} seed={seed} {tag} epochs={ep} done ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
