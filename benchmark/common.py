
import os, random, zipfile
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEEDS = [3,7,11,17,23,29,31,37,41,43]
CONDITIONS = ["Clean","Conflict30","Conflict50","Missing30","Missing50","Missing70","Noise30","Noise50"]

def seed_all(seed):
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    except Exception: pass

def _orient(x,n=None):
    x=np.asarray(x)
    if n is not None and x.shape[0]!=n and x.shape[1]==n: x=x.T
    return x.astype(np.float32)

def load_mfeat_zip(path):
    spec=[("mfeat-pix",240),("mfeat-fou",76),("mfeat-fac",216),
          ("mfeat-zer",47),("mfeat-kar",64),("mfeat-mor",6)]
    X=[]
    with zipfile.ZipFile(path) as z:
        for name,d in spec:
            with z.open(name) as f: a=np.loadtxt(f,dtype=np.float32)
            assert a.shape==(2000,d), (name,a.shape)
            X.append(a)
    y=np.repeat(np.arange(10,dtype=np.int64),200)
    return X,y

def load_cell_mat(path, label_keys=("Y","gt","y")):
    m=loadmat(path)
    y=None
    for k in label_keys:
        if k in m: y=np.asarray(m[k]).ravel().astype(np.int64); break
    if y is None: raise KeyError(f"No label key in {path}")
    if y.min()==1: y=y-1
    Xraw=m["X"].ravel()
    X=[_orient(x,len(y)) for x in Xraw]
    return X,y

def load_dataset(name, data_dir):
    data_dir=os.path.abspath(data_dir)
    key=name.lower()
    if key in ("mfeat","handwritten"):
        return load_mfeat_zip(os.path.join(data_dir,"multiple+features.zip"))
    table={
      "scene15":"Scene15.mat",
      "landuse21":"LandUse-21.mat",
      "caltech101-20":"Caltech101-20_balanced_33perclass.mat",
      "bdgp":"BDGP_fea.mat",
      "reuters1200":"Reuters-1200.mat",
    }
    if key not in table: raise ValueError(f"Unknown dataset {name}")
    return load_cell_mat(os.path.join(data_dir,table[key]))

def split_indices(y,seed,test_size=.30):
    idx=np.arange(len(y))
    return train_test_split(idx,test_size=test_size,random_state=seed,stratify=y)

def standardize_train_only(X,tr,te):
    Atr=[]; Ate=[]
    for x in X:
        sc=StandardScaler().fit(x[tr])
        Atr.append(sc.transform(x[tr]).astype(np.float32))
        Ate.append(sc.transform(x[te]).astype(np.float32))
    return Atr,Ate

def minmax_train_only(X,tr,te):
    Atr=[]; Ate=[]
    for x in X:
        lo=x[tr].min(0,keepdims=True); hi=x[tr].max(0,keepdims=True)
        den=hi-lo; den[den<1e-8]=1
        Atr.append(((x[tr]-lo)/den).astype(np.float32))
        Ate.append(((x[te]-lo)/den).astype(np.float32))
    return Atr,Ate

def missing_mask(n,V,p,seed):
    if p<=0: return np.ones((n,V),np.int64)
    rng=np.random.default_rng(seed)
    m=(rng.random((n,V))>=p).astype(np.int64)
    zero=np.where(m.sum(1)==0)[0]
    if len(zero):
        keep=rng.integers(0,V,len(zero)); m[zero,keep]=1
    return m

def condition_data(X,condition,seed):
    """Returns corrupted copies and availability mask.
    ConflictXX: XX% samples, one random view replaced by a different sample.
    MissingXX: independent per-view missing p, force >=1 observed.
    NoiseXX: XX% samples, one random view gets N(0,2^2) additive noise.
    """
    Xc=[a.copy() for a in X]
    n=len(X[0]); V=len(X); rng=np.random.default_rng(seed)
    mask=np.ones((n,V),np.int64)
    if condition=="Clean": return Xc,mask
    if condition.startswith("Missing"):
        p=float(condition.replace("Missing",""))/100
        mask=missing_mask(n,V,p,seed)
        for v in range(V): Xc[v][mask[:,v]==0]=0
        return Xc,mask
    frac=float(''.join(c for c in condition if c.isdigit()))/100
    ids=rng.choice(n,size=int(round(frac*n)),replace=False)
    if condition.startswith("Conflict"):
        for i in ids:
            v=int(rng.integers(V))
            j=int(rng.integers(n-1))
            if j>=i: j+=1
            Xc[v][i]=X[v][j]
    elif condition.startswith("Noise"):
        for i in ids:
            v=int(rng.integers(V))
            Xc[v][i]=Xc[v][i]+rng.normal(0,2.0,size=Xc[v][i].shape).astype(np.float32)
    else: raise ValueError(condition)
    return Xc,mask
