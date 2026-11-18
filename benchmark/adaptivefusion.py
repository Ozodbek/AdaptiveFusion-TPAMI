
import numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from .common import seed_all, condition_data

class ViewBlock(nn.Module):
    def __init__(self,d,hidden=64,shared=24,private=24,proj=32):
        super().__init__()
        self.enc=nn.Sequential(nn.Linear(d,hidden),nn.ReLU(),nn.Linear(hidden,shared+private))
        self.proj=nn.Linear(d,proj,bias=False)
        self.ds=nn.Linear(shared,proj,bias=False)
        self.dp=nn.Linear(private,proj,bias=False)
        self.shared=shared
    def forward(self,x):
        h=self.enc(x); s=h[:,:self.shared]; p=h[:,self.shared:]
        H=self.proj(x); R=self.ds(s)+self.dp(p)
        return s,p,H,R

class AdaptiveFusionNet(nn.Module):
    def __init__(self,dims,nclass,hidden=64,shared=24,private=24,proj=32):
        super().__init__()
        self.blocks=nn.ModuleList([ViewBlock(d,hidden,shared,private,proj) for d in dims])
        self.head=nn.Linear(shared+private,nclass)
    def encode_views(self,xs):
        return [b(x) for b,x in zip(self.blocks,xs)]

def _robust_rel(err,med,mad):
    z=(err-med)/(1.4826*mad+1e-6)
    return 1/(1+F.relu(z).pow(2))

def _fuse(S,P,mask,rel,beta_mode="adaptive"):
    B,V,D=S.shape
    K=mask.sum(1,keepdim=True).clamp_min(1)
    if beta_mode=="adaptive":
        beta=(K-1)/max(V-1,1)
    elif beta_mode=="fixed":
        beta=torch.ones_like(K)
    else: beta=torch.zeros_like(K)
    w=mask*((1-beta)+beta*rel)
    sf=(w.unsqueeze(-1)*S).sum(1)/(w.sum(1,keepdim=True)+1e-8)
    pf=(mask.unsqueeze(-1)*P).sum(1)/(K+1e-8)
    return torch.cat([sf,pf],1)

def _orth(s,p): return (F.normalize(s,dim=1)*F.normalize(p,dim=1)).sum(1).pow(2).mean()

def _align(S,mask):
    V=S.shape[1]; loss=0.; cnt=0
    for i in range(V):
        for j in range(i+1,V):
            ok=(mask[:,i]*mask[:,j])>0
            if ok.any():
                loss += (1-F.cosine_similarity(S[ok,i],S[ok,j]).mean()); cnt+=1
    return loss/max(cnt,1)

def _proto_loss(z,y):
    loss=0.; cnt=0
    for c in y.unique():
        q=z[y==c]
        if len(q)>1:
            ctr=q.mean(0,keepdim=True)
            loss += (1-F.cosine_similarity(q,ctr.expand_as(q))).mean(); cnt+=1
    return loss/max(cnt,1)

def train_model(Xtr,ytr,nclass,seed,epochs=18,device=None):
    seed_all(seed)
    device=device or ("cuda" if torch.cuda.is_available() else "cpu")
    model=AdaptiveFusionNet([x.shape[1] for x in Xtr],nclass).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-4)
    TX=[torch.tensor(x,device=device) for x in Xtr]; TY=torch.tensor(ytr,dtype=torch.long,device=device)
    N=len(ytr); V=len(Xtr); bs=min(256,N)

    # Training-only residual calibration from clean initial/current representation is updated each epoch.
    med=torch.ones(V,device=device); mad=torch.ones(V,device=device)
    rates=[0.,.3,.5,.7]
    for ep in range(epochs):
        perm=torch.randperm(N,device=device)
        for st in range(0,N,bs):
            ids=perm[st:st+bs]
            if len(ids)<2: continue
            xs=[x[ids] for x in TX]; yy=TY[ids]
            r=float(rates[(ep+st//bs)%len(rates)])
            rng=np.random.default_rng(seed*100000+ep*1000+st)
            m=np.ones((len(ids),V),np.float32) if r==0 else (rng.random((len(ids),V))>=r).astype(np.float32)
            zero=np.where(m.sum(1)==0)[0]
            if len(zero): m[zero,rng.integers(0,V,len(zero))]=1
            mt=torch.tensor(m,device=device)
            xmask=[x*mt[:,v:v+1] for v,x in enumerate(xs)]

            outs=model.encode_views(xmask)
            S=torch.stack([o[0] for o in outs],1); P=torch.stack([o[1] for o in outs],1)
            E=torch.stack([torch.sqrt(((o[2]-o[3])**2).mean(1)+1e-8) for o in outs],1)
            with torch.no_grad():
                # robust batch calibration; final evaluation recalibrates using all training samples only
                medb=E.median(0).values
                madb=(E-medb).abs().median(0).values.clamp_min(1e-4)
            rel=_robust_rel(E,medb,madb)*mt
            z=_fuse(S,P,mt,rel,"adaptive")
            logits=model.head(z)

            # clean branch
            oc=model.encode_views(xs)
            Sc=torch.stack([o[0] for o in oc],1); Pc=torch.stack([o[1] for o in oc],1)
            Ec=torch.stack([torch.sqrt(((o[2]-o[3])**2).mean(1)+1e-8) for o in oc],1)
            rc=_robust_rel(Ec,Ec.median(0).values,(Ec-Ec.median(0).values).abs().median(0).values.clamp_min(1e-4))
            zc=_fuse(Sc,Pc,torch.ones_like(mt),rc,"adaptive")
            lc=model.head(zc)

            lcls=F.cross_entropy(logits,yy)+F.cross_entropy(lc,yy)
            lrec=sum(F.mse_loss(o[3],o[2]) for o in outs)/V
            lorth=sum(_orth(o[0],o[1]) for o in outs)/V
            lalign=_align(S,mt)
            lproto=_proto_loss(z,yy)
            linv=(1-F.cosine_similarity(z,zc).mean())
            loss=lcls+.22*lrec+.06*lorth+.08*lalign+.12*lproto+.20*linv
            opt.zero_grad(); loss.backward(); opt.step()
    return model

@torch.no_grad()
def representations(model,Xtr,Xte,mtr,mte,beta_mode="adaptive",device=None):
    device=device or next(model.parameters()).device
    def enc(X):
        xs=[torch.tensor(x,device=device) for x in X]
        o=model.encode_views(xs)
        S=torch.stack([q[0] for q in o],1); P=torch.stack([q[1] for q in o],1)
        E=torch.stack([torch.sqrt(((q[2]-q[3])**2).mean(1)+1e-8) for q in o],1)
        return S,P,E
    Str,Ptr,Etr=enc(Xtr)
    med=Etr.median(0).values; mad=(Etr-med).abs().median(0).values.clamp_min(1e-4)
    Ste,Pte,Ete=enc(Xte)
    Mtr=torch.tensor(mtr,dtype=torch.float32,device=device); Mte=torch.tensor(mte,dtype=torch.float32,device=device)
    Rtr=_robust_rel(Etr,med,mad)*Mtr; Rte=_robust_rel(Ete,med,mad)*Mte
    ztr=_fuse(Str,Ptr,Mtr,Rtr,beta_mode).cpu().numpy()
    zte=_fuse(Ste,Pte,Mte,Rte,beta_mode).cpu().numpy()
    return ztr,zte

def evaluate(model,Xtr,ytr,Xte,yte,mtr,mte,beta_mode="adaptive"):
    ztr,zte=representations(model,Xtr,Xte,mtr,mte,beta_mode)
    sc=StandardScaler().fit(ztr); ztr=sc.transform(ztr); zte=sc.transform(zte)
    clf=LinearSVC(C=1,dual="auto",max_iter=10000).fit(ztr,ytr)
    pred=clf.predict(zte)
    return dict(Accuracy=100*accuracy_score(yte,pred),
                MacroF1=100*f1_score(yte,pred,average="macro"),
                BalancedAccuracy=100*balanced_accuracy_score(yte,pred))
