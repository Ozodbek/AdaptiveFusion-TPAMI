
import random, numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import accuracy_score,f1_score,balanced_accuracy_score

class BML(nn.Module):
    def __init__(self,dims,C,feat=128,drop=.5):
        super().__init__(); self.C=C
        self.enc=nn.ModuleList([nn.Sequential(
            nn.Linear(d,1024),nn.BatchNorm1d(1024),nn.ReLU(),nn.Dropout(drop),
            nn.Linear(1024,1024),nn.BatchNorm1d(1024),nn.ReLU(),nn.Dropout(drop),
            nn.Linear(1024,1024),nn.BatchNorm1d(1024),nn.ReLU(),nn.Dropout(drop),
            nn.Linear(1024,feat)) for d in dims])
        self.cls=nn.ModuleList([nn.Linear(feat,C) for _ in dims])
        self.route=nn.ModuleList([nn.Sequential(nn.Linear(feat+2,256),nn.ReLU(),nn.Linear(256,1),nn.Sigmoid()) for _ in dims])
    def forward(self,xs):
        Fv=[e(x) for e,x in zip(self.enc,xs)]; L=[c(f) for c,f in zip(self.cls,Fv)]
        P=[torch.softmax(l,1).clamp_min(1e-8) for l in L]
        ent=[(-(p*p.log()).sum(1,keepdim=True)/np.log(self.C)).clamp(0,1) for p in P]
        agr=[]
        for i,p in enumerate(P):
            s=0
            for j,q in enumerate(P):
                if i==j: continue
                s += (p*(p.log()-q.log())).sum(1,keepdim=True)+(q*(q.log()-p.log())).sum(1,keepdim=True)
            agr.append(s/max(len(P)-1,1))
        R=torch.cat([r(torch.cat([f, -torch.log((1-e).clamp_min(1e-8)),a],1)) for r,f,e,a in zip(self.route,Fv,ent,agr)],1)
        LS=torch.stack(L,1); fused=(R.unsqueeze(-1)*LS).sum(1)
        return fused,R

def _inject(xs,y,ratio,rng,one_view=False):
    out=[x.copy() for x in xs]; n=len(y); V=len(xs)
    clean=np.ones((n,V),np.float32); ids=rng.choice(n,size=int(round(ratio*n)),replace=False)
    for i in ids:
        if one_view: cor=[int(rng.integers(V))]
        else:
            keep=rng.choice(V,size=int(np.ceil(V/2)),replace=False); cor=[v for v in range(V) if v not in keep]
        for v in cor:
            j=int(rng.integers(n-1)); j += (j>=i)
            out[v][i]=xs[v][j]; clean[i,v]=float(y[j]==y[i])
    return out,clean

def run(Xtr,ytr,Xte,yte,seed,test_conditions=("Clean","Conflict30","Conflict50"),epochs=200,device=None,matched_one_view=True):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device=device or ("cuda" if torch.cuda.is_available() else "cpu")
    C=len(np.unique(ytr)); model=BML([x.shape[1] for x in Xtr],C).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=.002); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs,eta_min=1e-8)
    bs=min(2048,len(ytr))
    for ep in range(epochs):
        rng=np.random.default_rng(ep)
        xx,ind=_inject(Xtr,ytr,.5,rng,one_view=matched_one_view)
        ids=np.arange(len(ytr)); rng.shuffle(ids); model.train()
        for st in range(0,len(ids),bs):
            q=ids[st:st+bs]
            if len(q)<2: continue
            xs=[torch.tensor(x[q],device=device) for x in xx]; yy=torch.tensor(ytr[q],dtype=torch.long,device=device)
            ci=torch.tensor(ind[q],device=device); fused,r=model(xs)
            loss=F.cross_entropy(fused,yy)+F.binary_cross_entropy(r,ci)
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
    ans={}
    model.eval()
    for cond in test_conditions:
        ratio=0 if cond=="Clean" else float(cond.replace("Conflict",""))/100
        xx,_=_inject(Xte,yte,ratio,np.random.default_rng(seed*101+int(ratio*1000)),one_view=matched_one_view)
        with torch.no_grad(): pred=model([torch.tensor(x,device=device) for x in xx])[0].argmax(1).cpu().numpy()
        ans[cond]=dict(Accuracy=100*accuracy_score(yte,pred),MacroF1=100*f1_score(yte,pred,average="macro"),
                       BalancedAccuracy=100*balanced_accuracy_score(yte,pred))
    return ans
