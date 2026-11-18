
import time, random, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score,f1_score,balanced_accuracy_score
torch.set_num_threads(max(1,min(4,torch.get_num_threads())))

class AE(nn.Module):
    def __init__(self,d,latent=32,h=128):
        super().__init__()
        self.enc=nn.Sequential(nn.Linear(d,h),nn.BatchNorm1d(h),nn.ReLU(),
            nn.Linear(h,h),nn.BatchNorm1d(h),nn.ReLU(),
            nn.Linear(h,h),nn.BatchNorm1d(h),nn.ReLU(),nn.Linear(h,latent))
        self.dec=nn.Sequential(nn.Linear(latent,h),nn.BatchNorm1d(h),nn.ReLU(),
            nn.Linear(h,h),nn.BatchNorm1d(h),nn.ReLU(),
            nn.Linear(h,h),nn.BatchNorm1d(h),nn.ReLU(),nn.Linear(h,d))
    def forward(self,x):
        z=self.enc(x); return self.dec(z),z

class Model(nn.Module):
    def __init__(self,dims,nclass,latent=32):
        super().__init__(); self.aes=nn.ModuleList([AE(d,latent) for d in dims])
        self.clf=nn.Sequential(nn.Linear(latent*len(dims),max(16,latent//2)),nn.BatchNorm1d(max(16,latent//2)),
                               nn.ReLU(),nn.Linear(max(16,latent//2),nclass))
    def forward(self,xs):
        R=[]; Z=[]
        for ae,x in zip(self.aes,xs):
            r,z=ae(x); R.append(r); Z.append(z)
        return R,Z,self.clf(torch.cat(Z,1))

def _icl(z1,z2,alpha=10.,eps=1e-12):
    a=F.softmax(z1,1); b=F.softmax(z2,1)
    P=torch.einsum("bi,bj->ij",a,b); P=(P+P.T)/2; P=P/P.sum().clamp_min(eps)
    Pi=P.sum(1,keepdim=True).clamp_min(eps); Pj=P.sum(0,keepdim=True).clamp_min(eps); P=P.clamp_min(eps)
    return (-P*(P.log()-alpha*Pi.log()-alpha*Pj.log())).mean()

def _ccl(z,y,C):
    q=F.softmax(z,1); S=q@q.T; S=S-torch.diag_embed(torch.diag(S))
    oh=F.one_hot(y,C).float(); cnt=oh.sum(0,keepdim=True).repeat(len(y),1)-oh; cnt[cnt==0]=1
    Sm=(S@oh)/cnt; pr=Sm.argmax(1); return F.relu((y==pr).float()+Sm.max(1).values-(Sm*oh).sum(1)).mean()

def _knn_complete(Xtr,Xte,mtr,mte,k=10):
    outtr=[x.copy() for x in Xtr]; outte=[x.copy() for x in Xte]; V=len(Xtr)
    norms=[x/(np.linalg.norm(x,axis=1,keepdims=True)+1e-8) for x in Xtr]
    normte=[x/(np.linalg.norm(x,axis=1,keepdims=True)+1e-8) for x in Xte]
    for v in range(V):
        donors=np.where(mtr[:,v]>0)[0]
        mean=Xtr[v][donors].mean(0) if len(donors) else Xtr[v].mean(0)
        for split,out,Xn,M,otherN in [(0,outtr,norms,mtr,norms),(1,outte,normte,mte,norms)]:
            ids=np.where(M[:,v]==0)[0]
            for i in ids:
                oth=[u for u in range(V) if u!=v and M[i,u]>0]
                if not oth or not len(donors): out[v][i]=mean; continue
                dist=np.zeros(len(donors)); cnt=np.zeros(len(donors))
                for u in oth:
                    ok=mtr[donors,u]>0
                    base = norms[u][donors]
                    target = Xn[u][i]
                    dd=((base-target)**2).mean(1)
                    dist += np.where(ok,dd,0); cnt += ok
                ok=cnt>0
                if not ok.any(): out[v][i]=mean; continue
                score=np.where(ok,dist/np.maximum(cnt,1),np.inf)
                kk=min(k,int(ok.sum())); loc=np.argpartition(score,kk-1)[:kk]
                out[v][i]=Xtr[v][donors[loc]].mean(0)
    return outtr,outte

def run(Xtr,ytr,Xte,yte,mtr,mte,seed,epochs=200,device=None):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device=device or ("cuda" if torch.cuda.is_available() else "cpu")
    Xtrc,Xtec=_knn_complete(Xtr,Xte,mtr,mte)
    C=len(np.unique(ytr)); model=Model([x.shape[1] for x in Xtr],C).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=1e-4); TX=[torch.tensor(x,device=device) for x in Xtrc]
    TY=torch.tensor(ytr,dtype=torch.long,device=device); bs=min(256,len(ytr)); V=len(Xtr)
    for ep in range(epochs):
        g=torch.Generator(device=device); g.manual_seed(seed*100000+ep)
        perm=torch.randperm(len(TY),generator=g,device=device)
        model.train()
        for st in range(0,len(TY),bs):
            ids=perm[st:st+bs]
            if len(ids)<2: continue
            xs=[x[ids] for x in TX]; yy=TY[ids]
            R,Z,L=model(xs)
            loss=sum(F.mse_loss(r,x) for r,x in zip(R,xs))/V
            loss += sum(_icl(Z[i],Z[j]) for i in range(V) for j in range(i+1,V))
            loss += _ccl(torch.cat(Z,1),yy,C)+F.cross_entropy(L,yy)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        pred=model([torch.tensor(x,device=device) for x in Xtec])[2].argmax(1).cpu().numpy()
    return dict(Accuracy=100*accuracy_score(yte,pred),MacroF1=100*f1_score(yte,pred,average="macro"),
                BalancedAccuracy=100*balanced_accuracy_score(yte,pred))
