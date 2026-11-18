
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score,f1_score,balanced_accuracy_score

def fit_gcca(Xtr,latent=32,pca_max=64,reg=1e-2):
    Z=[]; preprocess=[]
    for x in Xtr:
        sc=StandardScaler().fit(x); z=sc.transform(x)
        q=min(pca_max,z.shape[1],z.shape[0]-1)
        p=PCA(q,random_state=0).fit(z); z=p.transform(z)
        Z.append(z); preprocess.append((sc,p))
    n=len(Z[0]); S=np.zeros((n,n))
    for z in Z:
        C=z.T@z+reg*np.eye(z.shape[1])
        P=z@np.linalg.solve(C,z.T); S+=P
    vals,vecs=np.linalg.eigh(S); G=vecs[:,-min(latent,n-1):]
    A=[]
    for z in Z:
        A.append(np.linalg.solve(z.T@z+reg*np.eye(z.shape[1]),z.T@G))
    return preprocess,A

def transform_gcca(X,mask,pre,A):
    zs=[]
    for x,(sc,p),a in zip(X,pre,A): zs.append(p.transform(sc.transform(x))@a)
    Z=np.stack(zs,1); M=mask[...,None]
    return (Z*M).sum(1)/np.maximum(M.sum(1),1)

def evaluate_gcca(Xtr,ytr,Xte,yte,mtr,mte):
    pre,A=fit_gcca(Xtr)
    ztr=transform_gcca(Xtr,mtr,pre,A); zte=transform_gcca(Xte,mte,pre,A)
    sc=StandardScaler().fit(ztr); ztr=sc.transform(ztr); zte=sc.transform(zte)
    clf=LinearSVC(C=1,dual="auto",max_iter=10000).fit(ztr,ytr); p=clf.predict(zte)
    return dict(Accuracy=100*accuracy_score(yte,p),MacroF1=100*f1_score(yte,p,average="macro"),
                BalancedAccuracy=100*balanced_accuracy_score(yte,p))
