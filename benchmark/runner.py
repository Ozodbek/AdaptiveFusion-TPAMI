
import argparse, os, json, time
import numpy as np, pandas as pd
from .common import *
from .adaptivefusion import train_model,evaluate
from .gcca import evaluate_gcca
from .dimvln_matched import run as run_dimvln
from .bml_matched import run as run_bml

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-dir",default="./data")
    ap.add_argument("--dataset",default="MFeat")
    ap.add_argument("--methods",default="AdaptiveFusion,SPRR,GCCA")
    ap.add_argument("--seeds",default=",".join(map(str,SEEDS)))
    ap.add_argument("--conditions",default=",".join(CONDITIONS))
    ap.add_argument("--epochs-af",type=int,default=18)
    ap.add_argument("--epochs-dimvln",type=int,default=200)
    ap.add_argument("--epochs-bml",type=int,default=200)
    ap.add_argument("--out",default="./results/results.csv")
    ap.add_argument("--resume",action="store_true",help="Skip completed Dataset/Method/Seed/Condition rows already in --out")
    args=ap.parse_args()
    seeds=[int(x) for x in args.seeds.split(",") if x]
    conds=[x for x in args.conditions.split(",") if x]
    methods=[x.strip() for x in args.methods.split(",") if x.strip()]
    X,y=load_dataset(args.dataset,args.data_dir); C=len(np.unique(y))
    rows=[]
    os.makedirs(os.path.dirname(args.out) or ".",exist_ok=True)
    done=set()
    if args.resume and os.path.exists(args.out):
        old=pd.read_csv(args.out)
        rows=old.to_dict("records")
        done=set((str(r["Dataset"]),str(r["Method"]),int(r["Seed"]),str(r["Condition"])) for _,r in old.iterrows())

    cond_code={c:i+1 for i,c in enumerate(CONDITIONS)}
    def is_done(method,seed,cond):
        return (args.dataset,method,int(seed),cond) in done
    def addrow(method,seed,cond,r):
        key=(args.dataset,method,int(seed),cond)
        if key not in done:
            rows.append(dict(Dataset=args.dataset,Method=method,Seed=seed,Condition=cond,**r))
            done.add(key)
            pd.DataFrame(rows).to_csv(args.out,index=False)

    for seed in seeds:
        tr,te=split_indices(y,seed,.30)
        Xtr,Xte=standardize_train_only(X,tr,te); ytr,yte=y[tr],y[te]

        af=None
        if "AdaptiveFusion" in methods or "SPRR" in methods:
            af=train_model(Xtr,ytr,C,seed,args.epochs_af)

        if "BML" in methods:
            bconds=[c for c in conds if c=="Clean" or c.startswith("Conflict")]
            todo=[c for c in bconds if not is_done("BML-matched",seed,c)]
            if todo:
                bres=run_bml(Xtr,ytr,Xte,yte,seed,test_conditions=todo,epochs=args.epochs_bml)
                for c,r in bres.items(): addrow("BML-matched",seed,c,r)

        for cond in conds:
            Xte_c,mte=condition_data(Xte,cond,seed*10007+cond_code.get(cond,99)*97)
            # Training representations use clean/full availability for inference classifier.
            mtr=np.ones((len(ytr),len(Xtr)),np.int64)
            if "AdaptiveFusion" in methods:
                r=evaluate(af,Xtr,ytr,Xte_c,yte,mtr,mte,"adaptive")
                addrow("AdaptiveFusion",seed,cond,r)
            if "SPRR" in methods:
                r=evaluate(af,Xtr,ytr,Xte_c,yte,mtr,mte,"fixed")
                addrow("SP-RR",seed,cond,r)
            if "GCCA" in methods:
                r=evaluate_gcca(Xtr,ytr,Xte_c,yte,mtr,mte)
                addrow("MAXVAR-GCCA",seed,cond,r)
            if "DIMvLN" in methods and (cond=="Clean" or cond.startswith("Missing")):
                # Matched DIMvLN trains under the same missingness condition.
                Xtr_c,mtr_c=condition_data(Xtr,cond,seed*20011+cond_code.get(cond,99)*193)
                r=run_dimvln(Xtr_c,ytr,Xte_c,yte,mtr_c,mte,seed,args.epochs_dimvln)
                addrow("DIMvLN-matched",seed,cond,r)
            pd.DataFrame(rows).to_csv(args.out,index=False)
            print(seed,cond,"done",flush=True)

if __name__=="__main__": main()
