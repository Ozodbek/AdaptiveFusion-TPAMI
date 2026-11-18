
import argparse, pandas as pd, numpy as np
from scipy.stats import ttest_rel, wilcoxon

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("csv"); ap.add_argument("--reference",default="AdaptiveFusion")
    a=ap.parse_args(); d=pd.read_csv(a.csv)
    s=d.groupby(["Dataset","Method","Condition"]).agg(
        Accuracy_mean=("Accuracy","mean"),Accuracy_std=("Accuracy","std"),
        MacroF1_mean=("MacroF1","mean"),MacroF1_std=("MacroF1","std"),
        n=("Seed","count")).reset_index()
    s.to_csv(a.csv.replace(".csv","_summary.csv"),index=False)
    out=[]
    for (ds,cond),g in d.groupby(["Dataset","Condition"]):
        ref=g[g.Method==a.reference]
        if ref.empty: continue
        for meth in sorted(g.Method.unique()):
            if meth==a.reference: continue
            q=ref[["Seed","Accuracy"]].merge(g[g.Method==meth][["Seed","Accuracy"]],on="Seed",suffixes=("_ref","_m"))
            if len(q)<2: continue
            diff=q.Accuracy_ref-q.Accuracy_m
            t,p=ttest_rel(q.Accuracy_ref,q.Accuracy_m)
            try: W,wp=wilcoxon(diff)
            except: W,wp=np.nan,np.nan
            dz=diff.mean()/diff.std(ddof=1) if diff.std(ddof=1)>0 else np.nan
            out.append([ds,cond,meth,len(q),diff.mean(),t,p,W,wp,dz])
    st=pd.DataFrame(out,columns=["Dataset","Condition","Competitor","n","MeanDiff_pp","paired_t","t_p","Wilcoxon_W","Wilcoxon_p","Cohen_dz"])
    st.to_csv(a.csv.replace(".csv","_paired_stats.csv"),index=False)
    print(s.to_string(index=False)); print("\nPaired stats:\n",st.to_string(index=False))
if __name__=="__main__": main()
