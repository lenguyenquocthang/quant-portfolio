"""Single-index model for 12 Vietnamese bank stocks vs VN30: corrected re-estimation.

Usage:  python run.py --data <path-to-xlsx> [--out results]
The raw workbook is NOT shipped with this repository (see README, "Data").
"""
import argparse, itertools, json, os, sys, warnings, unicodedata as U
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import brentq
from linearmodels.iv import IV2SLS
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from scipy import stats
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from panelkit import *  # noqa  (panel diagnostics + robust inference helpers)
warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--out", default="results")
args = ap.parse_args(); OUT = Path(args.out); OUT.mkdir(exist_ok=True, parents=True); (OUT/"figures").mkdir(exist_ok=True)
import statsmodels.api as sm
from scipy import stats
f = args.data

# ===== m3.py
r=pd.read_excel(f,'TSLN'); p=pd.read_excel(f,'Giá')
r.columns=[str(c).strip() for c in r.columns]; p.columns=[str(c).strip() for c in p.columns]
r=r.set_index('Date'); p=p.set_index('Date')
alt=np.log(p)/np.log(p.shift(1))
d=(alt-r).abs().max(); print('max abs diff TSLN vs lnP_t/lnP_{t-1}:',d.max(), d.to_dict())
banks=['ACB','BID','CTG','EIB','HDB','MBB','STB','TCB','TPB','VCB','VIB','VPB']
# paper SIM on TSLN (gross, 1+R series) 
def sim(y,x):
    X=sm.add_constant(x); m=sm.OLS(y,X).fit(); return m
paper={}; true={}
lr=np.log(p).diff().dropna(); sr=(p/p.shift(1)-1).dropna()
rr=r.dropna()
rows=[]
for b in banks:
    m1=sim(rr[b],rr['VN30'])
    m2=sim(lr[b],lr['VN30'])
    m3=sm.OLS(lr[b],sm.add_constant(lr['VN30'])).fit(cov_type='HAC',cov_kwds={'maxlags':5})
    rows.append([b,m1.params['VN30'],m1.rsquared,m1.params['const'],m2.params['VN30'],m2.rsquared,m2.params['const']*1e4,m3.bse['VN30'],m2.resid.std(), lr[b].std(), 1/np.log(p[b].iloc[0])])
R=pd.DataFrame(rows,columns=['bank','beta_paper','R2_paper','alphaG_paper','beta_true','R2_true','alpha_bps','se_HAC','sd_eps','sd_ret','k_i']); print(R)
print('sigma_M true',lr['VN30'].std(),'paper',rr['VN30'].std())
SIMTAB=R.to_dict('records')
# verify ratio of betas = k_i/k_M
kM=np.log(p['VN30'].iloc[0])
print((R.beta_paper/R.beta_true).round(3).tolist(), ((1/np.log(p[banks].iloc[0]))*kM).round(3).tolist())

# ===== m4.py
p=pd.read_excel(f,'Giá'); p.columns=[str(c).strip() for c in p.columns]; p=p.set_index('Date')
banks=['ACB','BID','CTG','EIB','HDB','MBB','STB','TCB','TPB','VCB','VIB','VPB']
lr=np.log(p).diff().dropna(); M=lr['VN30']; T=len(lr)
out={}
R=pd.DataFrame(SIMTAB); print(R[['bank','beta_paper','beta_true','R2_paper','R2_true','alpha_bps','se_HAC']])
# decomposition true
sM2=M.var(ddof=1); rows=[]
for b in banks:
    m=sm.OLS(lr[b],sm.add_constant(M)).fit(); be=m.params['VN30']; s2=lr[b].var(ddof=1)
    # Dimson (lags -1,+1)
    X=pd.concat([M,M.shift(1),M.shift(-1)],axis=1).dropna(); X.columns=['m','m_l1','m_f1']
    md=sm.OLS(lr[b].loc[X.index],sm.add_constant(X)).fit(cov_type='HAC',cov_kwds={'maxlags':5}); bd=md.params[['m','m_l1','m_f1']].sum()
    se_b=m.get_robustcov_results('HAC',maxlags=5).bse[1]
    rows.append(dict(bank=b,beta=be,se=se_b,lo=be-1.96*se_b,hi=be+1.96*se_b,R2=m.rsquared,alpha_bps=m.params['const']*1e4,alpha_ann=m.params['const']*252,t_alpha=m.params['const']/m.get_robustcov_results('HAC',maxlags=5).bse[0],
        tot=s2*1e6,sys=be**2*sM2*1e6,idio=m.resid.var(ddof=2)*1e6,beta_dimson=bd,beta_blume=2/3*be+1/3))
D=pd.DataFrame(rows); print(D.round(3)); out['sim']=D.round(5).to_dict('records'); out['sigmaM']=float(M.std()); out['sigmaM_paper']=0.002036
# residual correlations & LM test (Breusch-Pagan)
E=pd.DataFrame({b:sm.OLS(lr[b],sm.add_constant(M)).fit().resid for b in banks}); C=E.corr(); iu=np.triu_indices(12,1)
LM=T*np.sum(C.values[iu]**2); df=12*11/2; print('LM',LM,df,stats.chi2.sf(LM,df),'mean resid corr',C.values[iu].mean(),'max',C.values[iu].max())
out['lm']={'LM':float(LM),'df':df,'p':float(stats.chi2.sf(LM,df)),'mean_corr':float(C.values[iu].mean()),'max_corr':float(C.values[iu].max()),'min_corr':float(C.values[iu].min())}
# two-index: add equal-weighted index of OTHER banks orthogonalised to M
res=[]
for b in banks:
    other=lr[[x for x in banks if x!=b]].mean(axis=1)
    o_orth=sm.OLS(other,sm.add_constant(M)).fit().resid
    X=pd.concat([M,o_orth],axis=1); X.columns=['m','sector']
    m2=sm.OLS(lr[b],sm.add_constant(X)).fit(cov_type='HAC',cov_kwds={'maxlags':5})
    res.append((b,m2.params['m'],m2.params['sector'],m2.pvalues['sector'],m2.rsquared))
S=pd.DataFrame(res,columns=['bank','beta_m','beta_sector','p_sector','R2_2idx']); print(S.round(3)); out['two_index']=S.round(5).to_dict('records')
E2=pd.DataFrame({b:sm.OLS(lr[b],sm.add_constant(pd.concat([M,sm.OLS(lr[[x for x in banks if x!=b]].mean(axis=1),sm.add_constant(M)).fit().resid],axis=1))).fit().resid for b in banks})
C2=E2.corr(); LM2=T*np.sum(C2.values[iu]**2); print('LM after sector',LM2,stats.chi2.sf(LM2,df),C2.values[iu].mean()); out['lm2']={'LM':float(LM2),'mean_corr':float(C2.values[iu].mean())}
# subperiod beta stability
for lab,sl in [('2020-2021',slice('2020-01-01','2021-12-31')),('2022',slice('2022-01-01','2022-12-31')),('2023-24',slice('2023-01-01','2024-12-31'))]:
    bs=[sm.OLS(lr[b].loc[sl],sm.add_constant(M.loc[sl])).fit().params['VN30'] for b in banks]; print(lab,np.round(bs,2))
sub={lab:[float(sm.OLS(lr[b].loc[sl],sm.add_constant(M.loc[sl])).fit().params['VN30']) for b in banks] for lab,sl in [('2020-2021',slice('2020-01-01','2021-12-31')),('2022',slice('2022-01-01','2022-12-31')),('2023-24',slice('2023-01-01','2024-12-31'))]}
out['sub']=sub
# JB for returns
out['jb']={b:[float(stats.skew(lr[b])),float(stats.kurtosis(lr[b],fisher=False)),float(stats.jarque_bera(lr[b])[1])] for b in banks+['VN30']}
out['jb']['VN30']=[float(stats.skew(M)),float(stats.kurtosis(M,fisher=False)),float(stats.jarque_bera(M)[1])]
print(out['jb']['VN30'],out['jb']['TCB'])
# ADF
print('ADF VN30 ret',adfuller(M)[1],'ADF price',adfuller(p['VN30'])[1]); out['adf']={'ret':float(adfuller(M)[1]),'price':float(adfuller(p['VN30'])[1])}


# ===== m5.py
p=pd.read_excel(f,'Giá'); p.columns=[str(c).strip() for c in p.columns]; p=p.set_index('Date')
banks=['ACB','BID','CTG','EIB','HDB','MBB','STB','TCB','TPB','VCB','VIB','VPB']
lr=np.log(p).diff().dropna(); M=lr['VN30']
def egp(tr,rf=0.0):
    sM2=tr['VN30'].var(); rows=[]
    for b in banks:
        m=sm.OLS(tr[b],sm.add_constant(tr['VN30'])).fit(); rows.append((b,tr[b].mean()-rf,m.params['VN30'],m.resid.var(ddof=2)))
    D=pd.DataFrame(rows,columns=['b','ex','beta','s2']); D['rank']=D.ex/D.beta; D=D.sort_values('rank',ascending=False).reset_index(drop=True)
    D['num']=D.ex*D.beta/D.s2; D['den']=D.beta**2/D.s2
    D['C']=sM2*D.num.cumsum()/(1+sM2*D.den.cumsum())
    ks=[k for k in range(len(D)) if D['rank'][k]>D['C'][k]]
    if not ks: return None,D
    # contiguous
    k=0
    while k<len(D) and D['rank'][k]>D['C'][k]: k+=1
    Cs=D['C'][k-1]; sel=D.iloc[:k].copy(); sel['z']=sel.beta/sel.s2*(sel['rank']-Cs); sel['w']=sel.z/sel.z.sum(); return sel,D
tr=lr.loc[:'2022-12-31']; te=lr.loc['2023-01-01':]
sel,D=egp(tr); print(D[['b','ex','beta','rank','C']]); print(sel[['b','w']] if sel is not None else 'none')
res={}
if sel is not None:
    w=sel.set_index('b').w
    pe=(te[w.index]*w).sum(axis=1); ew=te[banks].mean(axis=1)
    def sh(x): return x.mean()/x.std()*np.sqrt(252)
    res={'egp_sharpe':float(sh(pe)),'ew_sharpe':float(sh(ew)),'vn30_sharpe':float(sh(te['VN30'])),'egp_ann_ret':float(pe.mean()*252),'ew_ann_ret':float(ew.mean()*252),'vn30_ann_ret':float(te['VN30'].mean()*252),'weights':{k:float(v) for k,v in w.items()},
         'egp_vol':float(pe.std()*np.sqrt(252)),'ew_vol':float(ew.std()*np.sqrt(252)),'vn30_vol':float(te['VN30'].std()*np.sqrt(252))}
    # Sharpe diff test via bootstrap block
    rng=np.random.default_rng(0); n=len(te); diffs=[]
    for _ in range(2000):
        idx=np.concatenate([np.arange(s,s+10)%n for s in rng.integers(0,n,n//10+1)])[:n]
        diffs.append(sh(pe.iloc[idx])-sh(ew.iloc[idx]))
    res['sharpe_diff_ci']=[float(np.percentile(diffs,2.5)),float(np.percentile(diffs,97.5))]
print(res)
out['egp']=res
print(pd.DataFrame(out['sim'])[['bank','beta','lo','hi','alpha_bps','t_alpha','tot','sys','idio','beta_dimson']].round(3))

json.dump(out, open(OUT/'sim_results.json','w'), indent=1, default=float)
