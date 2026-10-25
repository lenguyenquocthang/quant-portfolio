"""Bank NIM practicum (27 banks, 2018-2023): corrected re-estimation.

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
ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--kltn-data", default=None, help="thesis workbook (only used to back-fill NPL missing values)"); ap.add_argument("--out", default="results")
args = ap.parse_args(); OUT = Path(args.out); OUT.mkdir(exist_ok=True, parents=True); (OUT/"figures").mkdir(exist_ok=True)

a = pd.read_excel(args.data); a.columns = [c.strip() for c in a.columns]
k = pd.read_excel(args.kltn_data, sheet_name="Full"); k.columns = [c.strip() for c in k.columns]
a['covid_txt']=a.COVID.map(lambda s:U.normalize('NFC',s)=='Có')
a['COVID']=a.Year.isin([2020,2021,2022]).astype(float)       # correct
a['COVID_data']=a.covid_txt.astype(float)                    # label as stored in file (2019-2022)
a['COVID_2021']=a.Year.isin([2020,2021]).astype(float)
a=a.rename(columns={'ID':'bank','Year':'year'})
# NPL zeros -> use KLTN file (flag)
kk=k.rename(columns={'ID':'bank','Year':'year'})[['bank','year','NPL']].rename(columns={'NPL':'NPLk'})
a=a.merge(kk,on=['bank','year'],how='left')
zero=a.NPL==0; print('zeros',zero.sum())
a['NPL_raw']=a.NPL; a.loc[zero,'NPL']=a.loc[zero,'NPLk']
a['lnA']=a.SIZE*np.log(2)    # SIZE is log2
xs=['NPL','SIZE','CAP','INF','GDP','COVID']; bankv=['NPL','SIZE','CAP']; mac=['INF','GDP','COVID']
out={}
def run(df,xs,tag):
    P=make_panel(df,'bank','year',['NIM']+xs)
    f=fit_models(P,'NIM',xs); return P,f
P,f=run(a,xs,'main')
ct=coef_table(f); print(ct)
tab={}
for n in ['pooled','fe','re']:
    r=getattr(f,n); tab[n]={kx:[float(r.params[kx]),float(r.std_errors[kx]),float(r.pvalues[kx])] for kx in xs}
tw=f.twfe; out['twfe']={kx:[float(tw.params[kx]),float(tw.std_errors[kx]),float(tw.pvalues[kx])] for kx in tw.params.index if kx!='const'}
out['tab']=tab
out['r2']={'pooled':float(f.pooled.rsquared),'fe_within':float(f.fe.rsquared),'re_overall':float(f.re.rsquared_overall),'twfe_within':float(tw.rsquared)}
print(out['r2'])
out['F']={'stat':float(f.fe.f_pooled.stat),'p':float(f.fe.f_pooled.pval)}
out['BP']=breusch_pagan_lm(f.pooled.resids); out['hausman']=hausman_classic(f,bankv)
out['mundlak']=mundlak_test(P,'NIM',xs); out['wool']=wooldridge_ar(P,'NIM',xs)
out['mwald']=modified_wald(f.fe.resids,len(xs)); out['cd_fe']=pesaran_cd(f.fe.resids,min_overlap=4); out['cd_tw']=pesaran_cd(f.twfe.resids,min_overlap=4)
out['vif']=vif_table(P,xs).round(2).to_dict()
for kx in ['F','BP','hausman','mundlak','wool','mwald','cd_fe','cd_tw','vif']: print(kx,out[kx])
# Stata-coding reproduction of paper as-is data (NPL raw zeros)
a2=a.copy(); a2['COVID_s']=a2.covid_txt.map({True:1.0,False:2.0}); a2['NPL']=a2.NPL_raw
P2=make_panel(a2,'bank','year',['NIM','NPL','SIZE','CAP','INF','GDP','COVID_s'])
f2=fit_models(P2,'NIM',['NPL','SIZE','CAP','INF','GDP','COVID_s'])
print('REPRO'); print(coef_table(f2)[['pooled_b','fe_b','re_b']])
out['repro']={n:{kx:float(getattr(f2,n).params[kx]) for kx in ['NPL','SIZE','CAP','INF','GDP','COVID_s']} for n in ['pooled','fe','re']}
# NPL raw (zeros) with correct dummy
P3=make_panel(a.assign(NPL=a.NPL_raw),'bank','year',['NIM']+xs); f3=fit_models(P3,'NIM',xs)
out['npl_zero_fe']=[float(f3.fe.params['NPL']),float(f3.fe.std_errors['NPL']),float(f3.fe.pvalues['NPL'])]
print('NPL zeros FE',out['npl_zero_fe'], 'vs corrected', tab['fe']['NPL'])
# dummy sensitivity
for cv in ['COVID','COVID_data','COVID_2021']:
    Q=make_panel(a,'bank','year',['NIM','NPL','SIZE','CAP','INF','GDP',cv]); ff=fit_models(Q,'NIM',['NPL','SIZE','CAP','INF','GDP',cv])
    print(cv,'FE',ff.fe.params[cv],ff.fe.pvalues[cv],'GDP',ff.fe.params['GDP'],ff.fe.pvalues['GDP'])
    out['sens_'+cv]={'covid':[float(ff.fe.params[cv]),float(ff.fe.pvalues[cv])],'gdp':[float(ff.fe.params['GDP']),float(ff.fe.pvalues['GDP'])]}



xs=['NPL','SIZE','CAP','INF','GDP','COVID']

xs=['NPL','SIZE','CAP','INF','GDP','COVID']; bankv=['NPL','SIZE','CAP']; mac=['INF','GDP','COVID']
P=make_panel(a,'bank','year',['NIM']+xs); d=P.reset_index()
# macro identification
print(macro_identification(P,mac))
print(d.groupby('year')[mac].mean().corr().round(2))
# wild by year (FE-demeaned), Webb
cols=['NIM']+xs; dm=d[cols]-d.groupby('bank')[cols].transform('mean')
X=dm[xs].to_numpy(); y=dm['NIM'].to_numpy(); yr=d.year.to_numpy()
b,se=cluster_se_ols(y,X,yr); print('year-cluster se',dict(zip(xs,se.round(4))))
wild={}
for j,n in enumerate(xs):
    if n in mac:
        w=wild_cluster_p(y,X,yr,j,B=1999,seed=2); wild[n]={'b':float(b[j]),'se_year':float(se[j]),**w}
print(pd.DataFrame(wild).T); out['wild']=wild
# critical t with T-k-1 df
print('t crit df=2',stats.t.ppf(.975,2),'df=6-3-1=2')
# Ho-Saunders
macro=d.groupby('year')[['GDP','INF','COVID']].mean()
for cols_ in (['GDP','INF'],['GDP','INF','COVID']):
    hs=ho_saunders_two_step(d,'NIM',bankv,macro[cols_]); print(cols_); print(hs['step2']); out['hs_'+'_'.join(cols_)]=hs['step2'].round(5).to_dict()
# Standardised, within/between
tw=PanelOLS(P['NIM'],P[bankv],entity_effects=True,time_effects=True).fit(cov_type='clustered',cluster_entity=True)
fe=PanelOLS(P['NIM'],P[xs],entity_effects=True).fit(cov_type='clustered',cluster_entity=True)
sd=P[bankv].std(); wsd=(P[bankv]-P[bankv].groupby(level=0).transform('mean')).std()
print('between sd',P[bankv].groupby(level=0).mean().std().round(3).to_dict(),'within sd',wsd.round(3).to_dict(),'total',sd.round(3).to_dict())
out['std']={k:[float(fe.params[k]*sd[k]),float(fe.params[k]*wsd[k]),float(fe.pvalues[k])] for k in bankv}
out['sd']={'tot':sd.round(4).to_dict(),'within':wsd.round(4).to_dict()}
print(out['std'])
print('TWFE',tw.params.round(4).to_dict(),tw.pvalues.round(4).to_dict())
# between regression (bank means)
Bm=P.groupby(level=0).mean()
rb=sm.OLS(Bm['NIM'],sm.add_constant(Bm[bankv])).fit(cov_type='HC1'); print(rb.params.round(4).to_dict(),rb.pvalues.round(4).to_dict()); out['between']={k:[float(rb.params[k]),float(rb.pvalues[k])] for k in bankv}
# Winsor
W=P.copy()
for c_ in ['NIM','NPL','CAP']: W[c_]=winsorize(W[c_],0.01)
rw=PanelOLS(W['NIM'],W[xs],entity_effects=True).fit(cov_type='clustered',cluster_entity=True); print('winsor',rw.params.round(4).to_dict(),rw.pvalues.round(4).to_dict())
out['winsor']={k:[float(rw.params[k]),float(rw.pvalues[k])] for k in xs}
# LOBO
for t in ['NPL','SIZE','CAP']:
    l=leave_one_bank_out(P,'NIM',xs,t); print(t,l.iloc[[0,-1]].round(4).to_dict('index'),'max p',l.p.max())
    out['lobo_'+t]={'min':[l.index[0],float(l.coef.iloc[0]),float(l.p.iloc[0])],'max':[l.index[-1],float(l.coef.iloc[-1]),float(l.p.iloc[-1])],'maxp':float(l.p.max())}
print(P.NPL.idxmax(),P.NPL.max())
r=PanelOLS(P[P.NPL<10]['NIM'],P[P.NPL<10][xs],entity_effects=True).fit(cov_type='clustered',cluster_entity=True); print('NPL<10',r.params['NPL'],r.pvalues['NPL'],len(P[P.NPL<10]))
out['npl_lt10']=[float(r.params['NPL']),float(r.pvalues['NPL']),int((P.NPL<10).sum())]
# Year means
print(P.groupby(level=1)[['NIM','CAP','SIZE','NPL']].mean().round(3)); out['ymean']=P.groupby(level=1)[['NIM','CAP','SIZE','NPL']].mean().round(4).to_dict()
# dynamic bracket
P2=P.copy(); P2['L']=P2.groupby(level=0)['NIM'].shift(1); P2=P2.dropna()
Yd=pd.get_dummies(P2.index.get_level_values(1),prefix='y',drop_first=True,dtype=float); Yd.index=P2.index
Xd=pd.concat([P2[['L']+bankv],Yd],axis=1)
rp=PooledOLS(P2['NIM'],sm.add_constant(Xd)).fit(cov_type='clustered',cluster_entity=True); rf=PanelOLS(P2['NIM'],Xd,entity_effects=True).fit(cov_type='clustered',cluster_entity=True)
print('rho pooled',rp.params['L'],'FE',rf.params['L'],rf.std_errors['L'])
for kx in bankv: print(kx,'pooled',rp.params[kx],'FE',rf.params[kx],rf.pvalues[kx])
def nb(r,T):
    a_=(1+r)/(T-1); num=1-(1/T)*(1-r**T)/(1-r); den=1-2*r/((1-r)*(T-1))*(1-(1-r**T)/(T*(1-r))); return -a_*num/den
T_=5; 
try: rc=brentq(lambda r:r+nb(r,T_)-rf.params['L'],-0.9,0.95)
except Exception as e: rc=float('nan')
print('Nickell corrected (T=5)',rc, 'bias at rho=.5',nb(.5,5))
out['dyn']={'pooled_rho':float(rp.params['L']),'fe_rho':float(rf.params['L']),'fe_se':float(rf.std_errors['L']),'nick':float(rc),'fe':{k:[float(rf.params[k]),float(rf.std_errors[k]),float(rf.pvalues[k])] for k in ['L']+bankv}}
# rolling origin
ro=rolling_origin_eval(P,'NIM',bankv+['INF','GDP','COVID'],2021); print(ro); out['rolling']=ro.round(4).to_dict(); out['rolling_mean']=ro.mean().round(4).to_dict(); print(ro.mean())


json.dump(out, open(OUT/'thnn_results.json','w'), indent=1, default=float)
