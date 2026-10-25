"""Bank NIM determinants, 27 Vietnamese banks 2013-2023: corrected re-estimation of a thesis.

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

raw = pd.read_excel(args.data, sheet_name="Full"); raw.columns = [c.strip() for c in raw.columns]
# FIX 1: the text dummy "Có/Không" (Yes/No) was label-encoded alphabetically by Stata (Có=1, Không=2) -> sign inverted.
raw["COVID"] = raw["COVID-19"].map(lambda s: U.normalize("NFC", s) == "Có").astype(float)
df = raw.rename(columns={"ID": "bank", "Year": "year"})
bankv = ["CAR", "CAP", "LDR", "NPL"]; mac = ["ER", "INF", "GDP", "COVID"]; xs = bankv + mac
P = make_panel(df, "bank", "year", ["NIM"] + xs)
out = {}

# ===== 1. dummy audit and FGLS reproduction
xs=['CAR','CAP','LDR','NPL','ER','INF','GDP','COVID']
R={}
# audit
print(audit_dummy(P,'COVID',[2020,2021,2022]).T)
print(macro_identification(P,['ER','INF','GDP','COVID']))
# FGLS reproduction (xtgls, panels(hetero) corr(ar1)) -- Stata coding
P2=P.copy(); P2['COVID_s']=2-P2['COVID']
def xtgls(P,y,xs,iters=1):
    d=P.reset_index(); X=np.column_stack([np.ones(len(d)),d[xs]]); Y=d[y].to_numpy()
    b=np.linalg.lstsq(X,Y,rcond=None)[0]; e=Y-X@b
    d['e']=e
    # common AR(1) rho
    d['el']=d.groupby('bank')['e'].shift(1)
    m=d.dropna(subset=['el']); rho=(m.e*m.el).sum()/(m.el**2).sum()
    # Prais-Winsten
    Xs=[];Ys=[];W=[]
    for k,g in d.groupby('bank'):
        idx=g.index.to_numpy(); x=X[idx]; yv=Y[idx]
        xt=np.vstack([x[0]*np.sqrt(1-rho**2),x[1:]-rho*x[:-1]]); yt=np.r_[yv[0]*np.sqrt(1-rho**2),yv[1:]-rho*yv[:-1]]
        Xs.append(xt);Ys.append(yt)
    for _ in range(5):
        # hetero variances per panel from transformed residuals
        bt=np.linalg.lstsq(np.vstack(Xs),np.concatenate(Ys),rcond=None)[0]
        s=[np.mean((y_-x_@bt)**2) for x_,y_ in zip(Xs,Ys)]
        Xw=np.vstack([x_/np.sqrt(si) for x_,si in zip(Xs,s)]); Yw=np.concatenate([y_/np.sqrt(si) for y_,si in zip(Ys,s)])
        bt=np.linalg.lstsq(Xw,Yw,rcond=None)[0]
    cov=np.linalg.inv(Xw.T@Xw); se=np.sqrt(np.diag(cov))
    return pd.DataFrame({'b':bt,'se':se,'p':2*stats.norm.sf(abs(bt/se))},index=['const']+xs),rho
fg,rho=xtgls(P2,'NIM',['CAR','CAP','LDR','NPL','ER','INF','GDP','COVID_s'])
print(fg.round(5),rho)
fg2,_=xtgls(P2.assign(COVID_s=P2.COVID),'NIM',['CAR','CAP','LDR','NPL','ER','INF','GDP','COVID_s'])
print(fg2.round(5))

# ===== 2. main models and diagnostics
out={}
# 1. Models bank-clustered, corrected dummy
fits=fit_models(P,'NIM',xs)
ct=coef_table(fits); print(ct); out['coef']=ct.round(5).to_dict()
print('twfe params',fits.twfe.params.round(4).to_dict())
tw=fits.twfe
out['twfe']={k:[float(tw.params[k]),float(tw.std_errors[k]),float(tw.pvalues[k])] for k in tw.params.index}
out['r2']={'fe_within':float(fits.fe.rsquared),'fe_between':float(fits.fe.rsquared_between),'fe_overall':float(fits.fe.rsquared_overall),
           're_within':float(fits.re.rsquared),'re_between':float(fits.re.rsquared_between),'re_overall':float(fits.re.rsquared_overall),
           'pooled':float(fits.pooled.rsquared),'twfe_within':float(tw.rsquared)}
print(out['r2'])
# 2. diagnostics
out['F']={'stat':float(fits.fe.f_pooled.stat),'p':float(fits.fe.f_pooled.pval)}
out['BP']=breusch_pagan_lm(fits.pooled.resids); out['hausman']=hausman_classic(fits,bankv)
out['mundlak']=mundlak_test(P,'NIM',xs)
out['wool']=wooldridge_ar(P,'NIM',xs); out['mwald']=modified_wald(fits.fe.resids,len(xs))
out['cd_fe']=pesaran_cd(fits.fe.resids); out['cd_twfe']=pesaran_cd(fits.twfe.resids)
out['vif']=vif_table(P,xs).round(2).to_dict()
for k in ['F','BP','hausman','mundlak','wool','mwald','cd_fe','cd_twfe','vif']: print(k,out[k])
# 3. Macro: wild bootstrap by year on RE-demeaned? use FE-within-bank-demeaned? -> pooled with bank-FE removed via within transform
d=P.reset_index()
# year-level: regress bank-demeaned NIM on year-level vars: equals T-level analysis. Use pooled OLS on NIM with controls, cluster=year
Xm=np.column_stack([np.ones(len(d)),d[xs]]); y=d['NIM'].to_numpy()
b,se=cluster_se_ols(y,Xm,d['year'].to_numpy())
names=['const']+xs
wild={}
for j,n in enumerate(names):
    if n in mac:
        w=wild_cluster_p(y,Xm,d['year'].to_numpy(),j,B=1999,seed=1); wild[n]={'b':float(b[j]),'se_year':float(se[j]),**w}
print(pd.DataFrame(wild).T); out['wild_pooled']=wild
# with bank FE: demean by bank (within) then wild by year
dm=d.copy(); 
for c in ['NIM']+xs: dm[c]=d[c]-d.groupby('bank')[c].transform('mean')
Xf=dm[xs].to_numpy(); yf=dm['NIM'].to_numpy()
bf,sf=cluster_se_ols(yf,Xf,d['year'].to_numpy())
wildfe={}
for j,n in enumerate(xs):
    if n in mac:
        w=wild_cluster_p(yf,Xf,d['year'].to_numpy(),j,B=1999,seed=2); wildfe[n]={'b':float(bf[j]),'se_year':float(sf[j]),**w}
print(pd.DataFrame(wildfe).T); out['wild_fe']=wildfe
# 4. Ho-Saunders two-step
macro=d.groupby('year')[['GDP','INF','ER','COVID']].mean()
for cols in (['GDP','INF'],['GDP','INF','COVID'],['GDP','INF','ER']):
    hs=ho_saunders_two_step(d,'NIM',['CAP','LDR','NPL'],macro[cols])
    print(cols); print(hs['step2']); out['hs_'+'_'.join(cols)]=hs['step2'].round(5).to_dict()
    if cols==['GDP','INF','COVID']: out['pure']=hs['pure_spread'].round(4).to_dict(); out['gamma']=hs['gamma'].round(4).to_dict()
print(pd.Series(out['pure']))
# 5. dynamic
dd=df[['bank','year','NIM']+xs].copy(); dd['bank']=pd.factorize(dd['bank'])[0]+1
dd=dd.rename(columns={c:c.lower() for c in dd.columns})
try:
    cmd='nim L1.nim cap ldr npl | gmm(nim, 2:4) iv(cap ldr npl) | collapse nolevel timedumm'
    r=dynamic_gmm(dd,cmd)
    print(r)
except Exception as e: print('gmm err',e)
# Nickell bracket: pooled vs FE rho
d2=P.copy(); d2['L']=d2.groupby(level=0)['NIM'].shift(1); d2=d2.dropna()
rp=PooledOLS(d2['NIM'],sm.add_constant(d2[['L']+bankv])).fit(); 
rf=PanelOLS(d2['NIM'],d2[['L']+bankv],entity_effects=True).fit(cov_type='clustered',cluster_entity=True)
print('rho pooled',rp.params['L'],'rho FE',rf.params['L'],rf.std_errors['L'])
out['rho']={'pooled':float(rp.params['L']),'fe':float(rf.params['L']),'fe_se':float(rf.std_errors['L'])}

# ===== 3. dynamic panel with year dummies, Nickell, Anderson-Hsiao
# --- dynamic with year dummies: pooled vs FE bracket + AH + Nickell inversion
d=P.reset_index(); d['L']=d.groupby('bank')['NIM'].shift(1); d['L2']=d.groupby('bank')['NIM'].shift(2)
d['dY']=d.groupby('bank')['NIM'].diff()
d=d.set_index(['bank','year'])
dm=d.dropna(subset=['L'])
Yd=pd.get_dummies(dm.index.get_level_values(1),prefix='y',drop_first=True,dtype=float); Yd.index=dm.index
X=pd.concat([dm[['L']+bankv],Yd],axis=1)
rp=PooledOLS(dm['NIM'],sm.add_constant(X)).fit(cov_type='clustered',cluster_entity=True)
rf=PanelOLS(dm['NIM'],X,entity_effects=True).fit(cov_type='clustered',cluster_entity=True)
print('with year dummies: pooled rho',rp.params['L'],rp.std_errors['L'],' FE rho',rf.params['L'],rf.std_errors['L'])
for k in ['CAP','LDR','NPL','CAR']: print(k,'pooled',rp.params[k],'FE',rf.params[k])
out['dyn_year']={'pooled_rho':float(rp.params['L']),'fe_rho':float(rf.params['L']),'fe_se':float(rf.std_errors['L']),
  'fe':{k:[float(rf.params[k]),float(rf.std_errors[k]),float(rf.pvalues[k])] for k in ['L']+bankv}}
# Nickell inversion
def nick_bias(r,T):
    a=(1+r)/(T-1)
    num=1-(1/T)*(1-r**T)/(1-r)
    den=1-2*r/((1-r)*(T-1))*(1-(1-r**T)/(T*(1-r)))
    return -a*num/den
T_=10
f_=lambda r: r+nick_bias(r,T_)-rf.params['L']
rc=brentq(f_,0.01,0.95); print('Nickell-corrected rho (with year dummies)',rc)
rf0=PanelOLS(d.dropna(subset=['L'])['NIM'],d.dropna(subset=['L'])[['L']+bankv],entity_effects=True).fit()
rc0=brentq(lambda r:r+nick_bias(r,T_)-rf0.params['L'],0.01,0.95); print('no year dummies FE rho',rf0.params['L'],'corrected',rc0)
out['dyn_year']['nickell_rho']=float(rc); out['dyn_year']['nickell_rho_noyear']=float(rc0); out['dyn_year']['fe_rho_noyear']=float(rf0.params['L'])
# AH: Δy_t on Δy_{t-1}, Δx; instrument y_{t-2}
a=d.reset_index(); a['dL']=a.groupby('bank')['L'].diff()
for c in bankv: a['d'+c]=a.groupby('bank')[c].diff()
a=a.dropna(subset=['dY','dL','L2'])
Yd2=pd.get_dummies(a['year'],prefix='y',drop_first=True,dtype=float)
exog=pd.concat([a[['d'+c for c in bankv]],Yd2],axis=1)
ah=IV2SLS(a['dY'],exog,a[['dL']],a[['L2']]).fit(cov_type='clustered',clusters=a['bank'])
print('AH rho',ah.params['dL'],ah.std_errors['dL'],ah.first_stage.diagnostics.iloc[0].to_dict() if hasattr(ah,'first_stage') else '')
out['dyn_year']['AH']=[float(ah.params['dL']),float(ah.std_errors['dL'])]
# first-stage strength
fs=sm.OLS(a['dL'],sm.add_constant(pd.concat([exog,a[['L2']]],axis=1))).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(a['bank'])[0]})
print('first stage t of L2',fs.tvalues['L2'],'F',fs.tvalues['L2']**2)
out['dyn_year']['AH_firststage_F']=float(fs.tvalues['L2']**2)
# long run effects using bracket
b=rf.params; 
for k in ['CAP','LDR','NPL']:
    for r in (rc,0.87): print(k,'LR @rho',round(r,2),b[k]/(1-r))

# ===== 4. robustness
def fe(Pn,cols,y='NIM',te=False):
    r=PanelOLS(Pn[y],Pn[cols],entity_effects=True,time_effects=te,drop_absorbed=True).fit(cov_type='clustered',cluster_entity=True); return r
# standardized (bank-level) effects, FE/bank-cluster, 3 core vars
sd=P[bankv].std(); base=fe(P,bankv,te=True)
std={k:[float(base.params[k]*sd[k]),float(base.std_errors[k]*sd[k]),float(base.pvalues[k])] for k in bankv}
print('std effects (TWFE, per 1 sd) ',std); out['std']=std
# within-sd
wsd=(P[bankv]-P[bankv].groupby(level=0).transform('mean')).std()
out['std_within']={k:float(base.params[k]*wsd[k]) for k in bankv}; print(out['std_within'])
# winsorize
W=P.copy()
for c in ['NIM','NPL','LDR','CAP','CAR']: W[c]=winsorize(W[c],0.01)
rw=fe(W,xs); print('winsor FE',rw.params.round(4).to_dict(),rw.pvalues.round(4).to_dict())
out['winsor']={k:[float(rw.params[k]),float(rw.pvalues[k])] for k in rw.params.index}
# LOBO
for t in ['NPL','CAP','LDR']:
    l=leave_one_bank_out(P,'NIM',bankv+['GDP','INF'],t); print(t,l.iloc[[0,-1]].round(4).to_dict('index'),'max p',l.p.max())
    out['lobo_'+t]={'min':[l.index[0],float(l.coef.iloc[0])],'max':[l.index[-1],float(l.coef.iloc[-1])],'maxp':float(l.p.max())}
print('NPL max bank', P['NPL'].idxmax(), P['NPL'].max(), P['NIM'].idxmax(), P['NIM'].max(), P['LDR'].idxmax(),P['LDR'].max())
# drop extreme NPL obs
ex=P[P.NPL<10]; r=fe(ex,xs); print('NPL<10 FE NPL',r.params['NPL'],r.pvalues['NPL'],len(ex)); out['npl_lt10']=[float(r.params['NPL']),float(r.pvalues['NPL']),len(ex)]
# Ownership heterogeneity: SOCB
soc=['VCB','CTG','BID']
Q=P.copy(); Q['SOCB']=Q.index.get_level_values(0).isin(soc).astype(float)
for v in ['CAP','LDR','NPL']: Q[v+'_x']=Q[v]*Q['SOCB']
r=fe(Q,bankv+[v+'_x' for v in ['CAP','LDR','NPL']],te=True); print(pd.DataFrame({'b':r.params,'p':r.pvalues}))
out['socb']={k:[float(r.params[k]),float(r.pvalues[k])] for k in r.params.index}
print('NIM mean by SOCB',Q.groupby('SOCB')['NIM'].mean().to_dict())
# Year effects (event study vs 2019) with bank FE + controls
Y=pd.get_dummies(P.index.get_level_values(1),prefix='y',dtype=float); Y.index=P.index
Y=Y.drop(columns='y_2019'); E=pd.concat([P[bankv],Y],axis=1)
re=PanelOLS(P['NIM'],E,entity_effects=True).fit(cov_type='clustered',cluster_entity=True)
ev={c:[float(re.params[c]),float(re.std_errors[c]),float(re.pvalues[c])] for c in Y.columns}; print(pd.DataFrame(ev,index=['b','se','p']).T); out['event']=ev
# Mean NIM by year
mby=P.groupby(level=1)['NIM'].agg(['mean','median']); print(mby); out['nim_year']=mby.round(4).to_dict()
# COVID effect via ΔNIM design: pooled 2020-22 vs 2017-19 using bank FE +controls cluster bank (no year macro)
sub=P[P.index.get_level_values(1).isin([2017,2018,2019,2020,2021,2022])]
r=fe(sub,bankv+['COVID']); print('2017-22 FE covid',r.params['COVID'],r.std_errors['COVID'],r.pvalues['COVID'])
out['cov_1722']=[float(r.params['COVID']),float(r.std_errors['COVID']),float(r.pvalues['COVID'])]
# year-level wild with only COVID (no ER,GDP,INF)
d=P.reset_index(); dmn=d.copy()
for c in ['NIM']+bankv+['COVID']: dmn[c]=d[c]-d.groupby('bank')[c].transform('mean')
Xf=dmn[bankv+['COVID']].to_numpy(); yf=dmn['NIM'].to_numpy()
w=wild_cluster_p(yf,Xf,d['year'].to_numpy(),4,B=1999,seed=3); print('COVID only wild',w); out['wild_covid_only']=w
# Spec curve for COVID & GDP
rows=[]
for k in range(0,4):
    for macro_other in itertools.combinations(['ER','INF','GDP'],k):
        cols=bankv+list(macro_other)+['COVID']
        for te in [False]:
            r=fe(P,cols)
            rows.append(('COVID',','.join(macro_other),float(r.params['COVID']),float(r.pvalues['COVID'])))
sc=pd.DataFrame(rows,columns=['v','others','b','p']); print(sc)
out['spec_covid']=sc.round(4).values.tolist()
rows=[]
for k in range(0,3):
    for o in itertools.combinations(['ER','INF','COVID'],k):
        r=fe(P,bankv+['GDP']+list(o)); rows.append(('GDP',','.join(o),float(r.params['GDP']),float(r.pvalues['GDP'])))
sg=pd.DataFrame(rows,columns=['v','others','b','p']); print(sg); out['spec_gdp']=sg.round(4).values.tolist()
# rolling origin
ro=rolling_origin_eval(P,'NIM',bankv+['INF','GDP','COVID'],2017); print(ro); out['rolling']=ro.round(4).to_dict()
print(ro.mean())
out['rolling_mean']=ro.mean().round(4).to_dict()

json.dump(out, open(OUT/"kltn_results.json", "w"), indent=1, default=float)

# ===== 5. figures
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fits=fit_models(P,'NIM',xs)
tab={}
for n in ['pooled','fe','re']:
    r=getattr(fits,n); tab[n]={k:[float(r.params[k]),float(r.std_errors[k]),float(r.pvalues[k])] for k in xs}
print(pd.DataFrame({n:{k:f"{v[0]:.4f} ({v[1]:.4f}) p={v[2]:.3f}" for k,v in tab[n].items()} for n in tab}))
# descriptives by period correct dummy
print(P.groupby('COVID')[['NIM','CAP','LDR','NPL','GDP','INF','ER']].agg(['mean','std']).T.round(3))
# Fig1
o=out
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(6.4,3.3))
g=P.groupby(level=1)['NIM']; yrs=g.mean().index
q=g.quantile([.25,.75]).unstack()
ax.axvspan(2019.5,2022.5,color='#e8e8e8',zorder=0); ax.text(2021,q[0.75].max()+0.05,'COVID-19 (2020–2022)',ha='center',fontsize=8,color='#555')
ax.fill_between(yrs,q[0.25],q[0.75],color='#9ecae1',alpha=.5,label='Interquartile range')
ax.plot(yrs,g.mean(),'-o',color='#08519c',label='Mean'); ax.plot(yrs,g.median(),'--s',color='#d94801',ms=4,label='Median')
ax.set_xticks(list(yrs)); ax.set_xticklabels([str(y)[2:] for y in yrs]); ax.set_ylabel('NIM (%)'); ax.set_xlabel('Year (20xx)'); ax.legend(frameon=False,fontsize=8,loc='upper left')
fig.tight_layout(); fig.savefig(OUT/'figures'/'fig1_nim.png',dpi=200)
# Fig2: p-values bank-cluster vs wild-by-year
fig,ax=plt.subplots(figsize=(6.4,3.2))
labels=['GDP','COVID','INF','ER']
pb=[tab['fe'][k][2] for k in labels]; pw=[o['wild_fe'][k]['p_wild'] for k in labels]
xx=np.arange(4); ax.bar(xx-.18,pb,.36,color='#9ecae1',label='Bank-clustered SE (N=27)'); ax.bar(xx+.18,pw,.36,color='#08519c',label='Wild bootstrap by year (G=11)')
ax.axhline(.05,color='r',ls='--',lw=1); ax.text(3.45,.06,'5%',color='r',fontsize=8)
ax.set_xticks(xx); ax.set_xticklabels(labels); ax.set_ylabel('p-value (FE model)'); ax.legend(frameon=False,fontsize=8)
fig.tight_layout(); fig.savefig(OUT/'figures'/'fig2_p.png',dpi=200)
# Fig3 rho bracket / coefficient CAP,LDR static vs lag
fig,ax=plt.subplots(figsize=(6.4,3.2))
dy=o['dyn_year']['fe']; st={'CAP':tab['fe']['CAP'],'LDR':tab['fe']['LDR'],'NPL':tab['fe']['NPL']}
for i,k in enumerate(['CAP','LDR','NPL']):
    b,se,_=st[k]; ax.errorbar(i-.12,b,1.96*se,fmt='o',color='#08519c',capsize=3,label='Static FE' if i==0 else None)
    b2,se2,_=dy[k]; ax.errorbar(i+.12,b2,1.96*se2,fmt='s',color='#d94801',capsize=3,label='FE with NIM$_{t-1}$ + year effects' if i==0 else None)
ax.axhline(0,color='k',lw=.8); ax.set_xticks(range(3)); ax.set_xticklabels(['CAP','LDR','NPL']); ax.set_ylabel('Coefficient (±1.96 bank-clustered SE)'); ax.legend(frameon=False,fontsize=8,loc='upper right')
fig.tight_layout(); fig.savefig(OUT/'figures'/'fig3_dyn.png',dpi=200)
