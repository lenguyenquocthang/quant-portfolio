"""Sleep-disorder logit: reproduction, leakage-safe validation, Firth correction.

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
warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--out", default="results")
args = ap.parse_args(); OUT = Path(args.out); OUT.mkdir(exist_ok=True, parents=True); (OUT/"figures").mkdir(exist_ok=True)
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.discrete.discrete_model import MNLogit

d = pd.read_excel(args.data, keep_default_na=False); d.columns = [c.strip() for c in d.columns]
out = {}

# ===== s4.py
d['y']=(d['Sleep Disorder']!='None').astype(int); d['normal']=1-d.y
d['BMI']=d['BMI Category'].replace({'Normal Weight':'Normal'})
d['dur_g']=pd.cut(d['Sleep Duration'],[0,6.0,6.9,8.0,10],labels=['short','fair','good','long'])
d['q_g']=pd.cut(d['Quality of Sleep'],[0,6,8,10],labels=['normal','good','very_good'])
d['s_g']=pd.cut(d['Stress Level'],[0,4,6,10],labels=['low34','mid56','high78'])
# --- reproduce paper multivariate logit for P(normal): base duration=it, quality=tot, stress=bt (3-4)
X=pd.concat([pd.get_dummies(d.dur_g,dtype=float).drop(columns='short'),
             pd.get_dummies(d.q_g,dtype=float).drop(columns='good'),
             pd.get_dummies(d.s_g,dtype=float).drop(columns='low34')],axis=1)
m=sm.GLM(d.normal,sm.add_constant(X),family=sm.families.Binomial()).fit()
print(m.summary().tables[1]); print('dev',m.deviance,'null',m.null_deviance,'aic',m.aic,'iter',m.fit_history['iteration'] if 'iteration' in m.fit_history else '')
p=m.predict(sm.add_constant(X)); pred=(p>0.5).astype(int)
tn=((pred==0)&(d.normal==0)).sum(); tp=((pred==1)&(d.normal==1)).sum(); fp=((pred==1)&(d.normal==0)).sum(); fn=((pred==0)&(d.normal==1)).sum()
print('confusion normal-positive: tp',tp,'fn',fn,'fp',fp,'tn',tn,'acc',(tp+tn)/len(d),'brier',brier_score_loss(d.normal,p),'auc',roc_auc_score(d.normal,p))
out['repro']={'dev':float(m.deviance),'null':float(m.null_deviance),'aic':float(m.aic),'acc':float((tp+tn)/len(d)),'brier':float(brier_score_loss(d.normal,p)),'auc':float(roc_auc_score(d.normal,p))}
out['repro_coef']={k:[float(m.params[k]),float(m.bse[k]),float(m.pvalues[k])] for k in m.params.index}
# separation diagnostic
print(pd.crosstab([d.q_g],[d.s_g,d.y]))
print(pd.crosstab(d.q_g,d.y)); print(pd.crosstab(d.s_g,d.y))

# ===== s5.py
d['y']=(d['Sleep Disorder']!='None').astype(int)
d['BMI']=d['BMI Category'].replace({'Normal Weight':'Normal'})
d['dur_g']=pd.cut(d['Sleep Duration'],[0,6.0,6.9,8.0,10],labels=['short','fair','good','long'])
d['q_g']=pd.cut(d['Quality of Sleep'],[0,6,8,10],labels=['normal','good','very_good'])
d['s_g']=pd.cut(d['Stress Level'],[0,4,6,10],labels=['low34','mid56','high78'])
cov=['Gender','Age','Occupation','Sleep Duration','Quality of Sleep','Physical Activity Level','Stress Level','BMI','Blood Pressure','Heart Rate','Daily Steps']
d['grp']=d[cov].astype(str).agg('|'.join,axis=1).factorize()[0]
print('groups',d.grp.nunique())
def design(cols_num,cols_cat):
    X=d[cols_num].astype(float).copy()
    for c in cols_cat: X=pd.concat([X,pd.get_dummies(d[c],prefix=c,drop_first=True,dtype=float)],axis=1)
    return X
specs={
 'M0 original paper (binned duration, quality, stress)':design([],['dur_g','q_g','s_g']),
 'M1 no quality: duration (numeric), stress (each level)':design(['Sleep Duration'],['Stress Level']),
 'M2 = M1 + age, BMI, gender':design(['Sleep Duration','Age'],['Stress Level','BMI','Gender']),
 'M3 = M2 + occupation':design(['Sleep Duration','Age'],['Stress Level','BMI','Gender','Occupation']),
 'M4 = M2 + sleep quality (simultaneous outcome variable)':design(['Sleep Duration','Age','Quality of Sleep'],['Stress Level','BMI','Gender']),
}
def cv_eval(X,y,groups,C,reps=20):
    res=[]
    for r in range(reps):
        rng=np.random.RandomState(r); perm=rng.permutation(np.unique(groups)); mp={g:i for i,g in enumerate(perm)}; gg=np.array([mp[g] for g in groups])
        oof=np.zeros(len(y))
        for tr,te in GroupKFold(5).split(X,y,gg%1000 if False else gg):
            sc=StandardScaler().fit(X.iloc[tr]); 
            m=LogisticRegression(C=C,max_iter=5000).fit(sc.transform(X.iloc[tr]),y[tr]); oof[te]=m.predict_proba(sc.transform(X.iloc[te]))[:,1]
        res.append((roc_auc_score(y,oof),brier_score_loss(y,oof),((oof>.5)==y).mean()))
    a=np.array(res); return a.mean(0),a.std(0)
def cv_naive(X,y,C,reps=20):
    from sklearn.model_selection import StratifiedKFold
    res=[]
    for r in range(reps):
        oof=np.zeros(len(y))
        for tr,te in StratifiedKFold(5,shuffle=True,random_state=r).split(X,y):
            sc=StandardScaler().fit(X.iloc[tr]); m=LogisticRegression(C=C,max_iter=5000).fit(sc.transform(X.iloc[tr]),y[tr]); oof[te]=m.predict_proba(sc.transform(X.iloc[te]))[:,1]
        res.append((roc_auc_score(y,oof),brier_score_loss(y,oof)))
    return np.array(res).mean(0)
y=d.y.values; rows=[]; base=y.mean()*(1-y.mean())
for name,X in specs.items():
    sc=StandardScaler().fit(X); mi=LogisticRegression(C=1.0,max_iter=5000).fit(sc.transform(X),y); pin=mi.predict_proba(sc.transform(X))[:,1]
    (a,b,acc),(sa,sb,_)=cv_eval(X,y,d.grp.values,1.0)
    an,bn=cv_naive(X,y,1.0)
    rows.append([name,X.shape[1],roc_auc_score(y,pin),brier_score_loss(y,pin),a,sa,b,an,bn,acc])
    print(rows[-1])
R=pd.DataFrame(rows,columns=['model','k','auc_in','brier_in','auc_cvgroup','auc_sd','brier_cvgroup','auc_cvnaive','brier_cvnaive','acc_cvgroup'])
print(R.round(3).to_string()); print('Brier null',base)
out['cv']=R.round(4).to_dict('records'); out['brier_null']=float(base); out['n_groups']=int(d.grp.nunique())

# ===== s6.py
d['y']=(d['Sleep Disorder']!='None').astype(int)
d['BMI']=d['BMI Category'].replace({'Normal Weight':'Normal'})
cov=['Gender','Age','Occupation','Sleep Duration','Quality of Sleep','Physical Activity Level','Stress Level','BMI','Blood Pressure','Heart Rate','Daily Steps']
d['grp']=d[cov].astype(str).agg('|'.join,axis=1).factorize()[0]
print(pd.crosstab(d.BMI,d['Sleep Disorder'],margins=True)); print(pd.crosstab(d.BMI,d.y,normalize='index').round(3))
print(d.groupby('y')[['Age','Sleep Duration','Stress Level','Quality of Sleep']].mean().round(2))
print(pd.crosstab(pd.cut(d.Age,[0,35,45,60]),d['Sleep Disorder'],normalize='index').round(3))
X=pd.concat([d[['Sleep Duration','Age']],pd.get_dummies(d['Stress Level'],prefix='S',drop_first=True,dtype=float),pd.get_dummies(d.BMI,prefix='BMI',drop_first=True,dtype=float),pd.get_dummies(d.Gender,prefix='G',drop_first=True,dtype=float)],axis=1)
m=sm.Logit(d.y,sm.add_constant(X)).fit(disp=0,cov_type='cluster',cov_kwds={'groups':d.grp})
print(m.summary().tables[1]); print('McFadden',m.prsquared,'LLR p',m.llr_pvalue,'converged',m.mle_retvals['converged'])
tab=pd.DataFrame({'b':m.params,'se':m.bse,'p':m.pvalues,'OR':np.exp(m.params),'lo':np.exp(m.params-1.96*m.bse),'hi':np.exp(m.params+1.96*m.bse)}); print(tab.round(3))
# Maximum-likelihood perfect separation check on M2
print('max |b|',m.params.abs().max())
# AME for BMI Overweight & per-year age & per-hour sleep
Xc=sm.add_constant(X)
def ame(var,delta=1):
    X1=Xc.copy();X0=Xc.copy(); 
    if var in ('Sleep Duration','Age'): X1[var]+=delta
    else: X1[var]=1; X0[var]=0
    return float((m.predict(X1)-m.predict(X0)).mean())
out['m2']={k:[float(tab.loc[k,'b']),float(tab.loc[k,'se']),float(tab.loc[k,'p']),float(tab.loc[k,'OR']),float(tab.loc[k,'lo']),float(tab.loc[k,'hi'])] for k in tab.index}
out['m2_fit']={'mcf':float(m.prsquared),'llr_p':float(m.llr_pvalue),'aic':float(m.aic)}
out['ame']={'BMI_Overweight':ame('BMI_Overweight'),'BMI_Obese':ame('BMI_Obese') if 'BMI_Obese' in Xc else None,'Sleep Duration +1h':ame('Sleep Duration'),'Age +10':ame('Age',10)}
print(out['ame'])
# multinomial
d['cls']=d['Sleep Disorder'].map({'None':0,'Insomnia':1,'Sleep Apnea':2})
mm=MNLogit(d.cls,sm.add_constant(X)).fit(disp=0,maxiter=200); print(mm.summary().tables[1] if False else '')
pp=mm.predict(sm.add_constant(X)); pred=pp.values.argmax(1); print('multinomial acc',(pred==d.cls).mean()); print(pd.crosstab(d.cls,pred))
out['mnl']={'acc':float((pred==d.cls).mean()),'conf':pd.crosstab(d.cls,pred).values.tolist()}
print(pd.crosstab(d.BMI,d['Sleep Disorder']))

# ===== s7.py
d['y']=(d['Sleep Disorder']!='None').astype(int)
d['OW']=(d['BMI Category'].isin(['Overweight','Obese'])).astype(float)
d['male']=(d.Gender=='Male').astype(float)
d['stressHi']=(d['Stress Level']>=7).astype(float)
cov=['Gender','Age','Occupation','Sleep Duration','Quality of Sleep','Physical Activity Level','Stress Level','BMI Category','Blood Pressure','Heart Rate','Daily Steps']
d['grp']=d[cov].astype(str).agg('|'.join,axis=1).factorize()[0]
def firth(X,y,it=100):
    X=np.asarray(X,float); y=np.asarray(y,float); b=np.zeros(X.shape[1])
    for _ in range(it):
        p=1/(1+np.exp(-X@b)); W=p*(1-p); I=X.T@(X*W[:,None]); Iinv=np.linalg.inv(I)
        h=np.einsum('ij,jk,ik->i',X,Iinv,X)*W
        U=X.T@(y-p+h*(0.5-p)); step=Iinv@U
        if np.max(np.abs(step))>5: step=step/np.max(np.abs(step))*5
        b=b+step
        if np.max(np.abs(step))<1e-8: break
    p=1/(1+np.exp(-X@b)); W=p*(1-p); cov_=np.linalg.inv(X.T@(X*W[:,None]))
    return b,np.sqrt(np.diag(cov_))
# models
def fit_firth(cols,name):
    X=sm.add_constant(d[cols]); b,se=firth(X,d.y)
    tab=pd.DataFrame({'b':b,'se':se,'OR':np.exp(b),'lo':np.exp(b-1.96*se),'hi':np.exp(b+1.96*se),'p':2*stats.norm.sf(abs(b/se))},index=X.columns); print(name); print(tab.round(3)); return tab
t_full=fit_firth(['OW','Age','male','Sleep Duration','stressHi'],'Firth: BMI (overweight/obese), age, gender, sleep duration, high stress (>=7)')
# nested LR (ordinary logit w/ OW binary: no separation now?)
def ll(cols):
    X=sm.add_constant(d[cols]); m=sm.Logit(d.y,X).fit(disp=0,maxiter=200); return m
m_full=ll(['OW','Age','male','Sleep Duration','stressHi']); print(m_full.params.round(3).to_dict(), m_full.mle_retvals['converged'])
m_nodur=ll(['OW','Age','male','stressHi']); LR=2*(m_full.llf-m_nodur.llf); print('LR dur',LR,stats.chi2.sf(LR,1))
m_nostress=ll(['OW','Age','male','Sleep Duration']); LR2=2*(m_full.llf-m_nostress.llf); print('LR stress',LR2,stats.chi2.sf(LR2,1))
m_noow=ll(['Age','male','Sleep Duration','stressHi']); LR3=2*(m_full.llf-m_noow.llf); print('LR OW',LR3,stats.chi2.sf(LR3,1))
# cluster robust (profile) for full
mc=sm.Logit(d.y,sm.add_constant(d[['OW','Age','male','Sleep Duration','stressHi']])).fit(disp=0,maxiter=200,cov_type='cluster',cov_kwds={'groups':d.grp}); print(mc.pvalues.round(4).to_dict())
# raw association: duration alone and stress alone controlling nothing; and within BMI strata
for lab,g in d.groupby('OW'):
    mm=sm.Logit(g.y,sm.add_constant(g[['Sleep Duration']])).fit(disp=0); print('OW',lab,'n',len(g),'dur coef',round(mm.params['Sleep Duration'],3),round(mm.pvalues['Sleep Duration'],4),'y rate',round(g.y.mean(),3))
print(pd.crosstab(d.OW,d.y))
print(d.groupby('OW')[['Sleep Duration','Age','Stress Level']].mean().round(2))
out['firth']={k:[float(t_full.loc[k,c]) for c in ['b','se','OR','lo','hi','p']] for k in t_full.index}
out['lr']={'dur':[float(LR),float(stats.chi2.sf(LR,1))],'stress':[float(LR2),float(stats.chi2.sf(LR2,1))],'ow':[float(LR3),float(stats.chi2.sf(LR3,1))]}
out['cluster_p']=mc.pvalues.round(4).to_dict()

json.dump(out, open(OUT/'sleep_results.json','w'), indent=1, default=float)
