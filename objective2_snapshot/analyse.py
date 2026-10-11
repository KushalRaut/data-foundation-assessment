"""Fit two eight-predictor ordinary least-squares regressions and evaluate honestly.
Outputs: chronological holdout, development folds, coefficient inference, diagnostics,
match-level bootstrap uncertainty, figures and predictions. No held-out tuning.
"""
from pathlib import Path
import json, sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from build_data import FEATURES1,FEATURES2
ROOT=Path(__file__).resolve().parent; DATA=ROOT/'data'; OUT=ROOT/'output';OUT.mkdir(exist_ok=True)
CUTOFF=pd.Timestamp('2026-07-02')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':130})

def design(d,features):return sm.add_constant(d[features].astype(float),has_constant='add')
def metrics(y,p):
    y=np.asarray(y);p=np.asarray(p)
    return dict(MAE=float(np.mean(abs(y-p))),RMSE=float(np.sqrt(np.mean((y-p)**2))),
                R2=float(1-np.sum((y-p)**2)/np.sum((y-y.mean())**2)))

def bootstrap(test,pred,base):
    # Resample entire matches, keeping both team observations together in model 2.
    ids=test.match_id.unique(); indices=[np.flatnonzero(test.match_id.to_numpy()==m) for m in ids]
    rng=np.random.default_rng(1402026); y=test.target.to_numpy(); results=[]
    for _ in range(3000):
        ix=np.concatenate([indices[i] for i in rng.integers(0,len(ids),len(ids))])
        result=metrics(y[ix],pred[ix]);result['MAE_improvement']=float(np.mean(abs(y[ix]-base[ix]))-np.mean(abs(y[ix]-pred[ix])))
        results.append(result)
    return {c:[float(x) for x in np.quantile([r[c] for r in results],[.025,.975])] for c in results[0]}

def run_model(label,frame,features):
    train=frame[frame.date<=CUTOFF].copy();test=frame[frame.date>CUTOFF].copy()
    X=design(train,features);Xt=design(test,features)
    assert np.linalg.matrix_rank(X)==9, (label,'rank deficient')
    assert not set(train.match_id)&set(test.match_id)
    fit=sm.OLS(train.target,X).fit()
    if label=='2_1':robust=fit.get_robustcov_results(cov_type='HC3',use_t=True);cov='HC3'
    else:robust=fit.get_robustcov_results(cov_type='cluster',groups=train.match_id,use_correction=True,df_correction=True,use_t=True);cov='clustered by match'
    pred=np.asarray(fit.predict(Xt));base=np.repeat(train.target.mean(),len(test))
    simple=sm.OLS(train.target,design(train,['elo_difference'])).fit()
    rows=[]
    for split,d,p,b in [('Training',train,np.asarray(fit.fittedvalues),np.repeat(train.target.mean(),len(train))),('Holdout',test,pred,base)]:
        for method,values in [('OLS eight predictors',p),('Training mean baseline',b),('Elo only baseline',np.asarray(simple.predict(design(d,['elo_difference']))))]:
            rows.append(dict(model=label,split=split,method=method,n_rows=len(d),n_matches=d.match_id.nunique(),**metrics(d.target,values)))
    if label=='2_1':rows.append(dict(model=label,split='Holdout',method='Zero difference baseline',n_rows=len(test),n_matches=test.match_id.nunique(),**metrics(test.target,np.zeros(len(test)))))
    pd.DataFrame(rows).to_csv(OUT/f'metrics_{label}.csv',index=False)
    ci=robust.conf_int();coefs=pd.DataFrame({'variable':X.columns,'coefficient':fit.params.values,'SE':robust.bse,'lower95':ci[:,0],'upper95':ci[:,1],'p_value':robust.pvalues})
    coefs['standardized_effect']=np.r_[np.nan,fit.params.values[1:]*train[features].std().to_numpy()/train.target.std()]
    coefs.to_csv(OUT/f'coefficients_{label}.csv',index=False)
    vif=pd.DataFrame({'variable':features,'VIF':[variance_inflation_factor(X.to_numpy(),i) for i in range(1,9)]})
    vif.to_csv(OUT/f'vif_{label}.csv',index=False)
    infl=fit.get_influence(); cooks=infl.cooks_distance[0]; lev=infl.hat_matrix_diag
    influence=train[['match_id','date','team','opponent','target']].copy()
    influence['cooks_distance']=cooks;influence['leverage']=lev;influence['studentized_residual']=infl.resid_studentized_external
    influence.sort_values('cooks_distance',ascending=False).to_csv(OUT/f'influence_{label}.csv',index=False)
    # Predetermined sensitivity: omit the most influential TRAINING match (both rows if needed).
    influential_match=train.iloc[int(np.argmax(cooks))].match_id
    sensitivity=train[train.match_id!=influential_match]
    alt=sm.OLS(sensitivity.target,design(sensitivity,features)).fit()
    altpred=np.asarray(alt.predict(Xt))
    predictions=test[['match_id','date','team','opponent','target']].copy()
    predictions['prediction']=pred;predictions['mean_baseline']=base;predictions['residual']=test.target.to_numpy()-pred
    predictions.to_csv(OUT/f'predictions_{label}.csv',index=False)
    dev_dates=np.sort(train.date.unique());cv=[]
    # Expanding chronological blocks of WHOLE DATES, with no random shuffling.
    chunks=np.array_split(dev_dates,4)
    for fold in range(1,4):
        tr=frame[frame.date.isin(np.concatenate(chunks[:fold]))];va=frame[frame.date.isin(chunks[fold])]
        xx=design(tr,features)
        assert tr.date.max()<va.date.min()
        assert np.linalg.matrix_rank(xx)==9
        f=sm.OLS(tr.target,xx).fit();p=np.asarray(f.predict(design(va,features)))
        for method,prediction in [('OLS eight predictors',p),('Training mean baseline',np.repeat(tr.target.mean(),len(va)))]:
            cv.append(dict(model=label,fold=fold,method=method,train_matches=tr.match_id.nunique(),validation_matches=va.match_id.nunique(),
                           train_end=str(tr.date.max().date()),validation_start=str(va.date.min().date()),validation_end=str(va.date.max().date()),**metrics(va.target,prediction)))
    pd.DataFrame(cv).to_csv(OUT/f'temporal_validation_{label}.csv',index=False)
    bp=het_breuschpagan(fit.resid,X); sh=stats.shapiro(fit.resid)
    info=dict(model=label,n_train=len(train),n_test=len(test),train_matches=int(train.match_id.nunique()),test_matches=int(test.match_id.nunique()),
        train_end=str(train.date.max().date()),test_start=str(test.date.min().date()),test_end=str(test.date.max().date()),
        train_R2=float(fit.rsquared),train_adjusted_R2=float(fit.rsquared_adj),train_residual_SD=float(np.sqrt(fit.mse_resid)),
        robust_covariance=cov,robust_joint_F=float(robust.fvalue),robust_joint_p=float(robust.f_pvalue),
        breusch_pagan_p=float(bp[1]),shapiro_p=float(sh.pvalue),max_VIF=float(vif.VIF.max()),
        cooks_threshold=4/len(train),n_cooks_flag=int(np.sum(cooks>4/len(train))),most_influential_match=influential_match,
        sensitivity_omit_match=metrics(test.target,altpred),sensitivity_max_prediction_change=float(np.max(abs(pred-altpred))),
        holdout=metrics(test.target,pred),baseline=metrics(test.target,base),bootstrap95=bootstrap(test,pred,base),
        negative_predictions=int(np.sum(pred<0)),holdout_goal_min=int(test.target.min()),holdout_goal_max=int(test.target.max()),
        coefficients={k:float(v) for k,v in fit.params.items()})
    if label=='2_2':info['clipped_prediction_sensitivity']=metrics(test.target,np.maximum(0,pred))
    (OUT/f'model_{label}.json').write_text(json.dumps(info,indent=2))
    (OUT/f'ols_summary_{label}.txt').write_text(robust.summary().as_text())
    # Charts use only training diagnostics and final holdout predictions, clearly labelled.
    fig,ax=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    ax[0,0].scatter(fit.fittedvalues,fit.resid,c='#24627d',alpha=.65);ax[0,0].axhline(0,color='black',lw=1)
    ax[0,0].set(xlabel='Fitted value',ylabel='Residual',title='Training residuals versus fitted values')
    sm.qqplot(fit.resid,line='s',ax=ax[0,1],markerfacecolor='#24627d',markeredgecolor='#24627d',alpha=.6)
    ax[0,1].set_title('Training residual normal Q-Q plot')
    ax[1,0].scatter(test.target,pred,color='#b55e3b',alpha=.7)
    lo=min(test.target.min(),pred.min());hi=max(test.target.max(),pred.max());ax[1,0].plot([lo,hi],[lo,hi],'k--',lw=1)
    ax[1,0].set(xlabel='Observed outcome',ylabel='Predicted outcome',title=f'Holdout predictions ({len(test)} rows)')
    ax[1,1].bar(np.arange(len(cooks)),cooks,color='#24627d');ax[1,1].axhline(4/len(train),color='#b55e3b',ls='--')
    ax[1,1].set(xlabel='Training observation',ylabel="Cook's distance",title='Influence of training observations')
    fig.suptitle(f'Regression {label.replace("_", ".")} diagnostics',fontsize=14);fig.savefig(OUT/f'diagnostics_{label}.png',dpi=180,bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.8),layout='constrained')
    ax.errorbar(coefs.coefficient.iloc[1:],np.arange(8),xerr=np.array([coefs.coefficient.iloc[1:]-coefs.lower95.iloc[1:],coefs.upper95.iloc[1:]-coefs.coefficient.iloc[1:]]),fmt='o',color='#24627d',capsize=3)
    ax.set_yticks(np.arange(8),[s.replace('_',' ') for s in features]);ax.axvline(0,color='black',lw=.8);ax.invert_yaxis()
    ax.set(xlabel=f'Coefficient with 95% interval ({cov})',title=f'Regression {label.replace("_", ".")} training coefficients')
    fig.savefig(OUT/f'coefficients_{label}.png',dpi=180,bbox_inches='tight');plt.close(fig)
    return info

def main():
    matches=pd.read_csv(DATA/'matches_fifa.csv',parse_dates=['date'])
    d1=pd.read_csv(DATA/'regression_2_1.csv').merge(matches[['match_id','date','team_a','team_b']],on='match_id',validate='one_to_one').rename(columns={'goal_difference':'target','team_a':'team','team_b':'opponent'})
    meta=pd.read_csv(DATA/'team_match_metadata.csv',parse_dates=['date'])
    d2=pd.read_csv(DATA/'regression_2_2.csv').merge(meta,on='row_id',validate='one_to_one').rename(columns={'goals':'target'})
    for label,df,features in [('2_1',d1,FEATURES1),('2_2',d2,FEATURES2)]:
        df[features+['target']].describe().T.to_csv(OUT/f'descriptive_{label}.csv')
        df.loc[df.date<=CUTOFF,features].corr().to_csv(OUT/f'correlation_{label}.csv')
    info=[run_model('2_1',d1,FEATURES1),run_model('2_2',d2,FEATURES2)]
    split=matches[['match_id','date','stage']].copy();split['split']=np.where(split.date<=CUTOFF,'Training','Holdout');split.to_csv(DATA/'split_manifest.csv',index=False)
    fig,ax=plt.subplots(1,2,figsize=(9,3.5),layout='constrained')
    for a,d,label in zip(ax,[d1,d2],['Goal difference (104 matches)','Team goals (208 rows)']):
        a.hist(d.target,bins=np.arange(d.target.min()-.5,d.target.max()+1.5),color='#24627d',edgecolor='white');a.set(xlabel=label,ylabel='Frequency')
    fig.savefig(OUT/'outcome_distributions.png',dpi=180,bbox_inches='tight');plt.close(fig)
    (OUT/'results.json').write_text(json.dumps(info,indent=2))
    print(json.dumps([{k:v for k,v in x.items() if k not in ['coefficients']} for x in info],indent=2))
if __name__=='__main__':main()
