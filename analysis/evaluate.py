"""Evaluate frozen frequency-severity quotes, including policy bootstrap."""
from pathlib import Path
import sys,json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent; ROOT=OUT.parent
sys.path.insert(0,str(ROOT))
from metrics import score_groups,gini_from_groups,top_fraction
D=pd.read_csv(OUT/'test_policy_pricing_matched.csv')
y=D.observed_period_loss.to_numpy();e=D.Exposure.to_numpy()
M=['mean','expectile','quantile']; rows=[];scores={};per={};penalties=[]
for m in M:
 p=D[m+'_period'].to_numpy();a=D[m+'_annual'].to_numpy();u=np.maximum(y-p,0);v=np.maximum(p-y,0)
 group=score_groups(a,e,y,p);flag=top_fraction(group,.1)
 us=np.sort(u)[::-1];n=.01*len(u);k=int(n)
 row=dict(model=m,test_lr=y.sum()/p.sum(),premium_total=p.sum(),zero_prices=int((p==0).sum()),
     deficit_probability=np.mean(u>0),total_shortfall=u.sum(),shortfall_ES99=(us[:k].sum()+(n-k)*us[k])/n,
     rmse=np.sqrt(np.mean((y-p)**2)),gini=gini_from_groups(group),top10_loss_share=np.dot(y,flag)/y.sum())
 rows.append(row)
 per[m]=np.column_stack([u,(u>0).astype(float),(y-p)**2,(2*u*u+v*v)/e])
 for X in [1,1.25,1.5,2,3,5]:
  penalties.append(dict(model=m,X=X,score=np.sum((X*u*u+v*v)/e)/e.sum()))
pd.DataFrame(rows).to_csv(OUT/'test_metrics.csv',index=False)
pd.DataFrame(penalties).to_csv(OUT/'asymmetric_penalties.csv',index=False)
rng=np.random.default_rng(9282026);draws={m:[] for m in M[1:]}
for _ in range(600):
 w=rng.poisson(1,len(y));den=np.array([w.sum(),w.sum(),w.sum(),np.dot(w,e)])
 for m in M[1:]:draws[m].append((w@(per[m]-per['mean']))/den)
ci={}
for m in draws:
 arr=np.array(draws[m]);ci[m]={}
 for j,n in enumerate(['mean_shortfall','deficit_probability','MSE','asymmetric_X2']):
  ci[m][n]=dict(delta=float(np.sum(per[m][:,j]-per['mean'][:,j])/(e.sum() if j==3 else len(e))),low=float(np.quantile(arr[:,j],.025)),high=float(np.quantile(arr[:,j],.975)))
(OUT/'bootstrap.json').write_text(json.dumps(ci,indent=2))
# Selected learning curves, own objective; valid / first valid for legibility.
s=json.loads((OUT/'summary.json').read_text());h=json.loads((OUT/'learning_histories.json').read_text())
fig,axs=plt.subplots(2,3,figsize=(13,7))
for j,m in enumerate(M):
 name=s['selected'][m]['name'];hist=h[name];metric={'mean':'rmse','quantile':'quantile','expectile':'expectile'}[m]
 for i,split in enumerate(['train','valid']):
  ax=axs[i,j];values=np.array(hist[split][metric]);ax.plot(values/values[0],label=split)
  ax.axvline(s['selected'][m]['best_iteration'],color='black',ls=':',label='Best iteration')
  ax.set(title=f'{name}: {split}',xlabel='Boosting iteration',ylabel='Objective / first iteration');ax.legend();ax.grid(alpha=.2)
fig.tight_layout();fig.savefig(OUT/'learning_curves.png',dpi=150);plt.close(fig)
# Show validation RMSE separately from each model's native training objective.
fig,axs=plt.subplots(1,3,figsize=(13,4))
for ax,m in zip(axs,M):
 name=s['selected'][m]['name'];hh=h[name];v=np.array(hh['valid']['rmse'])
 ax.plot(v,label='Validation RMSE')
 ax.axvline(s['selected'][m]['best_iteration'],color='black',ls=':',label='Selected (native loss)')
 ax.axvline(int(np.argmin(v)),color='#b65a23',ls='--',label='Lowest RMSE in run')
 ax.set(title=m.title(),xlabel='Iteration',ylabel='Severity RMSE (EUR)');ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.tight_layout();fig.savefig(OUT/'validation_rmse_curves.png',dpi=150);plt.close(fig)
# Marginal best loss over other tuned parameters at fixed anchor alpha.
hp=pd.read_csv(OUT/'hyperparameter_search.csv')
import ast
for col in ['min_child_weight','reg_lambda','subsample','colsample_bytree']:
 hp[col]=hp.parameters.map(lambda v:ast.literal_eval(v)[col])
for xcol,ycol,filename,title in [
 ('reg_lambda','min_child_weight','hyperparameter_regularization.png','Regularization search | minimum over sampling settings'),
 ('colsample_bytree','subsample','hyperparameter_sampling.png','Sampling search | minimum over regularization settings')]:
 fig,axs=plt.subplots(1,3,figsize=(13,4.2),layout='constrained')
 for ax,m in zip(axs,M):
  z=hp[hp.kind.eq(m)];z=z.assign(index_loss=100*z.best_valid_score/z.best_valid_score.min())
  mat=z.pivot_table(index=ycol,columns=xcol,values='index_loss',aggfunc='min').sort_index()
  ax.imshow(mat.to_numpy(),cmap='YlOrRd',aspect='auto',vmin=100)
  for i in range(len(mat)):
   for j in range(len(mat.columns)):ax.text(j,i,f'{mat.iloc[i,j]:.3f}',ha='center',va='center',fontsize=9)
  ax.set(title=m.title(),xlabel=xcol,ylabel=ycol,xticks=range(len(mat.columns)),xticklabels=mat.columns,yticks=range(len(mat)),yticklabels=mat.index)
 fig.suptitle(title+' | native validation loss: best=100')
 fig.savefig(OUT/filename,dpi=150,bbox_inches='tight');plt.close(fig)
# Requested pair as a standalone plot.
r=pd.read_csv(OUT/'double_lift.csv').query("comparison == 'expectile_vs_mean'")
fig,ax=plt.subplots(figsize=(8,5))
for col,label,color in [('actual_lift','Observed loss cost','#252525'),('mean_lift','Mean','#3574b5'),('expectile_lift','Expectile','#ce5b25')]:
 ax.plot(r.row+1,r[col],'o-' if col=='actual_lift' else 's--',label=label,color=color)
ax.set(title='Frequency x Severity: Expectile / Mean',xlabel='Price-ratio decile (equal exposure)',ylabel='Lift (each series / portfolio average)',xticks=range(1,11))
ax.axhline(1,color='gray',lw=.8);ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(OUT/'double_lift_expectile_mean.png',dpi=170)
print(pd.DataFrame(rows).to_string(index=False));print('Penalty index Mean=100:')
z=pd.DataFrame(penalties).pivot(index='X',columns='model',values='score');print(z.div(z['mean'],axis=0)*100)
print('Paired policy bootstrap',json.dumps(ci,indent=2))
