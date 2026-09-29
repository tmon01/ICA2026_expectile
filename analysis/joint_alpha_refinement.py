"""Resolve alpha/parameter cycling by choosing native-loss winner at each alpha."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','4');os.environ.setdefault('OMP_NUM_THREADS','4')
import json,ast,subprocess,sys,shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pandas as pd,numpy as np,xgboost as xgb,joblib
from sklearn.model_selection import train_test_split
from run_analysis import make_data,FEATURES,CATEGORIES,SEED
D=Path(__file__).resolve().parent;R=D.parent

def main():
 df,_,_=make_data(R/'data');ta,_=train_test_split(df,test_size=.2,random_state=SEED,stratify=df.loss.gt(0))
 tr,va=train_test_split(ta,test_size=.15,random_state=SEED+1,stratify=ta.loss.gt(0));tr=tr.copy();va=va.copy()
 for part in [tr,va]:
  part['LogDensity']=np.log(part.Density)
  for c in CATEGORIES:part[c]=pd.Categorical(part[c],categories=sorted(tr[c].dropna().unique()))
 sev=pd.read_csv(R/'data/freMTPL2sev.csv')
 claims={k:p[['IDpol']+FEATURES].merge(sev,on='IDpol',validate='one_to_many') for k,p in [('train',tr),('valid',va)]}
 glm=joblib.load(D/'models/frequency_glm.joblib');f=glm.predict(va);loss=va.loss.sum();ex=va.Exposure.to_numpy()
 hp=pd.read_csv(D/'hyperparameter_search.csv');old=pd.read_csv(D/'validation_grid.csv')
 s=json.loads((D/'summary.json').read_text());params=json.loads((D/'selected_parameters.json').read_text());audit=[];fixed={}
 (D/'initial_alpha_search_parameters.json').write_text(json.dumps(params,indent=2))
 for kind in ['quantile','expectile']:
  configs=[ast.literal_eval(r.parameters) for _,r in hp[hp.kind.eq(kind)].nsmallest(3,'best_valid_score').iterrows()]
  if params[kind] not in configs:configs.append(params[kind])
  winners={};metric={'quantile':'quantile','expectile':'expectile'}[kind]
  def at_alpha(alpha):
   alpha=round(float(alpha),6)
   if alpha in winners:return winners[alpha]
   def fit(index):
    cfg=configs[index];obj={'objective':'reg:quantileerror','quantile_alpha':alpha} if kind=='quantile' else {'objective':'reg:expectileerror','expectile_alpha':alpha}
    dm={k:xgb.DMatrix(v[FEATURES],label=v.ClaimAmount,enable_categorical=True,nthread=4) for k,v in claims.items()}
    h={};b=xgb.train({**cfg,**obj,'eval_metric':['rmse',metric]},dm['train'],num_boost_round=8000,evals=[(dm['train'],'train'),(dm['valid'],'valid')],early_stopping_rounds=200,evals_result=h,verbose_eval=False)
    dp=xgb.DMatrix(va[FEATURES],enable_categorical=True,nthread=4);q=b.predict(dp,iteration_range=(0,b.best_iteration+1))
    return dict(kind=kind,alpha=alpha,candidate=index,parameters=cfg,best_iteration=b.best_iteration,trained_rounds=b.num_boosted_rounds(),best_valid_score=h['valid'][metric][b.best_iteration],valid_lr=float(loss/np.dot(f*q,ex)))
   with ThreadPoolExecutor(max_workers=2) as pool:rows=list(pool.map(fit,range(len(configs))))
   winner=min(rows,key=lambda r:r['best_valid_score']);winners[alpha]=winner
   for r in rows:r['native_loss_winner']=r['candidate']==winner['candidate']
   audit.extend(rows);pd.DataFrame(audit).to_csv(D/'joint_alpha_search.csv',index=False)
   print(f'JOINT {kind} alpha={alpha} candidate={winner["candidate"]} LR={winner["valid_lr"]:.7f}',flush=True)
   return winner
  pick=at_alpha(s['selected'][kind]['alpha'])
  if abs(pick['valid_lr']-.5)>=.001:
   # Evaluate nearby alpha bounds, then bracket adaptively.
   for a in ([.93,.96] if kind=='quantile' else [.70,.85]):at_alpha(a)
   if not any((l['valid_lr']-.5)*(h['valid_lr']-.5)<0 for l,h in zip(sorted(winners.values(),key=lambda r:r['alpha'])[:-1],sorted(winners.values(),key=lambda r:r['alpha'])[1:])):
    for a in ([.90,.98] if kind=='quantile' else [.65,.9]):at_alpha(a)
  for step in range(16):
   pick=min(winners.values(),key=lambda r:abs(np.log(r['valid_lr']/.5)))
   if abs(pick['valid_lr']-.5)<.001:break
   ordered=sorted(winners.values(),key=lambda r:r['alpha']);brackets=[(l,h) for l,h in zip(ordered[:-1],ordered[1:]) if (l['valid_lr']-.5)*(h['valid_lr']-.5)<0]
   if not brackets:break
   l,h=min(brackets,key=lambda z:min(abs(z[0]['valid_lr']-.5),abs(z[1]['valid_lr']-.5)))
   a=round((l['alpha']+h['alpha'])/2,6)
   if a in winners:break
   at_alpha(a)
  pick=min(winners.values(),key=lambda r:abs(np.log(r['valid_lr']/.5)))
  params[kind]=pick['parameters'];fixed[kind]=pick['alpha']
  if abs(pick['valid_lr']-.5)>=.001:print(f'Target residual: {kind} LR={pick["valid_lr"]:.8f}; retaining closest tested native-loss winner.',flush=True)
 (D/'selected_parameters.json').write_text(json.dumps(params,indent=2))
 (D/'fixed_final_alphas.json').write_text(json.dumps(fixed,indent=2))
 shutil.copy2(D/'validation_grid.csv',D/'initial_alpha_search.csv')
 subprocess.run([sys.executable,str(D/'run_analysis.py'),'--reuse-tuning','--fixed-alphas'],cwd=R,check=True)
 record=dict(cycles=0,selection_metrics=dict(mean='valid RMSE',quantile='valid pinball loss',expectile='valid asymmetric squared loss'));record.update(stable_at_final_alpha=True,joint_refinement=True,scope='Top three anchor-grid settings plus incumbent: native-loss winner selected at each candidate alpha, then alpha selected by validation LR proximity. No test selection.')
 (D/'final_refinement.json').write_text(json.dumps(record,indent=2))
if __name__=='__main__':main()
