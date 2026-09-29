"""Run from any directory: python analysis/run_analysis.py
Dependencies: numpy pandas scipy scikit-learn>=1.8 xgboost==3.3.0 matplotlib joblib.
Input: ../data/freMTPL2freq.csv and freMTPL2sev.csv (downloaded if absent).
No recorded outcome is capped or trimmed. Split at policy level before claim expansion.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder,SplineTransformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_poisson_deviance
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from urllib.request import urlretrieve
SEED = 20260928
FEATURES = ["VehPower", "VehAge", "DrivAge", "BonusMalus", "VehBrand",
            "VehGas", "Area", "Density", "Region"]
CATEGORIES = ["VehBrand", "VehGas", "Area", "Region"]



def make_data(data_dir):
    data_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("freq", "sev"):
        path = data_dir / f"freMTPL2{suffix}.csv"
        if not path.exists():
            urlretrieve(f"https://huggingface.co/datasets/mabilton/fremtpl2/resolve/main/freMTPL2{suffix}.csv",
                        path)
    freq = pd.read_csv(data_dir / "freMTPL2freq.csv")
    sev = pd.read_csv(data_dir / "freMTPL2sev.csv")
    assert freq.IDpol.is_unique and freq.Exposure.gt(0).all()
    by_policy = sev.groupby("IDpol",sort=False).ClaimAmount.agg(["sum","size"])
    df = freq.join(by_policy,on="IDpol")
    missing_label = df.ClaimNb.gt(0) & df["sum"].isna()
    disagreements = df["size"].notna() & df.ClaimNb.ne(df["size"])
    missing_ids = df.loc[missing_label,"IDpol"].copy()
    # Only missing outcomes are unfit for supervised learning. This preserves
    # every recorded loss, including count disagreements and extreme claims.
    df = df.loc[~missing_label].copy()
    df["loss"] = df["sum"].fillna(0.)
    df["annual_loss"] = df.loss/df.Exposure
    quality = dict(freq_policies=len(freq),claim_rows=len(sev),
                   claim_rows_without_policy=int((~sev.IDpol.isin(freq.IDpol)).sum()),
                   missing_label_policies=int(missing_label.sum()),
                   count_disagreements_retained=int(disagreements.sum()),
                   matched_recorded_claims_excluded=0,
                   largest_recorded_claim=float(sev.ClaimAmount.max()),
                   kept_policies=len(df),
                   no_target_capping=True,no_target_winsorization=True)
    return df, quality, missing_ids

OUT=Path(__file__).resolve().parent
(OUT/'models').mkdir(exist_ok=True)

def main():
    df,quality,missing=make_data(ROOT/'data')
    trall,te=train_test_split(df,test_size=.2,random_state=SEED,stratify=df.loss.gt(0))
    tr,va=train_test_split(trall,test_size=.15,random_state=SEED+1,stratify=trall.loss.gt(0))
    parts={'train':tr.copy(),'valid':va.copy(),'test':te.copy()}
    for k,part in parts.items():
        part['LogDensity']=np.log(part.Density)
        for col in CATEGORIES:
            part[col]=pd.Categorical(part[col],categories=sorted(tr[col].dropna().unique()))
    # An unpenalized Poisson GLM with log link; exposure-weighted rate likelihood
    # equals count likelihood with log(Exposure) offset, up to a constant.
    prep=ColumnTransformer([
        ('numeric',StandardScaler(),['VehPower','BonusMalus','LogDensity']),
        ('age_splines',SplineTransformer(n_knots=5,degree=3,include_bias=False,extrapolation='linear'),['VehAge','DrivAge']),
        ('category',OneHotEncoder(drop='first',handle_unknown='ignore',sparse_output=False),CATEGORIES)],remainder='drop')
    glm=make_pipeline(prep,PoissonRegressor(alpha=0,solver='newton-cholesky',max_iter=100,tol=1e-8))
    print('Fitting Poisson GLM',flush=True)
    glm.fit(parts['train'],parts['train'].ClaimNb/parts['train'].Exposure,
            poissonregressor__sample_weight=parts['train'].Exposure)
    joblib.dump(glm,OUT/'models/frequency_glm.joblib')
    freq={k:glm.predict(part) for k,part in parts.items()}
    freq_stats={k:dict(observed_claims=int(part.ClaimNb.sum()),predicted_claims=float(np.dot(freq[k],part.Exposure)),
        poisson_deviance=float(mean_poisson_deviance(part.ClaimNb/part.Exposure,freq[k],sample_weight=part.Exposure))) for k,part in parts.items()}
    sev=pd.read_csv(ROOT/'data/freMTPL2sev.csv')
    claims={k:part[['IDpol']+FEATURES].merge(sev,on='IDpol',how='inner',validate='one_to_many') for k,part in parts.items()}
    assert all(c.ClaimAmount.gt(0).all() for c in claims.values())
    dm={k:xgb.DMatrix(c[FEATURES],label=c.ClaimAmount,enable_categorical=True) for k,c in claims.items()}
    dp={k:xgb.DMatrix(part[FEATURES],enable_categorical=True) for k,part in parts.items()}
    common=dict(tree_method='hist',max_depth=1,eta=.01,min_child_weight=100,reg_lambda=100,nthread=4,seed=SEED,subsample=1,colsample_bytree=1)
    # Hyperparameter selection uses severity validation losses only, at fixed
    # alpha anchors near the initial loading target. Test results never enter it.
    anchors={'mean':None,'quantile':.936094,'expectile':.766797}
    # All 81 joint combinations; depth and alpha are fixed.
    settings=[dict(label=f'm{child}_l{l2}_s{sub}_c{col}',max_depth=1,
                   min_child_weight=child,reg_lambda=l2,subsample=sub,colsample_bytree=col)
              for child in [50,200,500] for l2 in [100,1000,10000]
              for sub in [.6,.8,1.] for col in [.6,.8,1.]]
    hp_rows=[];selected_parameters={}
    if '--reuse-tuning' in sys.argv:
        selected_parameters=json.loads((OUT/'selected_parameters.json').read_text())
    else:
        for kind,alpha in anchors.items():
            objective={'objective':'reg:squarederror'} if kind=='mean' else ({'objective':'reg:quantileerror','quantile_alpha':alpha} if kind=='quantile' else {'objective':'reg:expectileerror','expectile_alpha':alpha})
            family=[]
            from concurrent.futures import ThreadPoolExecutor
            import ast
            cache=[]
            def screen_one(original_cfg):
                cfg=original_cfg.copy();label=cfg.pop('label')
                params={**common,**cfg,'reg_alpha':0}
                for r in cache:
                    same_alpha=(alpha is None and pd.isna(r['anchor_alpha'])) or (alpha is not None and r['anchor_alpha']==alpha)
                    if r['kind']==kind and same_alpha and r['setting']==label:
                        cached=ast.literal_eval(r['parameters']) if isinstance(r['parameters'],str) else r['parameters']
                        if cached==params:
                            r=r.copy();r['parameters']=cached;r['anchor_alpha']=alpha
                            return r
                metric_name={'mean':'rmse','quantile':'quantile','expectile':'expectile'}[kind]
                eval_metrics=['rmse'] if kind=='mean' else ['rmse',metric_name]
                hp_params={**params,**objective,'eval_metric':eval_metrics}
                # Separate DMatrix objects per worker; only immutable frames shared.
                local={k:xgb.DMatrix(claims[k][FEATURES],label=claims[k].ClaimAmount,
                       enable_categorical=True,nthread=4) for k in ['train','valid']}
                hh={}
                booster=xgb.train(hp_params,local['train'],num_boost_round=1500,
                    evals=[(local['train'],'train'),(local['valid'],'valid')],early_stopping_rounds=200,
                    evals_result=hh,verbose_eval=False)
                if booster.best_iteration>=1300:
                    hh={}
                    booster=xgb.train(hp_params,local['train'],num_boost_round=6000,
                        evals=[(local['train'],'train'),(local['valid'],'valid')],early_stopping_rounds=200,
                        evals_result=hh,verbose_eval=False)
                best=booster.best_iteration;metric=metric_name
                return dict(kind=kind,anchor_alpha=alpha,setting=label,best_iteration=best,
                     best_valid_score=hh['valid'][metric][best],first_valid_score=hh['valid'][metric][0],
                     first_train_score=hh['train'][metric][0],best_round_train_score=hh['train'][metric][best],
                     trained_rounds=booster.num_boosted_rounds(),parameters=params,
                     valid_rmse_at_best=float(hh['valid']['rmse'][best]),
                     best_valid_rmse=float(min(hh['valid']['rmse'])),
                     best_rmse_iteration=int(np.argmin(hh['valid']['rmse'])))
            with ThreadPoolExecutor(max_workers=2) as pool:
                # map preserves candidate order for deterministic tie-breaking.
                for row in pool.map(screen_one,settings):
                    family.append(row);hp_rows.append(row)
                    print(f'TUNE {kind} {row["setting"]}: best={row["best_iteration"]}, valid={row["best_valid_score"]:.6f}',flush=True)
                    pd.DataFrame(hp_rows).to_csv(OUT/'hyperparameter_search.csv',index=False)
            selected_parameters[kind]=min(family,key=lambda r:r['best_valid_score'])['parameters']
        rmse_choices={kind:min([r for r in hp_rows if r['kind']==kind],key=lambda r:r['best_valid_rmse']) for kind in anchors}
        (OUT/'rmse_optimal_settings.json').write_text(json.dumps(rmse_choices,indent=2))
        (OUT/'selected_parameters.json').write_text(json.dumps(selected_parameters,indent=2))
    rows=[];models={};predictions={};histories={}
    cached_histories=json.loads((OUT/'learning_histories.json').read_text()) if '--reuse-models' in sys.argv else {}
    def fit(kind,alpha=None):
        name=kind if alpha is None else f'{kind}_{alpha:.6f}'
        if name in models:return next(r for r in rows if r['name']==name)
        obj={'objective':'reg:squarederror'} if kind=='mean' else ({'objective':'reg:quantileerror','quantile_alpha':alpha} if kind=='quantile' else {'objective':'reg:expectileerror','expectile_alpha':alpha})
        metric_name={'mean':'rmse','quantile':'quantile','expectile':'expectile'}[kind]
        obj['eval_metric']=['rmse'] if kind=='mean' else ['rmse',metric_name]
        hist={}
        if name in cached_histories:
            booster=xgb.Booster();booster.load_model(OUT/'models'/f'{name}.json');hist=cached_histories[name]
        else:
            booster=xgb.train({**selected_parameters[kind],**obj},dm['train'],num_boost_round=8000,
                evals=[(dm['train'],'train'),(dm['valid'],'valid')],early_stopping_rounds=200,evals_result=hist,verbose_eval=False)
        best=booster.best_iteration
        # Candidate selection needs validation prices only; defer train/test inference.
        predictions[name]={'valid':booster.predict(dp['valid'],iteration_range=(0,best+1)).astype(float)}
        # Never silently clip predictions: inspect and fail on nonpositive severity.
        assert all(np.all(v>0) and np.all(np.isfinite(v)) for v in predictions[name].values()),name
        prem={'valid':float(np.dot(freq['valid']*predictions[name]['valid'],parts['valid'].Exposure))}
        metric=metric_name
        row=dict(name=name,kind=kind,alpha=alpha,best_iteration=best,trained_rounds=booster.num_boosted_rounds(),
            valid_lr=float(parts['valid'].loss.sum()/prem['valid']),
            validation_metric=metric,first_valid_score=hist['valid'][metric][0],best_valid_score=hist['valid'][metric][best],
            first_train_score=hist['train'][metric][0],best_round_train_score=hist['train'][metric][best],valid_rmse_at_best=float(hist['valid']['rmse'][best]))
        rows.append(row);models[name]=booster;histories[name]=hist
        booster.save_model(OUT/'models'/f'{name}.json')
        print(f'{name}: best={best}, valid LR={row["valid_lr"]:.5f}',flush=True)
        return row
    fit('mean')
    grids={'quantile':[.5,.6,.7,.75,.8,.85,.9,.93,.96,.99], 'expectile':[.5,.55,.6,.65,.7,.75,.8,.85,.9,.95]}
    chosen={}
    fixed_alphas=json.loads((OUT/'fixed_final_alphas.json').read_text()) if '--fixed-alphas' in sys.argv else {}
    for kind,grid in grids.items():
        if kind in fixed_alphas:
            chosen[kind]=fit(kind,fixed_alphas[kind])['name']
            continue
        for alpha in grid:fit(kind,alpha)
        # Validation-only refinement, if a grid bracket exists. Non-monotonicity
        # is possible because each alpha fits a separate tree ensemble.
        for step in range(16):
            rr=sorted([r for r in rows if r['kind']==kind],key=lambda r:r['alpha'])
            pick=min(rr,key=lambda r:abs(np.log(r['valid_lr']/.5)))
            if abs(pick['valid_lr']-.5)<.001:break
            brackets=[(l,h) for l,h in zip(rr[:-1],rr[1:]) if (l['valid_lr']-.5)*(h['valid_lr']-.5)<0]
            if not brackets:break
            l,h=min(brackets,key=lambda z:min(abs(z[0]['valid_lr']-.5),abs(z[1]['valid_lr']-.5)))
            alpha=round((l['alpha']+h['alpha'])/2,6)
            if alpha in [r['alpha'] for r in rr]:break
            fit(kind,alpha)
        chosen[kind]=min([r for r in rows if r['kind']==kind],key=lambda r:abs(np.log(r['valid_lr']/.5)))['name']
    chosen['mean']='mean'
    for name in chosen.values():
        best=models[name].best_iteration
        for k in ['train','test']:
            predictions[name][k]=models[name].predict(dp[k],iteration_range=(0,best+1)).astype(float)
            assert np.isfinite(predictions[name][k]).all() and (predictions[name][k]>0).all()
    target_prem=np.mean([np.dot(freq['valid']*predictions[chosen[k]]['valid'],parts['valid'].Exposure) for k in ['quantile','expectile']])
    factor=target_prem/np.dot(freq['valid']*predictions['mean']['valid'],parts['valid'].Exposure)
    result={}
    prices=parts['test'][['IDpol','Exposure','ClaimNb','loss']].rename(columns={'loss':'observed_period_loss'}).copy()
    prices['frequency_annual']=freq['test']
    for kind,name in chosen.items():
        scale=factor if kind=='mean' else 1.
        stats={}
        for k,part in parts.items():
            q=freq[k]*predictions[name][k]*part.Exposure.to_numpy()*scale
            y=part.loss.to_numpy();u=np.maximum(y-q,0);v=np.maximum(q-y,0)
            stats[k]=dict(loss_ratio=float(y.sum()/q.sum()),premium_total=float(q.sum()),
                zero_prices=int((q==0).sum()),mean_shortfall=float(u.mean()),total_shortfall=float(u.sum()),
                deficit_probability=float((y>q).mean()),period_rmse=float(np.sqrt(np.mean((y-q)**2))),
                annual_weighted_rmse=float(np.sqrt(np.sum((y-q)**2/part.Exposure)/part.Exposure.sum())))
        result[kind]=dict(name=name,alpha=next(r['alpha'] for r in rows if r['name']==name),multiplier=float(scale),best_iteration=models[name].best_iteration,**stats)
        prices[kind+'_severity']=predictions[name]['test']
        prices[kind+'_annual']=freq['test']*predictions[name]['test']*scale
        prices[kind+'_period']=prices[kind+'_annual']*prices.Exposure
    temporary=OUT/'prices.partial.csv'
    prices.sort_values('IDpol').to_csv(temporary,index=False)
    check=pd.read_csv(temporary)
    assert len(check)==len(te) and check.IDpol.is_unique and np.isclose(check.observed_period_loss.sum(),te.loss.sum())
    temporary.replace(OUT/'test_policy_pricing_matched.csv')
    pd.DataFrame(rows).to_csv(OUT/'validation_grid.csv',index=False)
    split_table=pd.concat([p[['IDpol']].assign(split=k) for k,p in parts.items()])
    split_table.to_csv(OUT/'split_ids.partial.csv',index=False)
    assert len(pd.read_csv(OUT/'split_ids.partial.csv'))==len(df)
    (OUT/'split_ids.partial.csv').replace(OUT/'split_ids.csv')
    summary=dict(quality=quality,split={k:len(p) for k,p in parts.items()},claims={k:len(c) for k,c in claims.items()},
       frequency=freq_stats,glm_iterations=int(glm[-1].n_iter_),severity_parameters=selected_parameters,hyperparameter_anchor_alphas=anchors,
       initial_alpha_grids=grids,target_validation_lr=.5,alpha_policy="Select alpha using validation only; no test rescaling",selected=result,prediction_uses_best_iteration=True,
       no_loss_capping=True,no_prediction_clipping=True,seed=SEED,xgboost_version=xgb.__version__)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    (OUT/'learning_histories.json').write_text(json.dumps(histories))
    print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
