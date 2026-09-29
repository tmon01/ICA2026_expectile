"""Post-hoc downside diagnostics; does not refit or change any tariff or loss."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=Path(__file__).resolve().parent
MODELS=['mean','expectile','quantile']

def upper_mean(x,q):
    a=np.sort(x)[::-1]; k=(1-q)*len(a); n=int(k)
    return float((a[:n].sum()+(k-n)*a[n])/k)

def run():
    df=pd.read_csv(ROOT/'test_policy_pricing_matched.csv')
    assert len(df)==133780 and df.IDpol.is_unique
    y=df.observed_period_loss.to_numpy(float); e=df.Exposure.to_numpy(float)
    rows=[]; per={}
    for m in MODELS:
        p=df[m+'_period'].to_numpy(float); r=y-p
        u=np.maximum(r,0); v=np.maximum(-r,0)
        per[m]=dict(deficit_probability=(r>0).astype(float),
                    mean_shortfall=u,second_shortfall_moment=u*u,
                    expectile_066=(.66*u*u+.34*v*v)/e,
                    pinball_0948=.948*u+.052*v)
        row=dict(model=m,contracts=len(y),deficit_contracts=int((r>0).sum()),
                 deficit_probability=float((r>0).mean()),
                 exposure_weighted_deficit_probability=float(np.dot(e,r>0)/e.sum()),
                 total_shortfall=float(u.sum()),mean_shortfall=float(u.mean()),
                 mean_shortfall_given_deficit=float(u[r>0].mean()),
                 second_shortfall_moment=float(np.mean(u*u)),
                 root_second_shortfall_moment=float(np.sqrt(np.mean(u*u))),
                 total_surplus=float(v.sum()),net_surplus=float(v.sum()-u.sum()),
                 expectile_foc_ratio_066=float(.66*u.sum()/(.34*v.sum())),
                 implied_expectile_alpha=float(v.sum()/(u.sum()+v.sum())),
                 expectile_score_066=float(np.sum(per[m]['expectile_066'])/e.sum()),
                 pinball_score_0948=float(np.sum(per[m]['pinball_0948'])/e.sum()))
        for q in [.95,.99,.995,.999]: row[f'shortfall_ES_{q}']=upper_mean(u,q)
        for a in [.5,.66,.8,.9,.95]:
            row[f'expectile_score_{a}']=float(np.sum((a*u*u+(1-a)*v*v)/e)/e.sum())
        rows.append(row)
    rng=np.random.default_rng(663948); draws={}
    for m in ['expectile','quantile']:
        draws[m]={k:[] for k in per[m]}
    for _ in range(600):
        w=rng.poisson(1,len(y)); ne=w.sum(); ee=np.dot(w,e)
        for m in draws:
            for k in draws[m]:
                den=ee if k in ['expectile_066','pinball_0948'] else ne
                draws[m][k].append(float(np.dot(w,per[m][k]-per['mean'][k])/den))
    ci={m:{k:dict(delta=float(np.sum(per[m][k]-per['mean'][k])/
                                 (e.sum() if k in ['expectile_066','pinball_0948'] else len(y))),
                    low=float(np.quantile(v,.025)),high=float(np.quantile(v,.975)))
            for k,v in d.items()} for m,d in draws.items()}
    pd.DataFrame(rows).to_csv(OUT/'tail_metrics.csv',index=False)
    (OUT/'tail_results.json').write_text(json.dumps(dict(metrics=rows,
        paired_poisson_bootstrap_vs_mean=ci,bootstrap_replicates=600,
        scope='Frozen tariffs, post-hoc exploration, no removal or capping of observed losses'),indent=2))
    print(json.dumps(dict(metrics=rows,ci=ci),indent=2))

if __name__=='__main__': run()
