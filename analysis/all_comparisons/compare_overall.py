"""Held-out pricing diagnostics for the three matched freMTPL2 tariffs.

Uses only outputs_regularized/test_policy_pricing_matched.csv; no model fitting,
selection, or rescaling using the held-out outcomes.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SOURCE = Path(__file__).resolve().parents[1] / "test_policy_pricing_matched.csv"
DEST = Path(__file__).resolve().parent
NAMES = ["mean", "expectile", "quantile"]
TARGET_VALIDATION_LR = .50
BOOTSTRAP_REPS = 600
RNG_SEED = 1109


def score_groups(score, exposure, claim, period_premium):
    """Sorted, exact-score tie groups; never break identical quotes by row ID."""
    values, inverse = np.unique(score, return_inverse=True)
    return (values, inverse,
            np.bincount(inverse, weights=exposure),
            np.bincount(inverse, weights=claim),
            np.bincount(inverse, weights=period_premium))


def gini_from_groups(groups):
    _, _, e, y, _ = groups
    midpoint = (np.cumsum(e) - e/2) / e.sum()
    return float(2 * np.dot(y, midpoint) / y.sum() - 1)


def top_fraction(groups, share):
    """Per-policy fraction in highest 'share' of exposure, splitting cutoff ties."""
    _, inverse, e, _, _ = groups
    e_desc = e[::-1]
    lower = np.cumsum(e_desc) - e_desc
    allotment = np.clip(share*e.sum() - lower, 0, e_desc)
    group_fraction = np.divide(allotment, e_desc, out=np.zeros_like(e_desc), where=e_desc>0)
    return group_fraction[::-1][inverse]


def deciles(groups):
    """Allocate identical tariffs proportionally over crossed exposure deciles."""
    _, _, e, y, prem = groups
    n = 10
    start = np.cumsum(e) - e
    end = np.cumsum(e)
    total = e.sum()
    out = []
    for i in range(n):
        overlap = np.minimum(end,(i+1)*total/n) - np.maximum(start,i*total/n)
        fractions = np.clip(overlap,0,None) / e
        block_exposure = np.dot(e,fractions)
        block_loss = np.dot(y,fractions)
        block_prem = np.dot(prem,fractions)
        out.append(dict(decile=i+1,exposure=block_exposure,
                        observed_loss=block_loss, quoted_premium=block_prem,
                        observed_rate=block_loss/block_exposure,
                        tariff_rate=block_prem/block_exposure,
                        loss_ratio=block_loss/block_prem if block_prem>0 else None))
    return out


def concentration_curve(groups):
    """Share of losses caught by highest quoted annual rate; ties are pooled."""
    _, _, e, y, _ = groups
    xp = np.r_[0,np.cumsum(e[::-1])/e.sum()]
    fp = np.r_[0,np.cumsum(y[::-1])/y.sum()]
    return np.interp(np.linspace(0,1,101),xp,fp)


def compute():
    DEST.mkdir(exist_ok=True)
    df = pd.read_csv(SOURCE)
    assert df.IDpol.is_unique and df.notna().all().all()
    exposure = df.Exposure.to_numpy(dtype=float)
    loss = df.observed_period_loss.to_numpy(dtype=float)
    annual_actual = loss/exposure
    total_loss = loss.sum()
    total_exposure = exposure.sum()
    results = {}
    curves = {}
    top_flags = {}
    for name in NAMES:
        period = df[f"{name}_period"].to_numpy(dtype=float)
        annual = df[f"{name}_annual"].to_numpy(dtype=float)
        assert np.all(period>=0) and np.allclose(period,annual*exposure,atol=.01)
        groups = score_groups(annual,exposure,loss,period)
        top10 = top_fraction(groups,.10)
        top_flags[name]=top10
        curves[name]=concentration_curve(groups)
        top_loss = np.dot(loss,top10)
        top_exp = np.dot(exposure,top10)
        top_prem = np.dot(period,top10)
        zero = period==0
        error = loss-period
        rate_error=annual_actual-annual
        own_deciles=deciles(groups)
        decile_calibration_gap=sum(abs(x['observed_loss']-
                                       TARGET_VALIDATION_LR*x['quoted_premium'])
                                   for x in own_deciles)/total_loss
        portfolio_lr=total_loss/period.sum()
        decile_shape_gap=sum(abs(x['observed_loss']-
                                 portfolio_lr*x['quoted_premium'])
                             for x in own_deciles)/total_loss
        results[name]=dict(
            total_premium=float(period.sum()),
            loss_ratio=float(total_loss/period.sum()),
            annual_tariff=float(period.sum()/total_exposure),
            annual_weighted_mae=float(np.dot(exposure,np.abs(rate_error))/total_exposure),
            annual_weighted_rmse=float(np.sqrt(np.dot(exposure,rate_error**2)/total_exposure)),
            period_mae=float(np.abs(error).mean()),
            period_rmse=float(np.sqrt(np.mean(error**2))),
            annual_rmse_after_fixed_target_deload=float(np.sqrt(
                np.dot(exposure,(annual_actual-TARGET_VALIDATION_LR*annual)**2)/total_exposure)),
            exposure_weighted_gini=gini_from_groups(groups),
            top10_claim_capture=float(top_loss/total_loss),
            top10_observed_rate=float(top_loss/top_exp),
            top10_loss_ratio=float(top_loss/top_prem) if top_prem else None,
            top10_exposure=float(top_exp),
            decile_absolute_calibration_gap_fraction=float(decile_calibration_gap),
            decile_shape_gap_fraction=float(decile_shape_gap),
            zero_policy_share=float(zero.mean()),
            zero_exposure_share=float(exposure[zero].sum()/total_exposure),
            zero_claim_share=float(loss[zero].sum()/total_loss),
            zero_claim_count=int((loss[zero]>0).sum()),
            deciles=own_deciles,
        )
    # Stability with the pricing rules held fixed. Poisson(1) bootstrap is
    # the large-sample equivalent of sampling policies with replacement.
    rng=np.random.default_rng(RNG_SEED)
    differences={"expectile_minus_mean":[],"quantile_minus_mean":[],"quantile_minus_expectile":[]}
    mae_diffs={"expectile_minus_mean":[],"quantile_minus_mean":[]}
    rmse_diffs={"expectile_minus_mean":[],"quantile_minus_mean":[]}
    period_losses={k:df[f'{k}_period'].to_numpy(dtype=float) for k in NAMES}
    abs_errors={k:np.abs(loss-v) for k,v in period_losses.items()}
    squared_errors={k:(loss-v)**2 for k,v in period_losses.items()}
    for _ in range(BOOTSTRAP_REPS):
        count=rng.poisson(1,len(df))
        bootstrap_loss=np.dot(loss,count)
        if not bootstrap_loss: continue
        shares={k:np.dot(loss*top_flags[k],count)/bootstrap_loss for k in NAMES}
        differences["expectile_minus_mean"].append(shares["expectile"]-shares["mean"])
        differences["quantile_minus_mean"].append(shares["quantile"]-shares["mean"])
        differences["quantile_minus_expectile"].append(shares["quantile"]-shares["expectile"])
        for k in mae_diffs:
            left,right=k.split('_minus_')
            mae_diffs[k].append(np.dot(count,abs_errors[left]-abs_errors[right])/count.sum())
            rmse_diffs[k].append(np.sqrt(np.dot(count,squared_errors[left])/count.sum())-
                                 np.sqrt(np.dot(count,squared_errors[right])/count.sum()))
    ci={k:dict(estimate=results[k.split('_minus_')[0]]["top10_claim_capture"]-
                         results[k.split('_minus_')[1]]["top10_claim_capture"],
               low=float(np.quantile(v,.025)),high=float(np.quantile(v,.975)))
        for k,v in differences.items()}
    error_ci={}
    for label,boot,key in [('period_mae_difference',mae_diffs,'period_mae'),
                           ('period_rmse_difference',rmse_diffs,'period_rmse')]:
        error_ci[label]={k:dict(estimate=results[k.split('_minus_')[0]][key]-
                         results[k.split('_minus_')[1]][key],
                         low=float(np.quantile(v,.025)),high=float(np.quantile(v,.975)))
                         for k,v in boot.items()}
    summary=dict(source=str(SOURCE),policies=len(df),total_exposure=total_exposure,
                 total_claims=total_loss,claiming_policies=int((loss>0).sum()),
                 top10_claim_share=float(np.sort(loss)[-10:].sum()/total_loss),
                 results=results,top10_capture_difference_bootstrap=ci,
                 error_difference_bootstrap=error_ci)
    (DEST/'analysis.json').write_text(json.dumps(summary,indent=2))
    pd.DataFrame([dict(model=k,**row) for k,v in results.items() for row in v['deciles']]).to_csv(
        DEST/'exposure_deciles.csv',index=False)
    pd.DataFrame(dict(exposure_share=np.linspace(0,1,101),**curves)).to_csv(
        DEST/'loss_concentration.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(13,4.9))
    colors=dict(mean='#3274A1',expectile='#E68E27',quantile='#9A4F98')
    xx=np.linspace(0,100,101)
    for k in NAMES:
        axes[0].plot(xx,100*curves[k],label=k.capitalize(),lw=2.2,color=colors[k])
    axes[0].plot(xx,xx,ls='--',lw=1,color='grey',label='Random')
    axes[0].set(xlabel='Highest-risk exposure (%)',ylabel='Observed claims captured (%)',
                title='Loss concentration on held-out policies',xlim=(0,100),ylim=(0,100))
    axes[0].grid(alpha=.2);axes[0].legend(frameon=False,loc='lower right')
    for k in NAMES:
        d=results[k]['deciles']
        axes[1].plot([v['decile'] for v in d],[v['observed_rate'] for v in d],
                     marker='o',lw=2,label=k.capitalize(),color=colors[k])
    axes[1].set(xlabel='Tariff rank: equal-exposure decile',ylabel='Observed claim cost (EUR / year)',
                title='Observed loss by each tariff’s own rank',xticks=np.arange(1,11))
    axes[1].grid(alpha=.2);axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(DEST/'pricing_comparison.png',dpi=190)
    plt.close(fig)
    print(json.dumps({k:v for k,v in summary.items() if k!='results'},indent=2))
    for k,v in results.items():
        print(k,json.dumps({a:b for a,b in v.items() if a!='deciles'}))


if __name__=='__main__':
    compute()
