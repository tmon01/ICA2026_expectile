"""Run from project root: python frequency_severity/plot_diagnostics.py
Uses fixed existing test quotes; no fitting, filtering, capping, or repricing.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
ROOT=Path(__file__).resolve().parents[1]
OUT=Path(__file__).resolve().parent
D=pd.read_csv(OUT/'test_policy_pricing_matched.csv')
M=['mean','expectile','quantile']; names=dict(mean='Mean',expectile='Expectile',quantile='Quantile')
colors=dict(mean='#3574b5',expectile='#ce5b25',quantile='#23917d')
e=D.Exposure.to_numpy(); y=D.observed_period_loss.to_numpy()
p={m:D[m+'_period'].to_numpy() for m in M}
a={m:D[m+'_annual'].to_numpy() for m in M}
assert len(D)==133780 and np.all(e>0)
assert np.isclose(y.sum(),10976159.65)
assert np.all(np.isfinite(np.column_stack([e,y,*p.values()])))
T=np.column_stack([e,y,*p.values()]); total=T.sum(axis=0)

def buckets(score, edges=np.linspace(0,1,11)):
    # Pool identical scores before splitting: all members of a tie receive
    # identical fractional allocation; never rank within ties using outcomes.
    _, inv=np.unique(score,return_inverse=True)
    g=np.column_stack([np.bincount(inv,weights=T[:,k]) for k in range(5)])
    right=np.cumsum(g[:,0]); left=right-g[:,0]
    rows=[]
    for lo,hi in zip(edges[:-1],edges[1:]):
        w=np.maximum(0,np.minimum(right,hi*total[0])-np.maximum(left,lo*total[0]))/g[:,0]
        v=w@g
        row={'lo':lo,'hi':hi,'exposure':v[0],'loss':v[1],'actual_rate':v[1]/v[0], 'actual_lift':(v[1]/v[0])/(total[1]/total[0])}
        for j,m in enumerate(M):
            row[m+'_premium']=v[j+2]
            row[m+'_rate']=v[j+2]/v[0]
            row[m+'_lift']=(v[j+2]/v[0])/(total[j+2]/total[0])
            row[m+'_lr']=v[1]/v[j+2] if v[j+2]>0 else np.nan
        rows.append(row)
    r=pd.DataFrame(rows)
    assert np.isclose(r.exposure.sum(),total[0]*(edges[-1]-edges[0]))
    return r

plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140})
x=np.arange(1,11)
own={m:buckets(a[m]) for m in M}
fig,axs=plt.subplots(1,3,figsize=(16,4.8),sharey=True)
for ax,m in zip(axs,M):
    r=own[m];ax.plot(x,r.actual_lift,'o-',color='#252525',label='Observed loss cost')
    ax.plot(x,r[m+'_lift'],'s--',color=colors[m],label='Quoted annual price')
    ax.axhline(1,color='gray',lw=.7);ax.set(title=names[m],xlabel='Annual-price decile (equal exposure)',xticks=x)
    ax.grid(alpha=.2);ax.legend(fontsize=9)
axs[0].set_ylabel('Lift (each series / its portfolio average)')
fig.suptitle('Frequency x Severity | Simple Quantile Plots, validation LR target 50%')
fig.tight_layout();fig.savefig(OUT/'simple_quantile.png',bbox_inches='tight');plt.close(fig)
pairs=[('expectile','mean'),('quantile','mean'),('quantile','expectile')]
double={}
fig,axs=plt.subplots(1,3,figsize=(16,4.8),sharey=True)
for ax,(c,b) in zip(axs,pairs):
    assert np.all(p[b]>0)
    r=buckets(D[c+'_severity'].to_numpy()/D[b+'_severity'].to_numpy());double[c+'_vs_'+b]=r
    ax.plot(x,r.actual_lift,'o-',color='#252525',label='Observed loss cost')
    for m in [b,c]:ax.plot(x,r[m+'_lift'],'s--',color=colors[m],label=names[m])
    ax.set(title=f'{names[c]} / {names[b]}',xlabel='Price-ratio decile (equal exposure)',xticks=x)
    ax.axhline(1,color='gray',lw=.7);ax.grid(alpha=.2);ax.legend(fontsize=9)
axs[0].set_ylabel('Lift (each series / its portfolio average)')
fig.suptitle('Frequency x Severity | Double Lift Charts')
fig.tight_layout();fig.savefig(OUT/'double_lift.png',bbox_inches='tight');plt.close(fig)
fig,axs=plt.subplots(1,3,figsize=(16,4.8))
for ax,m in zip(axs,M):
    r=own[m]; vals=r[m+'_lr']
    ax.plot(x,vals,'o-',color=colors[m]);ax.axhline(.5,color='gray',ls='--',label='Validation target 50%')
    ax.axhline(1,color='#a22',lw=.8,label='Loss equals price')
    ax.set(title=names[m],xlabel='Own annual-price decile',xticks=x,ylabel='Observed loss / quoted premium')
    ax.yaxis.set_major_formatter(PercentFormatter(1));ax.set_ylim(bottom=0);ax.grid(alpha=.2)
    if r[m+'_lr'].isna().any():
        ax.text(.04,.85,'Undefined LR in zero-price groups',transform=ax.transAxes,fontsize=10)
    ax.legend(fontsize=8,loc='center left' if m=='quantile' else 'upper right')
fig.suptitle('Loss Ratio Charts | raw prices, separate y-axis scales')
fig.tight_layout();fig.savefig(OUT/'loss_ratio.png',bbox_inches='tight');plt.close(fig)
# Rank high-risk groups using each tariff; exact same fractional tie rule.
tails=[]
for m in M:
    for f in [.10,.05,.01]:
        r=buckets(a[m],np.array([1-f,1])).iloc[0].to_dict()
        r.update(model=m,top_exposure=f,loss_share=r['loss']/total[1]);tails.append(r)
tail=pd.DataFrame(tails)
# Common reference ranking makes all three prices comparable on identical policies.
common=buckets(a['mean'],np.array([0,.9,.95,.99,1]))
common['segment']=['Bottom 90%','90–95%','95–99%','Top 1%']
fig,axs=plt.subplots(1,2,figsize=(12,4.5))
for m in M:
    r=tail[tail.model==m]
    axs[0].plot([10,5,1],r[m+'_lr'],'o-',color=colors[m],label=names[m])
    axs[1].plot(np.arange(4),common[m+'_lr'],'o-',color=colors[m],label=names[m])
axs[0].set(xlabel='Highest annual-price group (% of exposure)',xticks=[10,5,1],title='Own ranking: group members differ');axs[0].invert_xaxis()
axs[1].set(xticks=np.arange(4),xticklabels=common.segment,title='Same groups: ranked by Mean annual price')
for ax in axs:
    ax.set_ylabel('Observed loss / quoted premium');ax.axhline(.5,color='gray',ls='--');ax.axhline(1,color='#a22',lw=.8)
    ax.yaxis.set_major_formatter(PercentFormatter(1));ax.grid(alpha=.2);ax.legend()
fig.suptitle('High-risk tail diagnostics | no losses capped or removed')
fig.tight_layout();fig.savefig(OUT/'high_risk_tail.png',bbox_inches='tight');plt.close(fig)
pd.concat(own,names=['model','row']).to_csv(OUT/'simple_quantile_and_lr.csv')
pd.concat(double,names=['comparison','row']).to_csv(OUT/'double_lift.csv')
tail.to_csv(OUT/'high_risk_tail.csv',index=False);common.to_csv(OUT/'common_mean_rank_groups.csv',index=False)
print('Own tails:')
for _,r in tail.iterrows():print(r['model'],r.top_exposure,'LR',r[r['model']+'_lr'],'loss share',r.loss_share)
print('Double lift Expectile/Mean:');print(double['expectile_vs_mean'][['actual_lift','mean_lift','expectile_lift','mean_lr','expectile_lr']].to_string(index=False))
print('Common Mean rank:');print(common[['segment','mean_lr','expectile_lr','quantile_lr']].to_string(index=False))
print('Quantile zero group:',int((p['quantile']==0).sum()),'exposure share',e[p['quantile']==0].sum()/e.sum(),'loss',y[p['quantile']==0].sum())
