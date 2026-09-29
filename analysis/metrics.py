"""Exposure-weighted ranking helpers; ties are pooled before allocation."""
import numpy as np

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

