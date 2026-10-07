#!/usr/bin/env python3
"""Propagate published Gaia FLAME mass intervals through Monte Carlo draws.

This is the cleaned executable form of the uncertainty-propagation calculation
that produced the archived manuscript results. The published 16th/50th/84th
percentiles are approximated by an asymmetric Gaussian around the median.
Because that approximation can produce non-physical negative masses, simulated
masses are floored at 0.05 M_sun before the downstream diagnostics, matching the
archived calculation.

Usage:
    python scripts/propagate_mass_uncertainty.py --n-mc 5000 --full-gmm

The full GMM option is computationally expensive because a two-component GMM is
refit for every realization. The final manuscript outputs are already checked in
under results/uncertainty/.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr
from sklearn.mixture import GaussianMixture

ROOT = Path(__file__).resolve().parents[1]
INFILE = ROOT / "data" / "derived" / "flame_mass_uncertainties.csv"
OUT = ROOT / "results" / "uncertainty"
SEED = 42
DEFAULT_MC = 5000
GMM_INIT = 20
FLOOR_MASS = 0.05


def sample_asymmetric_mass(median, lower, upper, rng):
    sigma_lo = np.maximum(median - lower, 1e-8)
    sigma_hi = np.maximum(upper - median, 1e-8)
    u = rng.random(len(median))
    z = norm.ppf(np.clip(u, 1e-8, 1 - 1e-8))
    return median + np.where(u < 0.5, sigma_lo * z, sigma_hi * z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-mc', type=int, default=DEFAULT_MC)
    ap.add_argument('--full-gmm', action='store_true')
    args = ap.parse_args()
    dat = pd.read_csv(INFILE)
    req = ['source_id','evolstage_flame','mass_flame_local','mass_flame_lower','mass_flame_upper']
    missing = [c for c in req if c not in dat.columns]
    if missing:
        raise ValueError(f'Missing required columns: {missing}')
    dat = dat.dropna(subset=req).copy()
    dat = dat[dat.mass_flame_upper >= dat.mass_flame_local]
    dat = dat[dat.mass_flame_lower <= dat.mass_flame_local]
    stage = dat.evolstage_flame.to_numpy(float)
    med = dat.mass_flame_local.to_numpy(float)
    lo = dat.mass_flame_lower.to_numpy(float)
    hi = dat.mass_flame_upper.to_numpy(float)
    rng = np.random.default_rng(SEED)
    early = (stage >= 490) & (stage < 550)
    late = (stage >= 750) & (stage < 890)
    rho = np.empty(args.n_mc)
    delta = np.empty(args.n_mc)
    low_fraction = np.full(args.n_mc, np.nan)
    for i in range(args.n_mc):
        m = sample_asymmetric_mass(med, lo, hi, rng)
        m = np.maximum(m, FLOOR_MASS)
        rho[i] = spearmanr(stage, m).statistic
        delta[i] = np.median(m[early]) - np.median(m[late])
        if args.full_gmm:
            g = GaussianMixture(n_components=2, random_state=SEED + i, n_init=GMM_INIT).fit(m[:, None])
            low_idx = int(np.argmin(g.means_.ravel()))
            low_fraction[i] = g.predict_proba(m[:, None])[:, low_idx].mean()
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {
        'N': int(len(dat)), 'MC_N': int(args.n_mc), 'seed': SEED,
        'floor_mass': FLOOR_MASS,
        'rho_point': float(spearmanr(stage, med).statistic),
        'rho_mc_median': float(np.median(rho)),
        'rho_mc_q025': float(np.quantile(rho, .025)),
        'rho_mc_q975': float(np.quantile(rho, .975)),
        'delta_point': float(np.median(med[early]) - np.median(med[late])),
        'delta_mc_median': float(np.median(delta)),
        'delta_mc_q025': float(np.quantile(delta, .025)),
        'delta_mc_q975': float(np.quantile(delta, .975)),
        'gmm_low_fraction_recomputed': bool(args.full_gmm),
    }
    if args.full_gmm:
        summary.update({
            'low_fraction_mc_median': float(np.median(low_fraction)),
            'low_fraction_mc_q025': float(np.quantile(low_fraction, .025)),
            'low_fraction_mc_q975': float(np.quantile(low_fraction, .975)),
        })
    (OUT / 'mc_regenerated_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    pd.DataFrame({'rho': rho, 'delta_median': delta, 'low_mass_fraction': low_fraction}).to_csv(
        OUT / 'mc_regenerated_realizations.csv', index=False)
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
