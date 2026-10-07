#!/usr/bin/env python3
"""Verify the checked-in repository against the manuscript's reported results."""
from __future__ import annotations
from pathlib import Path
import csv, hashlib, json, sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
try:
    import fitz
except ImportError as exc:
    raise SystemExit('PyMuPDF is required for PDF verification.') from exc

ROOT = Path(__file__).resolve().parents[1]
fail = []
def check(name, ok, detail=''):
    status = 'PASS' if ok else 'FAIL'
    print(f'[{status}] {name}' + (f' :: {detail}' if detail else ''))
    if not ok: fail.append((name, detail))

raw10 = ROOT/'data/raw/gaia_source_selection_top10000.csv'
raw = ROOT/'data/raw/gaia_stellar_evolution_9974.csv'
idsf = ROOT/'data/derived/primary_source_ids.csv'
primf = ROOT/'data/derived/primary_sample_with_diagnostics.csv'
unc = ROOT/'data/derived/flame_mass_uncertainties.csv'
out = ROOT/'results/downstream'
mc = ROOT/'results/uncertainty/mass_uncertainty_mc_summary.json'
pdf = ROOT/'paper/final_research_paper.pdf'

r10 = pd.read_csv(raw10)
r = pd.read_csv(raw)
ids = pd.read_csv(idsf)['source_id'].astype('int64')
primary = r[(r.evolstage_flame >= 490)&(r.evolstage_flame < 890)&(r.mass_flame > 0)&(r.lum_flame > 0)]
ptest = pd.read_csv(primf)

check('Initial Gaia selection rows', len(r10)==10000, str(len(r10)))
check('Matched Gaia/FLAME rows', len(r)==9974, str(len(r)))
check('Primary sample rows', len(primary)==1399, str(len(primary)))
check('Primary source ID list exact', set(ids)==set(primary.source_id.astype('int64')) and len(ids)==1399)
check('Primary diagnostic table exact IDs', set(ptest.source_id.astype('int64'))==set(ids) and len(ptest)==1399)
flags = primary['flags_flame'].astype(str).str.strip().str.replace('.0','',regex=False) if 'flags_flame' in primary else pd.Series([],dtype=str)
check('Primary FLAME flag is 10', len(flags)==1399 and set(flags)=={'10'}, str(sorted(flags.unique())[:10]))

master = json.loads((out/'00_MASTER_SUMMARY.json').read_text())
# Pull robustly from known result JSONs
assoc = json.loads((out/'02_global_association.json').read_text())
gmm = json.loads((out/'02b_gmm_model_selection.json').read_text())
low = json.loads((out/'03_low_mass_fraction_results.json').read_text())
metal = json.loads((out/'05_partial_spearman_metallicity.json').read_text())
perm = json.loads((out/'07_permutation_association_null.json').read_text())
mcj = json.loads(mc.read_text())

rho = float(assoc['rho'])
check('Global Spearman rho', abs(rho - (-0.360943)) < 5e-5, f'{rho:.6f}')
check('Global p-value', abs(float(assoc['rho_p_value']) - 2.669e-44) / 2.669e-44 < 0.02, str(assoc['rho_p_value']))
ci = assoc['rho_bootstrap']
check('Bootstrap 95% CI', abs(float(ci['q025'])+0.414455)<5e-4 and abs(float(ci['q975'])+0.305217)<5e-4, f"[{ci['q025']:.6f}, {ci['q975']:.6f}]")
check('GMM BIC selects k=3', int(gmm['bic_selected_component_count'])==3, str(gmm['bic_selected_component_count']))
lowvals = (float(low['low_component_mean_mass']), float(low['high_component_mean_mass']))
means2 = sorted(lowvals)
check('2-component means', abs(means2[0]-1.323086)<5e-4 and abs(means2[1]-2.830384)<5e-4, str(means2))
check('Low-mass fraction endpoints', abs(float(low['early_fraction'])-0.1064663)<5e-5 and abs(float(low['late_fraction'])-0.7942024)<5e-5, f"early={low['early_fraction']}, late={low['late_fraction']}")
check('Partial Spearman metallicity', abs(float(metal['rho_partial']) + 0.363573)<5e-4, str(metal['rho_partial']))
check('Permutation empirical p', abs(float(perm['two_sided_empirical_p']) - 1.9996000799840032e-4) < 1e-10, str(perm['two_sided_empirical_p']))
check('MC rho median', abs(float(mcj['rho_mc_median']) + 0.3596183) < 2e-4, str(mcj['rho_mc_median']))
check('MC rho 95% CI', abs(float(mcj['rho_mc_q025'])+0.3668884)<2e-4 and abs(float(mcj['rho_mc_q975'])+0.3508658)<2e-4, f"[{mcj['rho_mc_q025']}, {mcj['rho_mc_q975']}]")
check('MC delta median', abs(float(mcj['delta_mc_median']) - 1.4202624) < 2e-4, str(mcj['delta_mc_median']))
check('MC delta 95% CI', abs(float(mcj['delta_mc_q025'])-1.3983525)<2e-4 and abs(float(mcj['delta_mc_q975'])-1.4425746)<2e-4, f"[{mcj['delta_mc_q025']}, {mcj['delta_mc_q975']}]")

# Seed/provenance checks
code = (ROOT/'scripts/run_downstream_analysis.py').read_text()
check('Downstream uses repository-relative paths', 'parents[1]' in code and 'data/raw/gaia_stellar_evolution_9974.csv' in code)
check('No statsmodels dependency in active downstream', 'statsmodels' not in code)
uncode = (ROOT/'scripts/propagate_mass_uncertainty.py').read_text()
check('Uncertainty code matches archived floor', 'FLOOR_MASS = 0.05' in uncode and 'm = np.maximum(m, FLOOR_MASS)' in uncode)
check('Master seed is 42', 'SEED = 42' in code and 'SEED = 42' in uncode)

# PDF checks
check('Final paper exists', pdf.exists())
doc = fitz.open(pdf) if pdf.exists() else None
if doc:
    check('Final paper has 20 pages', len(doc)==20, str(len(doc)))
    text = '\n'.join(p.get_text() for p in doc)
    for phrase in ['not assigned within this manuscript','scripts should also be added','scripts are not included','10,000-resample','10,000 bootstrap']:
        check(f'No stale PDF phrase: {phrase}', phrase.lower() not in text.lower())
    for phrase in ['executable analysis code','master seed','seed 47','0.05 M_sun']:
        check(f'PDF contains updated reproducibility language: {phrase}', phrase.lower() in text.lower())
    for claim in ['1,399 stars', '−0.361', '2.67 × 10', '1.323', '2.830', '0.106', '0.794', '−0.364', '2 × 10']:
        check(f'PDF contains headline claim: {claim}', claim.lower() in text.lower())
    doc.close()

# Figure count
paper_figs = list((ROOT/'paper/figures').glob('figure_*.png'))
regen_figs = list((ROOT/'results/downstream/figures').glob('figure_*.png'))
check('Exact paper figure snapshots = 12', len(paper_figs)==12, str(len(paper_figs)))
check('Regenerated downstream figures = 12', len(regen_figs)==12, str(len(regen_figs)))
figcmp = ROOT/'docs/FIGURE_COMPARISON.json'
if figcmp.exists():
    cmp = json.loads(figcmp.read_text())
    check('Paper/regenerated figure comparison recorded', len(cmp)==12, str(len(cmp)))
    check('All figure similarities above 0.80', min(float(v) for v in cmp.values()) > 0.80, str(min(cmp.values())))
else:
    check('Paper/regenerated figure comparison exists', False)

max_file = max((p.stat().st_size for p in ROOT.rglob('*') if p.is_file() and p.name != 'SHA256SUMS.txt'), default=0)
check('No repository file exceeds 100 MB', max_file < 100*1024*1024, str(max_file))

# Checksums
sumfile = ROOT/'SHA256SUMS.txt'
if sumfile.exists():
    lines = [x.strip().split('  ',1) for x in sumfile.read_text().splitlines() if x.strip()]
    bad=[]
    for h,p in lines:
        fp=ROOT/p
        if not fp.exists() or hashlib.sha256(fp.read_bytes()).hexdigest()!=h: bad.append(p)
    check('SHA256SUMS all entries match', not bad, ', '.join(bad[:10]))
else:
    check('SHA256SUMS exists', False)

print(f'\nVerification failures: {len(fail)}')
if fail:
    for n,d in fail: print(' -', n, d)
    raise SystemExit(1)
