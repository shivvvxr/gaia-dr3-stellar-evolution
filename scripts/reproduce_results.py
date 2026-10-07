#!/usr/bin/env python3
"""Run the canonical downstream analysis and verify the checked-in results.

By default, the expensive mass-uncertainty GMM loop is not rerun because the paper's
5,000-realization output is checked in under results/uncertainty. Pass --regenerate-
uncertainty when an end-to-end re-computation of that diagnostic is desired.
"""
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
def run(cmd):
    print('\n$', ' '.join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--regenerate-uncertainty', action='store_true')
    ap.add_argument('--full-gmm', action='store_true')
    args=ap.parse_args()
    py=sys.executable
    run([py,'scripts/run_downstream_analysis.py'])
    if args.regenerate_uncertainty:
        cmd=[py,'scripts/propagate_mass_uncertainty.py','--n-mc','5000']
        if args.full_gmm: cmd.append('--full-gmm')
        run(cmd)
    run([py,'scripts/verify_results.py'])
if __name__=='__main__': main()
