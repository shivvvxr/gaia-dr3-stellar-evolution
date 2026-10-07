#!/usr/bin/env python3
"""Retrieve the original Gaia DR3 TOP 10000 selection and FLAME/GSP-Phot match.

Requires internet access and astroquery. The query intentionally has TOP 10000
without ORDER BY because that is the exact selection used by the project. For exact
analysis replication, the archived CSV and source-ID list remain authoritative.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from astroquery.gaia import Gaia
ROOT = Path(__file__).resolve().parents[1]
OUT_RAW = ROOT / 'data' / 'raw'

def main():
    query = """SELECT TOP 10000 source_id, ra, dec, parallax, parallax_error,
parallax_over_error, phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag,
bp_rp, ruwe FROM gaiadr3.gaia_source WHERE random_index < 1000000
AND parallax > 0 AND parallax_over_error > 10 AND phot_g_mean_mag < 15"""
    print('Submitting initial Gaia query ...')
    g = Gaia.launch_job_async(query, dump_to_file=False).get_results().to_pandas()
    OUT_RAW.mkdir(parents=True, exist_ok=True)
    g.to_csv(OUT_RAW / 'gaia_source_selection_top10000_retrieved.csv', index=False)
    print(f'Retrieved {len(g):,} rows')

if __name__ == '__main__':
    main()
