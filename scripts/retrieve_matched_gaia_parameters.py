#!/usr/bin/env python3
"""Retrieve Gaia astrophysical parameters for the archived TOP-10000 source list.

This is the cleaned standalone replacement for the matching step that was embedded
in the exploratory notebook. It queries the exact ten astrophysical columns represented
in data/raw/gaia_stellar_evolution_9974.csv and merges them with the archived source
selection. Because the original matching script was not present as a standalone file,
this script is documented as a clean reconstruction of that operation, not a byte-for-byte
copy of the original notebook code.

Requires internet access and astroquery/astropy.
"""
from __future__ import annotations
from pathlib import Path
import time
import pandas as pd
from astroquery.gaia import Gaia

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data' / 'raw' / 'gaia_source_selection_top10000.csv'
OUT = ROOT / 'data' / 'raw' / 'gaia_stellar_evolution_9974_retrieved.csv'
BATCH_SIZE = 200

ASTRO_COLS = [
    'source_id', 'teff_gspphot', 'logg_gspphot', 'mh_gspphot', 'ag_gspphot',
    'lum_flame', 'radius_flame', 'mass_flame', 'age_flame', 'evolstage_flame', 'flags_flame'
]

def main():
    src = pd.read_csv(SOURCE, dtype={'source_id':'int64'})
    ids = src.source_id.astype('int64').tolist()
    chunks=[]
    for start in range(0, len(ids), BATCH_SIZE):
        batch=ids[start:start+BATCH_SIZE]
        q = f"""SELECT source_id, teff_gspphot, logg_gspphot, mh_gspphot, ag_gspphot,
            lum_flame, radius_flame, mass_flame, age_flame, evolstage_flame, flags_flame
            FROM gaiadr3.astrophysical_parameters
            WHERE source_id IN ({','.join(map(str,batch))})"""
        print(f'Querying batch {start//BATCH_SIZE+1} ...')
        job=Gaia.launch_job_async(q, dump_to_file=False)
        chunks.append(job.get_results().to_pandas())
        time.sleep(0.5)
    ap=pd.concat(chunks, ignore_index=True).drop_duplicates('source_id')
    result=src.merge(ap, on='source_id', how='inner')
    result.to_csv(OUT, index=False)
    print(f'Retrieved matched rows: {len(result):,}')
    print(f'Saved: {OUT}')

if __name__=='__main__': main()
