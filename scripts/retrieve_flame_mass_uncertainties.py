#!/usr/bin/env python3
"""Query Gaia DR3 for published FLAME mass percentiles for the primary sample.

Requires internet access and astroquery/astropy. This script does not perform the
primary selection from scratch in Gaia Archive: it uses the archived 9,974-row
matched catalogue to identify the exact 1,399 source IDs analysed in the paper.
"""
from __future__ import annotations
from pathlib import Path
import time
import pandas as pd
from astroquery.gaia import Gaia
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'raw' / 'gaia_stellar_evolution_9974.csv'
IDS = ROOT / 'data' / 'derived' / 'primary_source_ids.csv'
OUT = ROOT / 'data' / 'derived' / 'flame_mass_uncertainties_retrieved.csv'
BATCH_SIZE = 100

def main():
    ids = pd.read_csv(IDS)['source_id'].astype('int64').tolist()
    chunks = []
    for start in range(0, len(ids), BATCH_SIZE):
        batch = ids[start:start+BATCH_SIZE]
        q = f"""SELECT source_id, mass_flame, mass_flame_lower, mass_flame_upper, flags_flame
FROM gaiadr3.astrophysical_parameters WHERE source_id IN ({','.join(map(str,batch))})"""
        print(f'Querying batch {start // BATCH_SIZE + 1} ...')
        job = Gaia.launch_job_async(q, dump_to_file=False)
        chunks.append(job.get_results().to_pandas())
        time.sleep(0.5)
    out = pd.concat(chunks, ignore_index=True).drop_duplicates('source_id')
    primary = pd.read_csv(DATA)[['source_id','evolstage_flame','mass_flame']]
    merged = primary.merge(out, on='source_id', how='left', suffixes=('_local','_archive'))
    merged.to_csv(OUT, index=False)
    print(f'Saved {len(merged):,} rows to {OUT}')
    print('Missing lower:', int(merged['mass_flame_lower'].isna().sum()))
    print('Missing upper:', int(merged['mass_flame_upper'].isna().sum()))

if __name__ == '__main__':
    main()
