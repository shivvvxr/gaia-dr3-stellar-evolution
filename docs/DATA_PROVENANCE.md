# Data provenance and acquisition

## Stage 1: initial Gaia selection

The project used the ADQL query in `queries/01_initial_gaia_selection.sql` with `TOP 10000`, `random_index < 1000000`, positive parallax, parallax S/N > 10, and `phot_g_mean_mag < 15`. No `ORDER BY` was used.

The archived output is `data/raw/gaia_source_selection_top10000.csv`.

## Stage 2: Gaia astrophysical-parameter match

The 10,000 source IDs were matched to Gaia DR3 `astrophysical_parameters` for the FLAME and GSP-Phot columns used by the paper. The cleaned standalone retrieval script is `scripts/retrieve_matched_gaia_parameters.py`. The exact original notebook-level matching implementation was not present as a separate script in the supplied project archive, so this script is a transparent reconstruction of the operation rather than a claim of byte-for-byte historical identity.

The archived matched output is `data/raw/gaia_stellar_evolution_9974.csv`. It contains 9,974 rows because 26 of the 10,000 initially selected source IDs did not appear in the matched astrophysical-parameter export.

## Stage 3: uncertainty retrieval

The primary 1,399 source IDs are stored in `data/derived/primary_source_ids.csv`. `scripts/retrieve_flame_mass_uncertainties.py` queries `mass_flame`, `mass_flame_lower`, `mass_flame_upper`, and `flags_flame` for those IDs.

## Exact replication rule

For downstream replication, use the archived CSVs and the archived primary source-ID list. A fresh `TOP 10000` query is not expected to return a byte-for-byte identical source subset because the query has no ordering clause.
