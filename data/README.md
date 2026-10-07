# Data provenance

`raw/gaia_source_selection_top10000.csv` is the archived 10,000-row Gaia source export produced by the project's initial ADQL query.

`raw/gaia_stellar_evolution_9974.csv` is the 9,974-row matched Gaia/FLAME/GSP-Phot table used by the final downstream analysis.

`derived/primary_source_ids.csv` contains the authoritative 1,399 source IDs satisfying the final sample criteria. Because the initial ADQL query uses `TOP 10000` without `ORDER BY`, the archived source-ID list is necessary for exact downstream replication.

`derived/primary_sample_with_diagnostics.csv` is the final 1,399-row diagnostic table used to generate the manuscript results.

`derived/flame_mass_uncertainties.csv` stores the published FLAME 16th/50th/84th percentile mass values retrieved for the primary sample.

The Gaia-derived raw material remains subject to ESA/Gaia/DPAC terms and citation requirements. See the repository-level `LICENSE-DATA.md`.
