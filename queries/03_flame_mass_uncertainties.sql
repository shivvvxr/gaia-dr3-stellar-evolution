-- Retrieve published FLAME mass percentiles for the 1,399 primary sources.
-- Replace SOURCE_ID_LIST with the archived IDs from data/derived/primary_source_ids.csv.
SELECT
    source_id,
    mass_flame,
    mass_flame_lower,
    mass_flame_upper,
    flags_flame
FROM gaiadr3.astrophysical_parameters
WHERE source_id IN (SOURCE_ID_LIST);
