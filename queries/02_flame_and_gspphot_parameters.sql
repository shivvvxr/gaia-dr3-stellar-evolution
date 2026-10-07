-- Clean reconstruction of the astrophysical-parameter match used to create the 9,974-row table.
-- Replace SOURCE_ID_LIST with the comma-separated IDs from the initial Gaia selection.
SELECT
    source_id,
    teff_gspphot,
    logg_gspphot,
    mh_gspphot,
    ag_gspphot,
    lum_flame,
    radius_flame,
    mass_flame,
    age_flame,
    evolstage_flame,
    flags_flame
FROM gaiadr3.astrophysical_parameters
WHERE source_id IN (SOURCE_ID_LIST);
