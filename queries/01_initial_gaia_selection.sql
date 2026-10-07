-- Exact initial ADQL selection reproduced in Section 2.2 of the paper.
-- Important: TOP is intentionally used without ORDER BY because this is the
-- actual query used for the archived starting catalogue. The resulting source
-- ID list is therefore the authoritative record of the analysed subset.
SELECT TOP 10000
    source_id,
    ra,
    dec,
    parallax,
    parallax_error,
    parallax_over_error,
    phot_g_mean_mag,
    phot_bp_mean_mag,
    phot_rp_mean_mag,
    bp_rp,
    ruwe
FROM gaiadr3.gaia_source
WHERE random_index < 1000000
  AND parallax > 0
  AND parallax_over_error > 10
  AND phot_g_mean_mag < 15;
