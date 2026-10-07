# Reproducibility protocol

## Exact inputs

1. `data/raw/gaia_source_selection_top10000.csv` contains the 10,000-row starting Gaia export.
2. `data/raw/gaia_stellar_evolution_9974.csv` contains the 9,974 matched rows used for analysis.
3. `data/derived/primary_source_ids.csv` contains the exact 1,399 analysed source IDs.
4. `data/derived/flame_mass_uncertainties.csv` contains the published FLAME mass intervals used for uncertainty propagation.

## Primary selection

The final selection is:

```text
490 <= evolstage_flame < 890
mass_flame > 0
lum_flame > 0
```

## Randomness

Master seed: `42`. The final downstream code derives deterministic offsets from the master seed: stage-mass bootstrap `43`, early-late bootstrap `44`, two-/three-component GMM fits use `43`, `44`, `45`, component-fraction bootstrap uses `45`, `46`, permutation uses `47`. The mass-uncertainty Monte Carlo uses `42`; its GMM realization i uses `42 + i`.

## Uncertainty model

For published median `M`, lower `M-`, and upper `M+` values, the asymmetric Gaussian scales are `sigma_low = M - M-` and `sigma_high = M+ - M`. A standard normal deviate is transformed with the lower scale for negative deviations and upper scale for positive deviations. Simulated masses below 0.05 M_sun are floored at 0.05 M_sun before diagnostics, matching the archived final uncertainty calculation.

## Verification

`python scripts/verify_results.py` checks row counts, source-ID identity, principal numerical results, PDF status, figure counts, active-script portability, and `SHA256SUMS.txt`.

## Clean-environment test

The canonical downstream code was executed successfully in the audit environment with Python 3.13.5 and the dependencies in `requirements.txt`; `statsmodels` is not required by the canonical downstream script. The repository also includes a GitHub Actions verification workflow that installs the declared requirements and runs `verify_results.py`.

Because external Gaia querying requires live network access, the retrieval scripts are separated from the archived analysis inputs.
