from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import (
    spearmanr,
    pearsonr,
    skew,
    kurtosis,
    rankdata,
)
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LinearRegression


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = ROOT / "data" / "raw" / "gaia_stellar_evolution_9974.csv"

OUT = ROOT / "results" / "downstream"
FIG = OUT / "figures"
TABLE = OUT / "tables"

FIG.mkdir(parents=True, exist_ok=True)
TABLE.mkdir(parents=True, exist_ok=True)

SEED = 42

# Keep these large enough for the manuscript-level uncertainty
# summaries, while allowing the script to remain practical on a laptop.
N_BOOT = 5000
N_PERM = 5000

# Primary evolutionary-stage selection.
# The interval is [STAGE_MIN, STAGE_MAX), so 890 is excluded.
STAGE_MIN = 490
STAGE_MAX = 890

# Early/late definitions.
EARLY_MIN = 490
EARLY_MAX = 550

LATE_MIN = 750
LATE_MAX = 890

# Non-overlapping evolutionary-stage bins.
STAGE_BINS = [
    ("490_550", 490, 550),
    ("550_650", 550, 650),
    ("650_750", 650, 750),
    ("750_890", 750, 890),
]

# GMM component counts used for model comparison.
GMM_COMPONENTS = (1, 2, 3)
GMM_FITTED_COMPONENTS_FOR_FRACTION = 2
GMM_N_INIT = 20

# Numerical safeguards.
MIN_N_FOR_CORRELATION = 3
MIN_N_FOR_PARTIAL_CORRELATION = 5
MIN_N_FOR_BOOTSTRAP = 5


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def save_json(obj, path):
    """Save JSON while handling NumPy scalar types safely."""

    def convert(value):
        if isinstance(value, dict):
            return {str(k): convert(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(v) for v in value]
        if isinstance(value, np.ndarray):
            return [convert(v) for v in value.tolist()]
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating, float)):
            return float(value) if np.isfinite(value) else None
        if isinstance(value, (np.bool_,)):
            return bool(value)
        if value is pd.NA or value is None:
            return None
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
        return value

    with open(path, "w", encoding="utf-8") as f:
        json.dump(convert(obj), f, indent=2, allow_nan=False)


def finite_pair(x, y):
    """Return finite numeric paired arrays.

    Works with pandas Series, NumPy arrays, lists, and other array-like
    inputs. This explicitly avoids calling .to_numpy() on a NumPy array.
    """

    x = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy(dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def spearman(x, y):
    x, y = finite_pair(x, y)

    if len(x) < MIN_N_FOR_CORRELATION:
        return np.nan, np.nan

    if np.unique(x).size < 2 or np.unique(y).size < 2:
        return np.nan, np.nan

    rho, p = spearmanr(x, y)

    if not np.isfinite(rho):
        return np.nan, np.nan

    return float(rho), float(p)


def pearson(x, y):
    x, y = finite_pair(x, y)

    if len(x) < MIN_N_FOR_CORRELATION:
        return np.nan, np.nan

    if np.unique(x).size < 2 or np.unique(y).size < 2:
        return np.nan, np.nan

    r, p = pearsonr(x, y)

    if not np.isfinite(r):
        return np.nan, np.nan

    return float(r), float(p)


def summarize(series):
    x = pd.to_numeric(series, errors="coerce").dropna()

    if len(x) == 0:
        return {"N": 0}

    return {
        "N": int(len(x)),
        "mean": float(x.mean()),
        "std": float(x.std(ddof=1)) if len(x) > 1 else np.nan,
        "median": float(x.median()),
        "min": float(x.min()),
        "max": float(x.max()),
        "q01": float(x.quantile(0.01)),
        "q05": float(x.quantile(0.05)),
        "q25": float(x.quantile(0.25)),
        "q75": float(x.quantile(0.75)),
        "q95": float(x.quantile(0.95)),
        "q99": float(x.quantile(0.99)),
    }


def bootstrap_rho(x, y, n_boot=N_BOOT, seed=SEED):
    x, y = finite_pair(x, y)

    observed, _ = spearman(x, y)

    if len(x) < MIN_N_FOR_BOOTSTRAP or not np.isfinite(observed):
        return {
            "observed": float(observed) if np.isfinite(observed) else np.nan,
            "median": np.nan,
            "q025": np.nan,
            "q975": np.nan,
            "valid_bootstrap_samples": 0,
        }

    rg = np.random.default_rng(seed)
    values = np.full(n_boot, np.nan, dtype=float)

    for i in range(n_boot):
        idx = rg.integers(0, len(x), len(x))
        values[i], _ = spearman(x[idx], y[idx])

    finite_values = values[np.isfinite(values)]

    if len(finite_values) == 0:
        return {
            "observed": float(observed),
            "median": np.nan,
            "q025": np.nan,
            "q975": np.nan,
            "valid_bootstrap_samples": 0,
        }

    return {
        "observed": float(observed),
        "median": float(np.median(finite_values)),
        "q025": float(np.percentile(finite_values, 2.5)),
        "q975": float(np.percentile(finite_values, 97.5)),
        "valid_bootstrap_samples": int(len(finite_values)),
    }


def bootstrap_delta(early, late, n_boot=N_BOOT, seed=SEED):
    early = pd.to_numeric(early, errors="coerce").dropna().to_numpy(dtype=float)
    late = pd.to_numeric(late, errors="coerce").dropna().to_numpy(dtype=float)

    if len(early) == 0 or len(late) == 0:
        return {
            "observed": np.nan,
            "median": np.nan,
            "q025": np.nan,
            "q975": np.nan,
            "valid_bootstrap_samples": 0,
        }

    observed = float(np.median(early) - np.median(late))

    rg = np.random.default_rng(seed)
    values = np.empty(n_boot, dtype=float)

    for i in range(n_boot):
        early_boot = rg.choice(early, size=len(early), replace=True)
        late_boot = rg.choice(late, size=len(late), replace=True)
        values[i] = np.median(early_boot) - np.median(late_boot)

    return {
        "observed": observed,
        "median": float(np.median(values)),
        "q025": float(np.percentile(values, 2.5)),
        "q975": float(np.percentile(values, 97.5)),
        "valid_bootstrap_samples": int(len(values)),
    }


def rank_partial_spearman(df, xcol, ycol, control_cols):
    columns = [xcol, ycol, *control_cols]

    sub = (
        df[columns]
        .apply(pd.to_numeric, errors="coerce")
        .dropna()
    )

    if len(sub) < MIN_N_FOR_PARTIAL_CORRELATION:
        return {
            "N": int(len(sub)),
            "rho_partial": np.nan,
            "p_value": np.nan,
        }

    x_rank = sub[xcol].rank(method="average").to_numpy(dtype=float)
    y_rank = sub[ycol].rank(method="average").to_numpy(dtype=float)

    if np.unique(x_rank).size < 2 or np.unique(y_rank).size < 2:
        return {
            "N": int(len(sub)),
            "rho_partial": np.nan,
            "p_value": np.nan,
        }

    z_rank = np.column_stack([
        sub[col].rank(method="average").to_numpy(dtype=float)
        for col in control_cols
    ])

    # Add intercept. No-control-column case is supported for completeness.
    if z_rank.ndim == 1:
        z_rank = z_rank.reshape(-1, 1)

    Z = np.column_stack([np.ones(len(sub)), z_rank])

    beta_x = np.linalg.lstsq(Z, x_rank, rcond=None)[0]
    beta_y = np.linalg.lstsq(Z, y_rank, rcond=None)[0]

    residual_x = x_rank - Z @ beta_x
    residual_y = y_rank - Z @ beta_y

    if np.unique(residual_x).size < 2 or np.unique(residual_y).size < 2:
        return {
            "N": int(len(sub)),
            "rho_partial": np.nan,
            "p_value": np.nan,
        }

    r, p = pearsonr(residual_x, residual_y)

    return {
        "N": int(len(sub)),
        "rho_partial": float(r),
        "p_value": float(p),
    }


def fit_gmm(masses, n_components, seed=SEED):
    x = (
        pd.to_numeric(pd.Series(masses), errors="coerce")
        .dropna()
        .to_numpy(dtype=float)
        .reshape(-1, 1)
    )

    if len(x) < n_components:
        raise ValueError(
            f"Cannot fit a {n_components}-component GMM to only {len(x)} values."
        )

    if np.unique(x).size < n_components:
        raise ValueError(
            f"Cannot fit a {n_components}-component GMM because the mass data "
            f"contain only {np.unique(x).size} unique values."
        )

    model = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        n_init=GMM_N_INIT,
        random_state=seed,
        reg_covar=1e-6,
    )

    model.fit(x)
    return model, x.ravel()


def gmm_density(xgrid, means, stds, weights):
    density = np.zeros_like(xgrid, dtype=float)

    for mu, sigma, weight in zip(means, stds, weights):
        sigma = max(float(sigma), 1e-8)
        density += (
            weight
            * np.exp(-0.5 * ((xgrid - mu) / sigma) ** 2)
            / (sigma * np.sqrt(2 * np.pi))
        )

    return density


def gmm_component_table(model):
    means_all = model.means_.ravel()
    order = np.argsort(means_all)

    rows = []

    # For one-dimensional full-covariance GMMs this indexing is stable
    # and explicit: [component, feature, feature].
    for rank, original_index in enumerate(order):
        mean = float(means_all[original_index])
        variance = float(model.covariances_[original_index, 0, 0])
        sigma = float(np.sqrt(max(variance, 0.0)))
        weight = float(model.weights_.ravel()[original_index])

        rows.append({
            "component_rank_by_mass": rank + 1,
            "original_component": int(original_index),
            "mean_mass": mean,
            "std_mass": sigma,
            "weight": weight,
        })

    return pd.DataFrame(rows), order


def bootstrap_fraction(values, n_boot=N_BOOT, seed=SEED):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return {
            "observed": np.nan,
            "median": np.nan,
            "q025": np.nan,
            "q975": np.nan,
            "valid_bootstrap_samples": 0,
        }

    rg = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=float)

    for i in range(n_boot):
        sample = rg.choice(values, size=len(values), replace=True)
        boot[i] = np.mean(sample)

    return {
        "observed": float(np.mean(values)),
        "median": float(np.median(boot)),
        "q025": float(np.percentile(boot, 2.5)),
        "q975": float(np.percentile(boot, 97.5)),
        "valid_bootstrap_samples": int(len(boot)),
    }


def safe_median(series):
    x = pd.to_numeric(series, errors="coerce").dropna()
    return float(x.median()) if len(x) else np.nan


def safe_mean(series):
    x = pd.to_numeric(series, errors="coerce").dropna()
    return float(x.mean()) if len(x) else np.nan


def safe_std(series):
    x = pd.to_numeric(series, errors="coerce").dropna()
    return float(x.std(ddof=1)) if len(x) > 1 else np.nan


def fmt_latex_number(value, decimals=3, scientific=False):
    """Return a finite value suitable for a LaTeX macro."""

    if value is None or not np.isfinite(value):
        return "NA"

    if scientific:
        return f"{value:.{decimals}e}"

    return f"{value:.{decimals}f}"


def permutation_spearman_null(x, y, n_perm=N_PERM, seed=SEED):
    """Fast permutation null for Spearman rho.

    After ranking x and y once, permuting y preserves the rank multiset.
    Therefore the Spearman correlation for each permutation is just the
    Pearson correlation between the fixed rank(x) and a permuted rank(y).
    """

    x, y = finite_pair(x, y)

    if len(x) < MIN_N_FOR_CORRELATION:
        return np.array([], dtype=float), np.nan, np.nan

    if np.unique(x).size < 2 or np.unique(y).size < 2:
        return np.array([], dtype=float), np.nan, np.nan

    rank_x = rankdata(x, method="average").astype(float)
    rank_y = rankdata(y, method="average").astype(float)

    centered_x = rank_x - rank_x.mean()
    centered_y = rank_y - rank_y.mean()

    denom_x = np.sqrt(np.sum(centered_x ** 2))
    denom_y = np.sqrt(np.sum(centered_y ** 2))

    observed_rho = float(np.sum(centered_x * centered_y) / (denom_x * denom_y))

    rg = np.random.default_rng(seed)
    values = np.empty(n_perm, dtype=float)

    for i in range(n_perm):
        permuted = rg.permutation(rank_y)
        values[i] = np.sum(centered_x * (permuted - permuted.mean())) / (
            denom_x * denom_y
        )

    empirical_p = (
        (np.sum(np.abs(values) >= abs(observed_rho)) + 1)
        / (n_perm + 1)
    )

    return values, observed_rho, float(empirical_p)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 72)
print("POST-FLAME-UNCERTAINTY ANALYSIS")
print("=" * 72)
print(f"Project root: {ROOT}")
print(f"Input file:   {DATA_FILE}")
print(f"Output dir:   {OUT}")

if not DATA_FILE.exists():
    raise FileNotFoundError(f"Could not find:\n{DATA_FILE}")

df = pd.read_csv(DATA_FILE)

print(f"Raw dataset rows:    {len(df):,}")
print(f"Raw dataset columns: {len(df.columns)}")


# ============================================================
# REQUIRED COLUMN CHECK AND NUMERIC NORMALIZATION
# ============================================================

required_columns = [
    "source_id",
    "parallax",
    "parallax_error",
    "parallax_over_error",
    "phot_g_mean_mag",
    "bp_rp",
    "ruwe",
    "teff_gspphot",
    "logg_gspphot",
    "mh_gspphot",
    "lum_flame",
    "radius_flame",
    "mass_flame",
    "age_flame",
    "evolstage_flame",
    "flags_flame",
]

missing = [column for column in required_columns if column not in df.columns]

if missing:
    raise ValueError(
        "The following required columns are missing:\n"
        + "\n".join(missing)
    )

numeric_columns = [
    column
    for column in required_columns
    if column not in {"source_id", "flags_flame"}
]

# flags_flame can be numeric, but retaining its original values is useful
# for provenance. Numeric conversion is not needed for this analysis.
for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")


# ============================================================
# PRIMARY SAMPLE
# ============================================================

primary_mask = (
    df["evolstage_flame"].notna()
    & df["mass_flame"].notna()
    & df["lum_flame"].notna()
    & np.isfinite(df["evolstage_flame"])
    & np.isfinite(df["mass_flame"])
    & np.isfinite(df["lum_flame"])
    & (df["mass_flame"] > 0)
    & (df["lum_flame"] > 0)
    & (df["evolstage_flame"] >= STAGE_MIN)
    & (df["evolstage_flame"] < STAGE_MAX)
)

primary = df.loc[primary_mask].copy()

print()
print("PRIMARY SAMPLE")
print("-" * 72)
print(f"N = {len(primary):,}")

if len(primary) < MIN_N_FOR_CORRELATION:
    raise ValueError(
        "Primary sample is too small for correlation analysis. "
        f"Found N={len(primary)}."
    )


# ============================================================
# 1. SAMPLE ATTRITION
# ============================================================

mask_stage = (
    df["evolstage_flame"].notna()
    & np.isfinite(df["evolstage_flame"])
    & (df["evolstage_flame"] >= STAGE_MIN)
    & (df["evolstage_flame"] < STAGE_MAX)
)

mask_mass = (
    mask_stage
    & df["mass_flame"].notna()
    & np.isfinite(df["mass_flame"])
    & (df["mass_flame"] > 0)
)

mask_luminosity = (
    mask_mass
    & df["lum_flame"].notna()
    & np.isfinite(df["lum_flame"])
    & (df["lum_flame"] > 0)
)

attrition = pd.DataFrame([
    {"selection_step": "Raw Gaia/FLAME input", "N": int(len(df))},
    {"selection_step": "Stage in [490, 890)", "N": int(mask_stage.sum())},
    {"selection_step": "Stage + positive FLAME mass", "N": int(mask_mass.sum())},
    {
        "selection_step": "Stage + positive FLAME mass + positive luminosity",
        "N": int(mask_luminosity.sum()),
    },
])

attrition.to_csv(TABLE / "01_sample_attrition.csv", index=False)


# ============================================================
# 2. PRIMARY SAMPLE SUMMARY
# ============================================================

primary_summary = {
    "mass_flame": summarize(primary["mass_flame"]),
    "lum_flame": summarize(primary["lum_flame"]),
    "teff_gspphot": summarize(primary["teff_gspphot"]),
    "logg_gspphot": summarize(primary["logg_gspphot"]),
    "mh_gspphot": summarize(primary["mh_gspphot"]),
    "evolstage_flame": summarize(primary["evolstage_flame"]),
    "parallax": summarize(primary["parallax"]),
    "ruwe": summarize(primary["ruwe"]),
}

save_json(primary_summary, OUT / "01_primary_summary.json")


# ============================================================
# 3. STAGE-BIN ANALYSIS
# ============================================================

stage_rows = []

for label, lower, upper in STAGE_BINS:
    sub = primary[
        (primary["evolstage_flame"] >= lower)
        & (primary["evolstage_flame"] < upper)
    ].copy()

    rho_bin, p_bin = spearman(
        sub["evolstage_flame"],
        sub["mass_flame"],
    )

    stage_rows.append({
        "bin": label,
        "stage_lower": lower,
        "stage_upper_exclusive": upper,
        "N": int(len(sub)),
        "median_mass": safe_median(sub["mass_flame"]),
        "mean_mass": safe_mean(sub["mass_flame"]),
        "std_mass": safe_std(sub["mass_flame"]),
        "rho_stage_mass": rho_bin,
        "p_value": p_bin,
        "median_stage": safe_median(sub["evolstage_flame"]),
        "median_logg": safe_median(sub["logg_gspphot"]),
        "median_teff": safe_median(sub["teff_gspphot"]),
        "median_luminosity": safe_median(sub["lum_flame"]),
    })

stage_bins = pd.DataFrame(stage_rows)
stage_bins.to_csv(TABLE / "02_stage_bin_statistics.csv", index=False)


# ============================================================
# FIGURE: STAGE HISTOGRAM
# ============================================================

plt.figure(figsize=(8, 5))
plt.hist(primary["evolstage_flame"].dropna(), bins=40)
plt.xlabel("FLAME evolutionary stage")
plt.ylabel("Number of stars")
plt.title("Distribution of FLAME evolutionary stage")
plt.tight_layout()
plt.savefig(FIG / "figure_11_stage_histogram.png", dpi=300)
plt.close()


# ============================================================
# FIGURE: STAGE-BIN MEDIAN MASS
# ============================================================

plt.figure(figsize=(8, 5))
plt.plot(
    np.arange(len(stage_bins)),
    stage_bins["median_mass"],
    marker="o",
)
plt.xticks(np.arange(len(stage_bins)), stage_bins["bin"])
plt.xlabel("FLAME evolutionary-stage bin")
plt.ylabel(r"Median FLAME mass ($M_\odot$)")
plt.title("Median FLAME mass across non-overlapping stage bins")
plt.tight_layout()
plt.savefig(FIG / "figure_12_stage_bin_median_mass.png", dpi=300)
plt.close()


# ============================================================
# 4. GLOBAL STAGE-MASS ASSOCIATION
# ============================================================

rho, p = spearman(
    primary["evolstage_flame"],
    primary["mass_flame"],
)

rho_bootstrap = bootstrap_rho(
    primary["evolstage_flame"],
    primary["mass_flame"],
    n_boot=N_BOOT,
    seed=SEED + 1,
)


# ============================================================
# 5. EARLY-LATE COMPARISON
# ============================================================

early = primary[
    (primary["evolstage_flame"] >= EARLY_MIN)
    & (primary["evolstage_flame"] < EARLY_MAX)
].copy()

late = primary[
    (primary["evolstage_flame"] >= LATE_MIN)
    & (primary["evolstage_flame"] < LATE_MAX)
].copy()

delta_bootstrap = bootstrap_delta(
    early["mass_flame"],
    late["mass_flame"],
    n_boot=N_BOOT,
    seed=SEED + 2,
)

global_results = {
    "N": int(len(primary)),
    "rho": rho,
    "rho_p_value": p,
    "rho_bootstrap": rho_bootstrap,
    "early_N": int(len(early)),
    "late_N": int(len(late)),
    "early_median_mass": safe_median(early["mass_flame"]),
    "late_median_mass": safe_median(late["mass_flame"]),
    "delta_early_minus_late": delta_bootstrap,
    "interpretation_note": (
        "This is an association among FLAME-derived quantities. "
        "It is not an independent measurement of stellar mass loss."
    ),
}

save_json(global_results, OUT / "02_global_association.json")


# ============================================================
# 6. GMM MASS-MIXTURE ANALYSIS
# ============================================================
#
# Important interpretation rule:
# A GMM fitted to FLAME mass estimates is a descriptive statistical
# decomposition. Its components are not automatically distinct physical
# stellar populations, and the component count should not be selected by
# interpretation alone.
#
# We therefore compare k=1,2,3 by BIC/AIC, report the best-BIC model,
# and additionally retain the requested 2-component fit as a sensitivity
# analysis for the low/high-mass posterior-fraction diagnostics.
# ============================================================

gmm_results = []
models = {}

mass_for_gmm = primary["mass_flame"].dropna().to_numpy(dtype=float)
X_mass = mass_for_gmm.reshape(-1, 1)

for k in GMM_COMPONENTS:
    model, _ = fit_gmm(
        mass_for_gmm,
        k,
        seed=SEED + k,
    )
    models[k] = model

    gmm_results.append({
        "n_components": int(k),
        "BIC": float(model.bic(X_mass)),
        "AIC": float(model.aic(X_mass)),
        "N": int(len(mass_for_gmm)),
    })

gmm_comparison = pd.DataFrame(gmm_results)
bic_1 = float(
    gmm_comparison.loc[
        gmm_comparison["n_components"] == 1,
        "BIC",
    ].iloc[0]
)
gmm_comparison["BIC_improvement_vs_1"] = bic_1 - gmm_comparison["BIC"]

gmm_comparison.to_csv(TABLE / "03_gmm_model_comparison.csv", index=False)

best_k = int(
    gmm_comparison.loc[
        gmm_comparison["BIC"].idxmin(),
        "n_components",
    ]
)

gmm = models[GMM_FITTED_COMPONENTS_FOR_FRACTION]
component_table, component_order = gmm_component_table(gmm)
component_table.to_csv(
    TABLE / "04_gmm_two_component_parameters.csv",
    index=False,
)

posterior = gmm.predict_proba(primary[["mass_flame"]].to_numpy(dtype=float))
labels = gmm.predict(primary[["mass_flame"]].to_numpy(dtype=float))

low_component = int(component_order[0])
high_component = int(component_order[-1])

primary["gmm_component"] = labels
primary["gmm_low_component_probability"] = posterior[:, low_component]
primary["gmm_high_component_probability"] = posterior[:, high_component]
primary["gmm_low_component_hard"] = (labels == low_component).astype(int)
primary["gmm_high_component_hard"] = (labels == high_component).astype(int)

low_mean = float(component_table.iloc[0]["mean_mass"])
high_mean = float(component_table.iloc[-1]["mean_mass"])
low_weight = float(component_table.iloc[0]["weight"])
high_weight = float(component_table.iloc[-1]["weight"])


after_gmm_summary = {
    "bic_selected_component_count": int(best_k),
    "two_component_analysis_component_count": int(GMM_FITTED_COMPONENTS_FOR_FRACTION),
    "two_component_is_bic_best": bool(best_k == GMM_FITTED_COMPONENTS_FOR_FRACTION),
}
save_json(after_gmm_summary, OUT / "02b_gmm_model_selection.json")


# ============================================================
# FIGURE: GMM MASS DISTRIBUTION
# ============================================================

mass_values = primary["mass_flame"].dropna().to_numpy(dtype=float)

xgrid = np.linspace(
    max(0.0, mass_values.min() - 0.2),
    mass_values.max() + 0.2,
    1000,
)

means = component_table["mean_mass"].to_numpy(dtype=float)
stds = component_table["std_mass"].to_numpy(dtype=float)
weights = component_table["weight"].to_numpy(dtype=float)

density = gmm_density(xgrid, means, stds, weights)

plt.figure(figsize=(9, 5))
plt.hist(mass_values, bins=45, density=True, alpha=0.6)
plt.plot(xgrid, density, linewidth=2)

for mean, sigma, weight in zip(means, stds, weights):
    sigma = max(float(sigma), 1e-8)
    component_density = (
        weight
        * np.exp(-0.5 * ((xgrid - mean) / sigma) ** 2)
        / (sigma * np.sqrt(2 * np.pi))
    )
    plt.plot(xgrid, component_density, linestyle="--")

plt.xlabel(r"FLAME mass ($M_\odot$)")
plt.ylabel("Density")
plt.title("Two-component Gaussian-mixture description of FLAME mass")
plt.tight_layout()
plt.savefig(FIG / "figure_13_gmm_mass_mixture.png", dpi=300)
plt.close()


# Rebuild the early/late subsets after the GMM posterior columns have
# been added to `primary`. The earlier early/late copies were created
# before these diagnostic columns existed.
early_gmm = primary[
    (primary["evolstage_flame"] >= EARLY_MIN)
    & (primary["evolstage_flame"] < EARLY_MAX)
].copy()

late_gmm = primary[
    (primary["evolstage_flame"] >= LATE_MIN)
    & (primary["evolstage_flame"] < LATE_MAX)
].copy()


# ============================================================
# 7. LOW-MASS COMPONENT FRACTION BY STAGE
# ============================================================

fraction_rows = []

for label, lower, upper in STAGE_BINS:
    sub = primary[
        (primary["evolstage_flame"] >= lower)
        & (primary["evolstage_flame"] < upper)
    ]

    fraction_rows.append({
        "bin": label,
        "stage_lower": lower,
        "stage_upper_exclusive": upper,
        "N": int(len(sub)),
        "low_component_probability_fraction": safe_mean(
            sub["gmm_low_component_probability"]
        ),
        "low_component_hard_fraction": safe_mean(
            sub["gmm_low_component_hard"]
        ),
    })

fractions = pd.DataFrame(fraction_rows)
fractions.to_csv(TABLE / "05_low_mass_fraction_by_stage.csv", index=False)


# ============================================================
# EARLY/LATE LOW-MASS FRACTION
# ============================================================

low_fraction_early = safe_mean(early_gmm["gmm_low_component_probability"])
low_fraction_late = safe_mean(late_gmm["gmm_low_component_probability"])

low_fraction_early_boot = bootstrap_fraction(
    early_gmm["gmm_low_component_probability"].to_numpy(),
    n_boot=N_BOOT,
    seed=SEED + 3,
)

low_fraction_late_boot = bootstrap_fraction(
    late_gmm["gmm_low_component_probability"].to_numpy(),
    n_boot=N_BOOT,
    seed=SEED + 4,
)

low_fraction_change = (
    low_fraction_late - low_fraction_early
    if np.isfinite(low_fraction_early) and np.isfinite(low_fraction_late)
    else np.nan
)

low_fraction_results = {
    "low_component_mean_mass": low_mean,
    "high_component_mean_mass": high_mean,
    "low_component_weight": low_weight,
    "high_component_weight": high_weight,
    "early_fraction": low_fraction_early,
    "late_fraction": low_fraction_late,
    "early_bootstrap": low_fraction_early_boot,
    "late_bootstrap": low_fraction_late_boot,
    "late_minus_early": low_fraction_change,
    "interpretation_note": (
        "Fractions are posterior averages from a two-component statistical "
        "GMM fitted to FLAME mass estimates. They are not direct estimates "
        "of a physically defined stellar population fraction."
    ),
}

save_json(low_fraction_results, OUT / "03_low_mass_fraction_results.json")


# ============================================================
# FIGURE: LOW-MASS FRACTION
# ============================================================

plt.figure(figsize=(8, 5))
x = np.arange(len(fractions))
plt.plot(
    x,
    fractions["low_component_probability_fraction"],
    marker="o",
)
plt.xticks(x, fractions["bin"])
plt.ylim(0, 1)
plt.xlabel("FLAME evolutionary-stage bin")
plt.ylabel("Mean posterior probability of low-mass GMM component")
plt.title("Low-mass component fraction across evolutionary stage")
plt.tight_layout()
plt.savefig(FIG / "figure_14_low_mass_fraction_vs_stage.png", dpi=300)
plt.close()


# ============================================================
# 8. GMM COMPONENT PHYSICAL DIAGNOSTICS
# ============================================================

component_diagnostic_rows = []

for component in sorted(primary["gmm_component"].dropna().unique()):
    component = int(component)
    sub = primary[primary["gmm_component"] == component]

    component_diagnostic_rows.append({
        "component": component,
        "N": int(len(sub)),
        "median_mass": safe_median(sub["mass_flame"]),
        "median_stage": safe_median(sub["evolstage_flame"]),
        "median_teff": safe_median(sub["teff_gspphot"]),
        "median_logg": safe_median(sub["logg_gspphot"]),
        "median_luminosity": safe_median(sub["lum_flame"]),
        "median_metallicity": safe_median(sub["mh_gspphot"]),
    })

component_diagnostics = pd.DataFrame(component_diagnostic_rows)
component_diagnostics.to_csv(
    TABLE / "06_gmm_component_physical_diagnostics.csv",
    index=False,
)


# ============================================================
# FIGURE: MASS VS STAGE, GMM COMPONENTS
# ============================================================

plt.figure(figsize=(8, 6))

for component in sorted(primary["gmm_component"].unique()):
    component = int(component)
    sub = primary[primary["gmm_component"] == component]
    plt.scatter(
        sub["evolstage_flame"],
        sub["mass_flame"],
        s=10,
        alpha=0.45,
        label=f"GMM component {component}",
    )

plt.xlabel("FLAME evolutionary stage")
plt.ylabel(r"FLAME mass ($M_\odot$)")
plt.title("Mass-stage distribution separated by GMM component")
plt.legend()
plt.tight_layout()
plt.savefig(FIG / "figure_15_mass_stage_gmm_components.png", dpi=300)
plt.close()


# ============================================================
# FIGURE: TEFF-LOGG
# ============================================================

plt.figure(figsize=(8, 6))

for component in sorted(primary["gmm_component"].unique()):
    component = int(component)
    sub = primary[primary["gmm_component"] == component].dropna(
        subset=["teff_gspphot", "logg_gspphot"]
    )

    if len(sub) == 0:
        continue

    plt.scatter(
        sub["teff_gspphot"],
        sub["logg_gspphot"],
        s=10,
        alpha=0.45,
        label=f"GMM component {component}",
    )

plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
plt.xlabel(r"$T_{\rm eff}$ (K)")
plt.ylabel(r"$\log g$")
plt.title("Teff-logg distribution by GMM mass component")
plt.legend()
plt.tight_layout()
plt.savefig(FIG / "figure_16_teff_logg_gmm_components.png", dpi=300)
plt.close()


# ============================================================
# 9. GMM COMPONENT FRACTION BY STAGE
# ============================================================

component_stage_rows = []

for label, lower, upper in STAGE_BINS:
    sub = primary[
        (primary["evolstage_flame"] >= lower)
        & (primary["evolstage_flame"] < upper)
    ]

    component_stage_rows.append({
        "bin": label,
        "N": int(len(sub)),
        "low_hard_fraction": safe_mean(sub["gmm_low_component_hard"]),
        "high_hard_fraction": safe_mean(sub["gmm_high_component_hard"]),
        "median_mass": safe_median(sub["mass_flame"]),
        "median_logg": safe_median(sub["logg_gspphot"]),
        "median_teff": safe_median(sub["teff_gspphot"]),
        "median_luminosity": safe_median(sub["lum_flame"]),
    })

pd.DataFrame(component_stage_rows).to_csv(
    TABLE / "07_component_fraction_stage_bins.csv",
    index=False,
)


# ============================================================
# 10. DISTANCE / SELECTION ANALYSIS
# ============================================================

# Naive inverse-parallax distance is used only as a descriptive selection
# diagnostic. It should not be presented as a Bayesian distance estimate.
primary["distance_pc_naive"] = np.where(
    np.isfinite(primary["parallax"]) & (primary["parallax"] > 0),
    1000.0 / primary["parallax"],
    np.nan,
)

primary["distance_kpc_naive"] = primary["distance_pc_naive"] / 1000.0

save_json(
    summarize(primary["distance_kpc_naive"]),
    OUT / "04_distance_summary.json",
)

selection_pairs = [
    ("evolstage_flame", "distance_kpc_naive"),
    ("mass_flame", "distance_kpc_naive"),
    ("evolstage_flame", "phot_g_mean_mag"),
    ("mass_flame", "phot_g_mean_mag"),
]

selection_rows = []

for xcol, ycol in selection_pairs:
    srho, sp = spearman(primary[xcol], primary[ycol])
    pr, pp = pearson(primary[xcol], primary[ycol])

    selection_rows.append({
        "x": xcol,
        "y": ycol,
        "N": int(primary[[xcol, ycol]].dropna().shape[0]),
        "spearman_rho": srho,
        "spearman_p": sp,
        "pearson_r": pr,
        "pearson_p": pp,
    })

pd.DataFrame(selection_rows).to_csv(
    TABLE / "08_selection_correlations.csv",
    index=False,
)


# ============================================================
# DISTANCE CUT SENSITIVITY
# ============================================================

distance_rows = []

for distance_limit in [0.5, 1, 2, 3, 4]:
    sub = primary[primary["distance_kpc_naive"] <= distance_limit]
    rr, pp = spearman(sub["evolstage_flame"], sub["mass_flame"])

    distance_rows.append({
        "distance_cut_kpc": distance_limit,
        "N": int(len(sub)),
        "rho_stage_mass": rr,
        "p_value": pp,
        "median_mass": safe_median(sub["mass_flame"]),
    })

distance_sensitivity = pd.DataFrame(distance_rows)
distance_sensitivity.to_csv(
    TABLE / "09_distance_cut_sensitivity.csv",
    index=False,
)


# ============================================================
# FIGURE: DISTANCE SENSITIVITY
# ============================================================

plt.figure(figsize=(8, 5))
plt.plot(
    distance_sensitivity["distance_cut_kpc"],
    distance_sensitivity["rho_stage_mass"],
    marker="o",
)
plt.axhline(0, linestyle="--", linewidth=1)
plt.xlabel("Maximum naive distance (kpc)")
plt.ylabel(r"Spearman $\rho$ (stage, mass)")
plt.title("Mass-stage correlation under distance cuts")
plt.tight_layout()
plt.savefig(FIG / "figure_17_distance_sensitivity.png", dpi=300)
plt.close()


# ============================================================
# FIGURE: STAGE VS DISTANCE
# ============================================================

stage_distance = primary.dropna(subset=["distance_kpc_naive", "evolstage_flame"])

plt.figure(figsize=(8, 5))
plt.scatter(
    stage_distance["distance_kpc_naive"],
    stage_distance["evolstage_flame"],
    s=8,
    alpha=0.35,
)
plt.xlabel("Naive inverse-parallax distance (kpc)")
plt.ylabel("FLAME evolutionary stage")
plt.title("Evolutionary stage versus naive distance")
plt.tight_layout()
plt.savefig(FIG / "figure_18_stage_distance_selection.png", dpi=300)
plt.close()


# ============================================================
# 11. QUALITY-CUT ROBUSTNESS
# ============================================================

base = primary.copy()

q01 = base["mass_flame"].quantile(0.01)
q99 = base["mass_flame"].quantile(0.99)

base["mass_1st_to_99th"] = base["mass_flame"].between(q01, q99, inclusive="both")

quality_tests = [
    ("primary", np.ones(len(base), dtype=bool)),
    ("RUWE_lt_1.4", base["ruwe"] < 1.4),
    ("RUWE_lt_1.2", base["ruwe"] < 1.2),
    ("parallax_SNR_gt_20", base["parallax_over_error"] > 20),
    (
        "RUWE_lt_1.4_and_SNR_gt_20",
        (base["ruwe"] < 1.4) & (base["parallax_over_error"] > 20),
    ),
    ("mass_1_to_5", base["mass_flame"].between(1, 5, inclusive="both")),
    ("mass_1st_to_99th_percentile", base["mass_1st_to_99th"]),
    ("G_lt_14", base["phot_g_mean_mag"] < 14),
    ("G_lt_13", base["phot_g_mean_mag"] < 13),
]

quality_rows = []

for test_name, mask in quality_tests:
    mask = np.asarray(mask, dtype=bool)
    sub = base.loc[mask].copy()
    rr, pp = spearman(sub["evolstage_flame"], sub["mass_flame"])

    quality_rows.append({
        "test": test_name,
        "N": int(len(sub)),
        "rho": rr,
        "p_value": pp,
        "median_mass": safe_median(sub["mass_flame"]),
    })

quality_sensitivity = pd.DataFrame(quality_rows)
quality_sensitivity.to_csv(
    TABLE / "10_quality_cut_sensitivity.csv",
    index=False,
)


# ============================================================
# FIGURE: QUALITY-CUT ROBUSTNESS
# ============================================================

plt.figure(figsize=(9, 5))
plt.plot(
    np.arange(len(quality_sensitivity)),
    quality_sensitivity["rho"],
    marker="o",
)
plt.axhline(rho, linestyle="--", linewidth=1)
plt.xticks(
    np.arange(len(quality_sensitivity)),
    quality_sensitivity["test"],
    rotation=45,
    ha="right",
)
plt.ylabel(r"Spearman $\rho$ (stage, mass)")
plt.title("Sensitivity of the stage-mass association to sample restrictions")
plt.tight_layout()
plt.savefig(FIG / "figure_19_quality_cut_sensitivity.png", dpi=300)
plt.close()


# ============================================================
# 12. METALLICITY ANALYSIS
# ============================================================

metallicity_pairs = [
    ("evolstage_flame", "mh_gspphot"),
    ("mass_flame", "mh_gspphot"),
    ("lum_flame", "mh_gspphot"),
]

metallicity_rows = []

for xcol, ycol in metallicity_pairs:
    rr, pp = spearman(primary[xcol], primary[ycol])
    metallicity_rows.append({
        "x": xcol,
        "y": ycol,
        "N": int(primary[[xcol, ycol]].dropna().shape[0]),
        "rho": rr,
        "p_value": pp,
    })

pd.DataFrame(metallicity_rows).to_csv(
    TABLE / "11_metallicity_correlations.csv",
    index=False,
)


# ============================================================
# PARTIAL SPEARMAN CONTROLLING FOR METALLICITY
# ============================================================

partial_metallicity = rank_partial_spearman(
    primary,
    "evolstage_flame",
    "mass_flame",
    ["mh_gspphot"],
)

save_json(
    partial_metallicity,
    OUT / "05_partial_spearman_metallicity.json",
)


# ============================================================
# METALLICITY TERTILE ANALYSIS
# ============================================================

metal = primary.dropna(subset=["mh_gspphot"]).copy()

if len(metal) >= 30 and metal["mh_gspphot"].nunique() >= 3:
    # Use integer group labels rather than hard-coded qcut labels so this
    # remains valid even if duplicate edges collapse one or more bins.
    metal["metallicity_tertile"] = pd.qcut(
        metal["mh_gspphot"],
        q=3,
        labels=False,
        duplicates="drop",
    )

    tertile_rows = []

    for group, sub in metal.groupby("metallicity_tertile", observed=True):
        if len(sub) == 0:
            continue

        rr, pp = spearman(
            sub["evolstage_flame"],
            sub["mass_flame"],
        )

        tertile_rows.append({
            "metallicity_group": int(group) + 1,
            "N": int(len(sub)),
            "median_mh": safe_median(sub["mh_gspphot"]),
            "rho_stage_mass": rr,
            "p_value": pp,
            "median_mass": safe_median(sub["mass_flame"]),
        })

    pd.DataFrame(tertile_rows).to_csv(
        TABLE / "12_metallicity_tertile_sensitivity.csv",
        index=False,
    )
else:
    pd.DataFrame([
        {
            "status": "not_run",
            "reason": (
                "Insufficient non-null metallicity values or fewer than "
                "three unique metallicity values."
            ),
        }
    ]).to_csv(
        TABLE / "12_metallicity_tertile_sensitivity.csv",
        index=False,
    )


# ============================================================
# 13. SIMPLE REGRESSION DIAGNOSTIC
# ============================================================

reg = primary.dropna(subset=["evolstage_flame", "mass_flame"]).copy()

X = reg[["evolstage_flame"]].to_numpy(dtype=float)
y = reg["mass_flame"].to_numpy(dtype=float)

linear_model = LinearRegression()
linear_model.fit(X, y)
predicted = linear_model.predict(X)
residuals = y - predicted

regression_summary = {
    "N": int(len(reg)),
    "intercept": float(linear_model.intercept_),
    "slope": float(linear_model.coef_[0]),
    "R2": float(linear_model.score(X, y)),
    "residual_skewness": float(skew(residuals, bias=False)) if len(residuals) > 2 else np.nan,
    "residual_excess_kurtosis": float(kurtosis(residuals, bias=False)) if len(residuals) > 3 else np.nan,
    "interpretation_note": (
        "Diagnostic linear fit only. It is not used as the primary inference "
        "because the stage-mass relation need not be linear."
    ),
}

save_json(regression_summary, OUT / "06_simple_linear_regression.json")


# ============================================================
# FIGURE: LINEAR REGRESSION
# ============================================================

plt.figure(figsize=(8, 5))
plt.scatter(
    reg["evolstage_flame"],
    reg["mass_flame"],
    s=8,
    alpha=0.25,
)

xx = np.linspace(
    reg["evolstage_flame"].min(),
    reg["evolstage_flame"].max(),
    200,
)

yy = linear_model.predict(xx.reshape(-1, 1))
plt.plot(xx, yy, linewidth=2)
plt.xlabel("FLAME evolutionary stage")
plt.ylabel(r"FLAME mass ($M_\odot$)")
plt.title("Simple linear regression diagnostic")
plt.tight_layout()
plt.savefig(FIG / "figure_20_stage_mass_linear_fit.png", dpi=300)
plt.close()


# ============================================================
# 14. MASS-LUMINOSITY DIAGNOSTIC
# ============================================================

mass_lum = primary[
    (primary["mass_flame"] > 0)
    & (primary["lum_flame"] > 0)
].copy()

mass_lum["log_lum_flame"] = np.log10(mass_lum["lum_flame"])

mass_lum_rho, mass_lum_p = spearman(
    mass_lum["mass_flame"],
    mass_lum["lum_flame"],
)

pd.DataFrame([
    {
        "N": int(len(mass_lum)),
        "rho_mass_luminosity": mass_lum_rho,
        "p_value": mass_lum_p,
    }
]).to_csv(
    TABLE / "13_mass_luminosity_diagnostic.csv",
    index=False,
)


# ============================================================
# FIGURE: MASS-LUMINOSITY
# ============================================================

plt.figure(figsize=(8, 5))
plt.scatter(
    mass_lum["log_lum_flame"],
    mass_lum["mass_flame"],
    s=8,
    alpha=0.25,
)
plt.xlabel(r"$\log_{10}(L/L_\odot)$ from FLAME")
plt.ylabel(r"FLAME mass ($M_\odot$)")
plt.title("Mass-luminosity diagnostic")
plt.tight_layout()
plt.savefig(FIG / "figure_21_mass_luminosity_diagnostic.png", dpi=300)
plt.close()


# ============================================================
# 15. PERMUTATION ASSOCIATION NULL
#
# IMPORTANT:
# This is NOT a physical no-mass-loss stellar-evolution model.
# It only tests whether the observed stage-mass pairing is stronger
# than expected after randomly breaking the pairing between stage and mass.
# ============================================================

permutation_values, observed_rho, empirical_p = permutation_spearman_null(
    primary["evolstage_flame"],
    primary["mass_flame"],
    n_perm=N_PERM,
    seed=SEED + 5,
)

if len(permutation_values):
    permutation_summary = {
        "N": int(len(primary)),
        "N_permutations": int(N_PERM),
        "observed_rho": float(observed_rho),
        "permutation_median": float(np.median(permutation_values)),
        "permutation_q025": float(np.percentile(permutation_values, 2.5)),
        "permutation_q975": float(np.percentile(permutation_values, 97.5)),
        "two_sided_empirical_p": float(empirical_p),
        "interpretation": (
            "Association null only; not a physical no-mass-loss "
            "stellar-evolution model."
        ),
    }
else:
    permutation_summary = {
        "N": int(len(primary)),
        "N_permutations": int(N_PERM),
        "observed_rho": np.nan,
        "permutation_median": np.nan,
        "permutation_q025": np.nan,
        "permutation_q975": np.nan,
        "two_sided_empirical_p": np.nan,
        "interpretation": (
            "Permutation analysis could not be performed because the "
            "input was insufficient or non-variable."
        ),
    }

save_json(
    permutation_summary,
    OUT / "07_permutation_association_null.json",
)


# ============================================================
# FIGURE: PERMUTATION NULL
# ============================================================

if len(permutation_values):
    plt.figure(figsize=(8, 5))
    plt.hist(permutation_values, bins=50, density=True, alpha=0.7)
    plt.axvline(
        observed_rho,
        linestyle="--",
        linewidth=2,
        label="Observed rho",
    )
    plt.xlabel(r"Permuted Spearman $\rho$")
    plt.ylabel("Density")
    plt.title("Permutation null for the stage-mass association")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG / "figure_22_permutation_null.png", dpi=300)
    plt.close()


# ============================================================
# 16. MASTER CORRELATION TABLE
# ============================================================

correlation_pairs = [
    ("evolstage_flame", "mass_flame"),
    ("evolstage_flame", "lum_flame"),
    ("evolstage_flame", "teff_gspphot"),
    ("evolstage_flame", "logg_gspphot"),
    ("evolstage_flame", "mh_gspphot"),
    ("mass_flame", "lum_flame"),
    ("mass_flame", "teff_gspphot"),
    ("mass_flame", "logg_gspphot"),
    ("mass_flame", "mh_gspphot"),
    ("mass_flame", "distance_kpc_naive"),
    ("evolstage_flame", "distance_kpc_naive"),
]

master_rows = []

for xcol, ycol in correlation_pairs:
    rr, pp = spearman(primary[xcol], primary[ycol])
    N_pair = int(primary[[xcol, ycol]].dropna().shape[0])

    master_rows.append({
        "x": xcol,
        "y": ycol,
        "N": N_pair,
        "rho": rr,
        "p_value": pp,
    })

pd.DataFrame(master_rows).to_csv(
    TABLE / "14_master_correlations.csv",
    index=False,
)


# ============================================================
# 17. SAVE ENRICHED PRIMARY SAMPLE
# ============================================================

primary.to_csv(
    TABLE / "15_primary_sample_with_diagnostics.csv",
    index=False,
)


# ============================================================
# 18. MASTER SUMMARY JSON
# ============================================================

master_summary = {
    "analysis_name": "Post-FLAME-uncertainty downstream analysis",
    "seed": SEED,
    "bootstrap_realizations": N_BOOT,
    "permutation_realizations": N_PERM,
    "N_raw": int(len(df)),
    "N_primary": int(len(primary)),
    "primary_stage_range": [STAGE_MIN, STAGE_MAX],
    "primary_stage_interval_convention": "[STAGE_MIN, STAGE_MAX)",
    "global_stage_mass_rho": rho,
    "global_stage_mass_p": p,
    "global_stage_mass_bootstrap": rho_bootstrap,
    "early_N": int(len(early)),
    "late_N": int(len(late)),
    "early_median_mass": safe_median(early["mass_flame"]),
    "late_median_mass": safe_median(late["mass_flame"]),
    "early_late_delta": delta_bootstrap,
    "gmm_bic_selected_k": int(best_k),
    "gmm_two_component_analysis_k": int(GMM_FITTED_COMPONENTS_FOR_FRACTION),
    "gmm_two_component_is_bic_best": bool(best_k == GMM_FITTED_COMPONENTS_FOR_FRACTION),
    "gmm_low_component_mean_mass": low_mean,
    "gmm_high_component_mean_mass": high_mean,
    "gmm_low_component_weight": low_weight,
    "gmm_high_component_weight": high_weight,
    "gmm_low_fraction_early": low_fraction_early,
    "gmm_low_fraction_late": low_fraction_late,
    "gmm_low_fraction_change": low_fraction_change,
    "partial_spearman_controlling_metallicity": partial_metallicity,
    "permutation_association_null": permutation_summary,
    "distance_sensitivity": distance_sensitivity.to_dict(orient="records"),
    "quality_cut_sensitivity": quality_sensitivity.to_dict(orient="records"),
    "simple_regression": regression_summary,
    "mass_luminosity_rho": mass_lum_rho,
    "mass_luminosity_p": mass_lum_p,
    "physical_null_model": {
        "performed": False,
        "description": (
            "No physical no-mass-loss stellar-evolution null was constructed. "
            "The permutation analysis is an association null only."
        ),
    },
}

save_json(master_summary, OUT / "00_MASTER_SUMMARY.json")


# ============================================================
# 19. LATEX MACROS
# ============================================================

latex_lines = [
    "% Auto-generated from results/downstream.\n",
    "% Do not edit manually.\n\n",
    f"\\newcommand{{\\PostMCPrimaryN}}{{{len(primary):,}}}\n",
    f"\\newcommand{{\\PostMCRho}}{{{fmt_latex_number(rho)}}}\n",
    f"\\newcommand{{\\PostMCRhoP}}{{{fmt_latex_number(p, scientific=True)}}}\n",
    f"\\newcommand{{\\PostMCRhoBootLow}}{{{fmt_latex_number(rho_bootstrap['q025'])}}}\n",
    f"\\newcommand{{\\PostMCRhoBootHigh}}{{{fmt_latex_number(rho_bootstrap['q975'])}}}\n",
    f"\\newcommand{{\\PostMCEarlyN}}{{{len(early):,}}}\n",
    f"\\newcommand{{\\PostMCLateN}}{{{len(late):,}}}\n",
    f"\\newcommand{{\\PostMCEarlyMedianMass}}{{{fmt_latex_number(early['mass_flame'].median() if len(early) else np.nan)}}}\n",
    f"\\newcommand{{\\PostMCLateMedianMass}}{{{fmt_latex_number(late['mass_flame'].median() if len(late) else np.nan)}}}\n",
    f"\\newcommand{{\\PostMCDeltaMass}}{{{fmt_latex_number(delta_bootstrap['observed'])}}}\n",
    f"\\newcommand{{\\PostMCDeltaMassLow}}{{{fmt_latex_number(delta_bootstrap['q025'])}}}\n",
    f"\\newcommand{{\\PostMCDeltaMassHigh}}{{{fmt_latex_number(delta_bootstrap['q975'])}}}\n",
    f"\\newcommand{{\\PostGMMlowMean}}{{{fmt_latex_number(low_mean)}}}\n",
    f"\\newcommand{{\\PostGMMhighMean}}{{{fmt_latex_number(high_mean)}}}\n",
    f"\\newcommand{{\\PostGMMlowWeight}}{{{fmt_latex_number(low_weight)}}}\n",
    f"\\newcommand{{\\PostGMMhighWeight}}{{{fmt_latex_number(high_weight)}}}\n",
    f"\\newcommand{{\\PostGMMlowFracEarly}}{{{fmt_latex_number(low_fraction_early)}}}\n",
    f"\\newcommand{{\\PostGMMlowFracLate}}{{{fmt_latex_number(low_fraction_late)}}}\n",
    f"\\newcommand{{\\PostGMMlowFracChange}}{{{fmt_latex_number(low_fraction_change)}}}\n",
    f"\\newcommand{{\\PostPartialRhoMetal}}{{{fmt_latex_number(partial_metallicity['rho_partial'])}}}\n",
    f"\\newcommand{{\\PostPartialPMeta}}{{{fmt_latex_number(partial_metallicity['p_value'], scientific=True)}}}\n",
    f"\\newcommand{{\\PostPermutationP}}{{{fmt_latex_number(empirical_p, scientific=True)}}}\n",
    f"\\newcommand{{\\PostGMMBestK}}{{{best_k}}}\n",
]

with open(OUT / "post_mc_latex_values.tex", "w", encoding="utf-8") as f:
    f.writelines(latex_lines)


# ============================================================
# 20. MANIFEST
# ============================================================

# Exclude the manifest itself while constructing the initial list, then add
# it explicitly so the saved manifest describes its own final file set.
created_files = sorted([
    path.relative_to(OUT).as_posix()
    for path in OUT.rglob("*")
    if path.is_file() and path.name != "analysis_manifest.json"
])

manifest = {
    "input_file": "data/raw/gaia_stellar_evolution_9974.csv",
    "output_directory": "results/downstream",
    "seed": SEED,
    "bootstrap_realizations": N_BOOT,
    "permutation_realizations": N_PERM,
    "starting_point": (
        "This pipeline starts after the completed FLAME mass-uncertainty "
        "Monte Carlo analysis."
    ),
    "uncertainty_analysis_status": (
        "Previously completed. The present pipeline does not rerun or overwrite it."
    ),
    "physical_null_model": (
        "Not performed. The permutation result is only an association null."
    ),
    "gmm_interpretation": (
        "The 2-component GMM is retained as a descriptive sensitivity analysis. "
        "BIC across k=1,2,3 is reported separately; a GMM component is not "
        "automatically a physical stellar population."
    ),
    "created_files": created_files + ["analysis_manifest.json"],
}

save_json(manifest, OUT / "analysis_manifest.json")


# ============================================================
# FINAL CONSOLE SUMMARY
# ============================================================

print()
print("=" * 72)
print("POST-MC ANALYSIS COMPLETE")
print("=" * 72)
print(f"Primary N: {len(primary):,}")
print(f"Stage-mass Spearman rho: {rho:.6f}")
print(
    "Stage-mass bootstrap 95% CI: "
    f"[{rho_bootstrap['q025']:.6f}, {rho_bootstrap['q975']:.6f}]"
)
print(f"Early median mass: {safe_median(early['mass_flame']):.6f} Msun")
print(f"Late median mass:  {safe_median(late['mass_flame']):.6f} Msun")
print(f"Early-late delta:  {delta_bootstrap['observed']:.6f} Msun")
print(f"GMM BIC-selected k: {best_k}")
print(f"GMM 2-component low mean:  {low_mean:.6f} Msun")
print(f"GMM 2-component high mean: {high_mean:.6f} Msun")
print(
    "Low-mass posterior fraction early: "
    f"{low_fraction_early:.6f}"
)
print(
    "Low-mass posterior fraction late:  "
    f"{low_fraction_late:.6f}"
)
print(
    "Partial rho controlling [M/H]: "
    f"{partial_metallicity['rho_partial']:.6f}"
)
print(f"Permutation empirical p: {empirical_p:.6e}")
print()
print(f"All results saved to:\n{OUT}")
print()
print("No Gaia queries were performed.")
print("The completed FLAME mass-uncertainty Monte Carlo files were not rerun or overwritten.")
print("No physical no-mass-loss stellar-evolution null was fabricated.")
print("The 2-component GMM is treated as descriptive unless BIC supports k=2.")
print("=" * 72)
