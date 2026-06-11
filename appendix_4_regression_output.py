"""
Appendix 4 -- Full regression output (H1, H2, first difference)
===============================================================

This script reproduces the full statistical output reported in Appendix 4 of
the thesis. For every model it reports the coefficient, the robust standard
error, the z-statistic and the exact p-value, plus R-squared, F and N. Because
all models use HC1 heteroskedasticity-robust standard errors, statsmodels
reports a z-statistic (shown here as "z").

This is the code behind the main results: H1 (Section 4.3), H2 (Section 4.4)
and the first-difference models (Section 4.5).

It is part of the replication code for the master's thesis:
  "Does It Matter Where FDI Comes From? Inward FDI, Source-Country
   Concentration, and Entrepreneurial Ecosystem Quality across African
   Countries" (K.W. Vonk, MSc International Management, Utrecht University).

WHAT IT READS
-------------
  * aeei1_dataset.csv            (wave 1, FDI 2019)
  * aeei2_dataset.csv            (wave 2, FDI 2022)
  * first_difference_dataset.csv (27 countries in both waves)
Inward FDI and HHI are from IMF CDIS.

WHAT IT WRITES (next to this script)
------------------------------------
  * appendix_4_full_output.csv

HOW TO RUN
----------
  python appendix_4_regression_output.py
Set DATA_DIR below to the folder that holds the CSV files.
"""

import os
import pandas as pd
import statsmodels.api as sm


# ---------------------------------------------------------------------------
# 0. Where the input data lives, and where to write the output.
# ---------------------------------------------------------------------------
DATA_DIR = r"C:\Thesis-IMI-AEEI-2026\thesis_data\imf_cdis\analysis_ready"
OUT_DIR  = os.path.dirname(os.path.abspath(__file__))

# Readable names for the table rows.
NICE = {
    "const": "Constant", "ln_fdi_gdp": "ln(FDI / GDP)",
    "hhi": "Source concentration (HHI)", "resource_exports_pct": "Resource exports (%)",
    "fdi_c": "ln(FDI / GDP)", "hhi_c": "Source concentration (HHI)",
    "int_c": "ln(FDI / GDP) x HHI",
    "delta_ln_fdi_gdp": "Change in ln(FDI / GDP)", "delta_hhi": "Change in HHI",
    "delta_fdi_hhi": "Change in ln(FDI / GDP) x HHI",
    "delta_resource_exports_pct": "Change in resource exports (%)",
}


# ---------------------------------------------------------------------------
# 1. The three model specifications (identical to the main analysis).
# ---------------------------------------------------------------------------
def fit_h1(name):
    """H1 baseline: AEEI ~ ln(FDI/GDP) + HHI + resource."""
    d = pd.read_csv(os.path.join(DATA_DIR, name))[["aeei", "ln_fdi_gdp", "hhi", "resource_exports_pct"]]
    d = d.apply(pd.to_numeric, errors="coerce").dropna()
    return sm.OLS(d["aeei"], sm.add_constant(d[["ln_fdi_gdp", "hhi", "resource_exports_pct"]])).fit(cov_type="HC1")


def fit_h2(name):
    """H2 interaction model with mean-centered FDI and HHI."""
    d = pd.read_csv(os.path.join(DATA_DIR, name))[["aeei", "ln_fdi_gdp", "hhi", "resource_exports_pct"]]
    d = d.apply(pd.to_numeric, errors="coerce").dropna().copy()
    d["fdi_c"] = d["ln_fdi_gdp"] - d["ln_fdi_gdp"].mean()
    d["hhi_c"] = d["hhi"] - d["hhi"].mean()
    d["int_c"] = d["fdi_c"] * d["hhi_c"]
    return sm.OLS(d["aeei"], sm.add_constant(d[["fdi_c", "hhi_c", "int_c", "resource_exports_pct"]])).fit(cov_type="HC1")


def fit_fd(cols):
    """First-difference change model on the chosen differenced predictors."""
    d = pd.read_csv(os.path.join(DATA_DIR, "first_difference_dataset.csv"))[["delta_aeei"] + cols]
    d = d.apply(pd.to_numeric, errors="coerce").dropna()
    return sm.OLS(d["delta_aeei"], sm.add_constant(d[cols])).fit(cov_type="HC1")


# ---------------------------------------------------------------------------
# 2. Turn one fitted model into a tidy full-output block.
# ---------------------------------------------------------------------------
def block(model, label):
    rows = []
    for term in model.params.index:
        rows.append({
            "Model":     label,
            "Variable":  NICE.get(term, term),
            "Coef.":     round(model.params[term], 3),
            "Robust SE": round(model.bse[term], 3),
            "z":         round(model.tvalues[term], 2),
            "p":         round(model.pvalues[term], 3),
        })
    # Add the fit statistics as their own rows.
    rows.append({"Model": label, "Variable": "R-squared", "Coef.": round(model.rsquared, 3),
                 "Robust SE": "", "z": "", "p": ""})
    rows.append({"Model": label, "Variable": "Observations", "Coef.": int(model.nobs),
                 "Robust SE": "", "z": "", "p": ""})
    return rows


# ---------------------------------------------------------------------------
# 3. Build every model and print + save the combined output.
# ---------------------------------------------------------------------------
all_rows = []
all_rows += block(fit_h1("aeei1_dataset.csv"), "H1 baseline, AEEI1 (2019)")
all_rows += block(fit_h1("aeei2_dataset.csv"), "H1 baseline, AEEI2 (2022)")
all_rows += block(fit_h2("aeei1_dataset.csv"), "H2 interaction, AEEI1 (2019)")
all_rows += block(fit_h2("aeei2_dataset.csv"), "H2 interaction, AEEI2 (2022)")
all_rows += block(fit_fd(["delta_ln_fdi_gdp", "delta_hhi", "delta_resource_exports_pct"]),
                  "First difference, change model 1")
all_rows += block(fit_fd(["delta_ln_fdi_gdp", "delta_hhi", "delta_fdi_hhi", "delta_resource_exports_pct"]),
                  "First difference, change model 2")

out = pd.DataFrame(all_rows)
print(out.to_string(index=False))
csv = os.path.join(OUT_DIR, "appendix_4_full_output.csv")
out.to_csv(csv, index=False)
print("\nSaved:", csv)
print("Done.")
