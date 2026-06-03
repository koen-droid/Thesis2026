"""
Appendix 2 -- Variance Inflation Factors (multicollinearity)
============================================================

This script produces the Variance Inflation Factor (VIF) table reported in
Appendix 2 of the thesis (referenced from Section 4.2). The VIF checks whether
the predictors are too closely linked to be used together: it tells us how much
each predictor is explained by the other predictors. A value above 5 usually
signals a multicollinearity problem.

Because the moderator test (H2) uses an interaction term, ln(FDI/GDP) and HHI
are mean-centered before they are multiplied. This is the standard way to stop
the product term from getting a high VIF for no real reason. The table reports
the VIF for the centered version, which is exactly what the H2 regression uses.

It is part of the replication code for the master's thesis:
  "Does It Matter Where FDI Comes From? Inward FDI, Source-Country
   Concentration, and Entrepreneurial Ecosystem Quality across African
   Countries" (K.W. Vonk, MSc International Management, Utrecht University).

WHAT IT READS
-------------
The two analysis-ready datasets built by the data pipeline:
  * aeei1_dataset.csv  (wave 1, FDI measured in 2019, n = 28)
  * aeei2_dataset.csv  (wave 2, FDI measured in 2022, n = 30)
Inward FDI and HHI are from IMF CDIS.

WHAT IT WRITES (next to this script)
------------------------------------
  * appendix_2_vif.csv / .xlsx

HOW TO RUN
----------
  python appendix_2_vif.py
Set DATA_DIR below to the folder that holds the two CSV files.
"""

import os
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant


# ---------------------------------------------------------------------------
# 0. Where the input data lives, and where to write the output.
# ---------------------------------------------------------------------------
DATA_DIR = r"C:\Thesis-IMI-AEEI-2026\thesis_data\imf_cdis\analysis_ready"
OUT_DIR  = os.path.dirname(os.path.abspath(__file__))  # same folder as this script


# ---------------------------------------------------------------------------
# 1. Load the two waves.
# ---------------------------------------------------------------------------
print("Loading datasets ...")
aeei1 = pd.read_csv(os.path.join(DATA_DIR, "aeei1_dataset.csv"))
aeei2 = pd.read_csv(os.path.join(DATA_DIR, "aeei2_dataset.csv"))
print(f"  Wave 1: {len(aeei1)} countries")
print(f"  Wave 2: {len(aeei2)} countries")


# ---------------------------------------------------------------------------
# 2. Compute the VIF table for one wave (with the mean-centered interaction).
# ---------------------------------------------------------------------------
def vif_table(df, wave_label):
    """Return a VIF table for the four predictors used in the H2 model."""
    work = df.copy()

    # Mean-center the two ingredients of the interaction, then multiply.
    work["ln_fdi_gdp_c"] = work["ln_fdi_gdp"] - work["ln_fdi_gdp"].mean()
    work["hhi_c"]        = work["hhi"]        - work["hhi"].mean()
    work["fdi_hhi_c"]    = work["ln_fdi_gdp_c"] * work["hhi_c"]

    # Predictors exactly as the H2 regression uses them.
    predictors = ["ln_fdi_gdp_c", "hhi_c", "fdi_hhi_c", "resource_exports_pct"]
    nice = {
        "ln_fdi_gdp_c":         "ln(FDI / GDP), centered",
        "hhi_c":                "HHI, centered",
        "fdi_hhi_c":            "ln(FDI/GDP) x HHI (interaction)",
        "resource_exports_pct": "Resource exports (%)",
    }

    # Drop rows with any missing predictor (resource exports missing for 2).
    X = work[predictors].apply(pd.to_numeric, errors="coerce").dropna()
    X = add_constant(X)  # VIF needs the intercept column included

    rows = []
    for i, name in enumerate(X.columns):
        if name == "const":
            continue  # do not report VIF for the intercept
        rows.append({
            "Predictor": nice[name],
            "VIF":       round(variance_inflation_factor(X.values, i), 3),
        })
    out = pd.DataFrame(rows)
    out = out.rename(columns={"VIF": f"VIF ({wave_label})"})
    print(f"\nVIF ({wave_label}, n = {len(X)}):")
    print(out.to_string(index=False))
    return out


# ---------------------------------------------------------------------------
# 3. Run both waves and put them side by side in one table.
# ---------------------------------------------------------------------------
vif1 = vif_table(aeei1, "Wave 1")
vif2 = vif_table(aeei2, "Wave 2")

# Merge on the predictor name so the two waves sit in adjacent columns.
vif_all = vif1.merge(vif2, on="Predictor")
print("\nCombined VIF table:")
print(vif_all.to_string(index=False))


# ---------------------------------------------------------------------------
# 4. Save.
# ---------------------------------------------------------------------------
csv  = os.path.join(OUT_DIR, "appendix_2_vif.csv")
xlsx = os.path.join(OUT_DIR, "appendix_2_vif.xlsx")
vif_all.to_csv(csv, index=False)
vif_all.to_excel(xlsx, index=False)
print("\nSaving:")
print("  " + csv)
print("  " + xlsx)
print("\nDone.")
