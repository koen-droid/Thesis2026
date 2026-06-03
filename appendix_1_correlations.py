"""
Appendix 1 -- Pearson correlation matrices
==========================================

This script produces the Pearson correlation matrices reported in Appendix 1
of the thesis (referenced from Section 4.2). It computes, for each AEEI wave,
the simple correlations between the dependent variable (the AEEI score) and the
model predictors.

It is part of the replication code for the master's thesis:
  "Does It Matter Where FDI Comes From? Inward FDI, Source-Country
   Concentration, and Entrepreneurial Ecosystem Quality across African
   Countries" (K.W. Vonk, MSc International Management, Utrecht University).

WHAT IT READS
-------------
The two analysis-ready datasets built by the data pipeline:
  * aeei1_dataset.csv  (wave 1, FDI measured in 2019, n = 28)
  * aeei2_dataset.csv  (wave 2, FDI measured in 2022, n = 30)
Inward FDI is the inward direct investment position from IMF CDIS.

WHAT IT WRITES (next to this script)
------------------------------------
  * appendix_1_correlations_wave1.csv / .xlsx
  * appendix_1_correlations_wave2.csv / .xlsx

HOW TO RUN
----------
  python appendix_1_correlations.py
Set DATA_DIR below to the folder that holds the two CSV files.
"""

import os
import pandas as pd


# ---------------------------------------------------------------------------
# 0. Where the input data lives, and where to write the output.
# ---------------------------------------------------------------------------
# Point this at the folder that contains aeei1_dataset.csv and aeei2_dataset.csv.
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
# 2. The variables, with short readable labels for the table.
# ---------------------------------------------------------------------------
# Dependent variable first, then the predictors.
labels = {
    "aeei":                 "AEEI score",
    "ln_fdi_gdp":           "ln(FDI / GDP)",
    "hhi":                  "Source concentration (HHI)",
    "resource_exports_pct": "Resource exports (%)",
}


# ---------------------------------------------------------------------------
# 3. Build the correlation matrix for one wave.
# ---------------------------------------------------------------------------
def correlation_table(df, wave_label):
    """Return a rounded Pearson correlation matrix among DV and predictors."""
    cols = list(labels.keys())
    sub = df[cols].apply(pd.to_numeric, errors="coerce")  # force numeric
    sub = sub.rename(columns=labels)                      # readable names
    corr = sub.corr(method="pearson").round(3)
    print(f"\nCorrelation matrix ({wave_label}):")
    print(corr.to_string())
    return corr


# ---------------------------------------------------------------------------
# 4. Run both waves and save.
# ---------------------------------------------------------------------------
corr1 = correlation_table(aeei1, "Wave 1 (2019 FDI)")
corr2 = correlation_table(aeei2, "Wave 2 (2022 FDI)")


def save(df, name):
    csv  = os.path.join(OUT_DIR, name + ".csv")
    xlsx = os.path.join(OUT_DIR, name + ".xlsx")
    df.to_csv(csv)            # keep the row labels (the variable names)
    df.to_excel(xlsx)
    print("  " + csv)
    print("  " + xlsx)


print("\nSaving:")
save(corr1, "appendix_1_correlations_wave1")
save(corr2, "appendix_1_correlations_wave2")
print("\nDone.")
