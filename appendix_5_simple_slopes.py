"""
Appendix 5 -- Simple slopes of FDI at low, mean and high concentration
======================================================================

This script reproduces the simple-slopes table reported in Appendix 5 of the
thesis (referenced from Section 4.4). It reports the effect (slope) of
ln(FDI/GDP) on the AEEI score at three levels of source-country concentration:
one standard deviation below the mean HHI (low), at the mean, and one standard
deviation above (high). Each slope, its robust standard error, z-statistic and
p-value come from a single linear contrast test on the H2 interaction model, so
every number is a real model result rather than a hand calculation.

It is part of the replication code for the master's thesis:
  "Does It Matter Where FDI Comes From? Inward FDI, Source-Country
   Concentration, and Entrepreneurial Ecosystem Quality across African
   Countries" (K.W. Vonk, MSc International Management, Utrecht University).

WHAT IT READS
-------------
  * aeei1_dataset.csv  (wave 1, FDI 2019)
  * aeei2_dataset.csv  (wave 2, FDI 2022)

WHAT IT WRITES (next to this script)
------------------------------------
  * appendix_5_simple_slopes.csv

HOW TO RUN
----------
  python appendix_5_simple_slopes.py
Set DATA_DIR below to the folder that holds the CSV files.
"""

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm


# ---------------------------------------------------------------------------
# 0. Where the input data lives, and where to write the output.
# ---------------------------------------------------------------------------
DATA_DIR = r"C:\Thesis-IMI-AEEI-2026\thesis_data\imf_cdis\analysis_ready"
OUT_DIR  = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# 1. Fit the H2 interaction model and compute the three simple slopes.
#    The model column order is [const, fdi_c, hhi_c, int_c, resource], so the
#    slope of FDI at a centered HHI value h is the contrast [0, 1, 0, h, 0].
# ---------------------------------------------------------------------------
def simple_slopes(name, label):
    d = pd.read_csv(os.path.join(DATA_DIR, name))[["aeei", "ln_fdi_gdp", "hhi", "resource_exports_pct"]]
    d = d.apply(pd.to_numeric, errors="coerce").dropna().copy()
    d["fdi_c"] = d["ln_fdi_gdp"] - d["ln_fdi_gdp"].mean()
    d["hhi_c"] = d["hhi"] - d["hhi"].mean()
    d["int_c"] = d["fdi_c"] * d["hhi_c"]
    model = sm.OLS(d["aeei"], sm.add_constant(d[["fdi_c", "hhi_c", "int_c", "resource_exports_pct"]])).fit(cov_type="HC1")
    sd = d["hhi"].std()

    rows = []
    for name_level, h in [("Low concentration (-1 SD)", -sd),
                          ("Mean concentration", 0.0),
                          ("High concentration (+1 SD)", sd)]:
        t = model.t_test([0, 1, 0, h, 0])
        rows.append({
            "Wave":          label,
            "Concentration": name_level,
            "Slope":         round(float(np.ravel(t.effect)[0]), 3),
            "Robust SE":     round(float(np.ravel(t.sd)[0]), 3),
            "z":             round(float(np.ravel(t.tvalue)[0]), 2),
            "p":             round(float(np.ravel(t.pvalue)[0]), 3),
        })
    return rows


# ---------------------------------------------------------------------------
# 2. Run both waves and save.
# ---------------------------------------------------------------------------
rows = simple_slopes("aeei1_dataset.csv", "AEEI1 (2019)") + \
       simple_slopes("aeei2_dataset.csv", "AEEI2 (2022)")
out = pd.DataFrame(rows)
print(out.to_string(index=False))
csv = os.path.join(OUT_DIR, "appendix_5_simple_slopes.csv")
out.to_csv(csv, index=False)
print("\nSaved:", csv)
print("Done.")
