"""
Appendix 3 -- Within-country change scatter (delta FDI vs delta AEEI)
====================================================================

This script produces the scatter plot reported in Appendix 3 of the thesis
(referenced from Section 4.5). Each dot is one country observed in both AEEI
waves. The horizontal axis is the change in FDI intensity (delta ln(FDI/GDP))
and the vertical axis is the change in the AEEI score, both measured as wave 2
minus wave 1. A flat fitted line with a 95% band shows the within-country null
result: changes in FDI intensity are not linked to changes in ecosystem quality.

It is part of the replication code for the master's thesis:
  "Does It Matter Where FDI Comes From? Inward FDI, Source-Country
   Concentration, and Entrepreneurial Ecosystem Quality across African
   Countries" (K.W. Vonk, MSc International Management, Utrecht University).

WHAT IT READS
-------------
  * first_difference_dataset.csv  (27 countries present in both waves)

WHAT IT WRITES (next to this script)
------------------------------------
  * appendix_3_within_country_scatter.png

HOW TO RUN
----------
  python appendix_3_within_country_scatter.py
Set DATA_DIR below to the folder that holds the CSV file.
"""

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 0. Where the input data lives, and where to write the output.
# ---------------------------------------------------------------------------
DATA_DIR = r"C:\Thesis-IMI-AEEI-2026\thesis_data\imf_cdis\analysis_ready"
OUT_DIR  = os.path.dirname(os.path.abspath(__file__))

X, Y, LABEL = "delta_ln_fdi_gdp", "delta_aeei", "iso3"


# ---------------------------------------------------------------------------
# 1. Load the countries present in both waves.
# ---------------------------------------------------------------------------
print("Loading first-difference dataset ...")
df = pd.read_csv(os.path.join(DATA_DIR, "first_difference_dataset.csv"))
d = df[[LABEL, X, Y]].copy()
d[X] = pd.to_numeric(d[X], errors="coerce")
d[Y] = pd.to_numeric(d[Y], errors="coerce")
d = d.dropna(subset=[X, Y])
print(f"  {len(d)} countries")


# ---------------------------------------------------------------------------
# 2. Simple bivariate regression line and 95% confidence band.
# ---------------------------------------------------------------------------
x, y = d[X].values, d[Y].values
fit = sm.OLS(y, sm.add_constant(x)).fit()
b0, b1 = fit.params
p_slope = fit.pvalues[1]

x_grid = np.linspace(x.min(), x.max(), 100)
pred = fit.get_prediction(sm.add_constant(x_grid)).summary_frame(alpha=0.05)


# ---------------------------------------------------------------------------
# 3. Draw the scatter with country labels.
# ---------------------------------------------------------------------------
plt.rcParams["font.family"] = "serif"
fig, ax = plt.subplots(figsize=(7.6, 5.6))
ax.axhline(0, color="0.6", lw=0.8, ls="--")
ax.axvline(0, color="0.6", lw=0.8, ls="--")
ax.fill_between(x_grid, pred["mean_ci_lower"], pred["mean_ci_upper"],
                color="0.80", alpha=0.7, label="95% confidence")
ax.plot(x_grid, pred["mean"], color="black", lw=1.6, label="Fitted line")
ax.scatter(x, y, s=34, color="#1f4e79", edgecolor="white", linewidth=0.6, zorder=3)
for _, row in d.iterrows():
    ax.annotate(row[LABEL], (row[X], row[Y]), xytext=(3, 3),
                textcoords="offset points", fontsize=7, color="0.25")

ax.set_xlabel("Change in ln(FDI / GDP)  (wave 2 - wave 1)", fontsize=10)
ax.set_ylabel("Change in AEEI score  (wave 2 - wave 1)", fontsize=10)
ax.tick_params(labelsize=9)
ax.legend(loc="upper right", fontsize=8, frameon=False)

note = (f"Each dot is one of the {len(d)} countries present in both AEEI waves. "
        f"The fitted line is flat (slope = {b1:.3f}, p = {p_slope:.2f}).")
fig.text(0.02, -0.01, note, ha="left", va="top", fontsize=8, style="italic")

fig.tight_layout()
out = os.path.join(OUT_DIR, "appendix_3_within_country_scatter.png")
fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
print("Saved:", out)
print("Done.")
