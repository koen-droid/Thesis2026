import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

DATA_DIR = r"C:\Thesis-IMI-AEEI-2026\thesis_data\imf_cdis\analysis_ready"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

X, Y, LABEL = "delta_ln_fdi_gdp", "delta_aeei", "iso3"

df = pd.read_csv(os.path.join(DATA_DIR, "first_difference_dataset.csv"))[[LABEL, X, Y]]
df[X] = pd.to_numeric(df[X], errors="coerce")
df[Y] = pd.to_numeric(df[Y], errors="coerce")
df = df.dropna(subset=[X, Y])

x, y = df[X].values, df[Y].values
fit = sm.OLS(y, sm.add_constant(x)).fit()
grid = np.linspace(x.min(), x.max(), 100)
pred = fit.get_prediction(sm.add_constant(grid)).summary_frame(alpha=0.05)

plt.rcParams["font.family"] = "serif"
fig, ax = plt.subplots(figsize=(7.6, 5.6))
ax.axhline(0, color="0.6", lw=0.8, ls="--")
ax.axvline(0, color="0.6", lw=0.8, ls="--")
ax.fill_between(grid, pred["mean_ci_lower"], pred["mean_ci_upper"], color="0.80", alpha=0.7, label="95% confidence")
ax.plot(grid, pred["mean"], color="black", lw=1.6, label="Fitted line")
ax.scatter(x, y, s=34, color="#1f4e79", edgecolor="white", linewidth=0.6, zorder=3)
for _, row in df.iterrows():
    ax.annotate(row[LABEL], (row[X], row[Y]), xytext=(3, 3), textcoords="offset points", fontsize=7, color="0.25")
ax.set_xlabel("Change in ln(FDI / GDP)  (wave 2 - wave 1)", fontsize=10)
ax.set_ylabel("Change in AEEI score  (wave 2 - wave 1)", fontsize=10)
ax.tick_params(labelsize=9)
ax.legend(loc="upper right", fontsize=8, frameon=False)

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "appendix_3_within_country_scatter.png"), dpi=300, bbox_inches="tight", facecolor="white")
print(f"slope = {fit.params[1]:.3f}, p = {fit.pvalues[1]:.3f}, n = {len(df)}")
