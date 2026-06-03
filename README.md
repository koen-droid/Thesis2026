# Thesis 2026 — Appendix code

Replication code for the appendices of the master's thesis:

> **Does It Matter Where FDI Comes From? Inward FDI, Source-Country
> Concentration, and Entrepreneurial Ecosystem Quality across African
> Countries**
> K.W. Vonk — MSc International Management, Utrecht University.

Each script below reproduces one appendix table. The thesis text links to this
repository so readers can see exactly how each table was produced.

## Appendices

| Appendix | Script | What it produces |
|---|---|---|
| Appendix 1 | [`appendix_1_correlations.py`](appendix_1_correlations.py) | Pearson correlation matrices among the AEEI score and the predictors, per wave (referenced from Section 4.2). |
| Appendix 2 | [`appendix_2_vif.py`](appendix_2_vif.py) | Variance Inflation Factors (VIF) for the model predictors, per wave; the interaction term is mean-centered (referenced from Section 4.2). |

## Data

The scripts read the two analysis-ready datasets (`aeei1_dataset.csv`,
`aeei2_dataset.csv`). Inward FDI and source-country concentration (HHI) come
from the IMF Coordinated Direct Investment Survey (CDIS); GDP is from the World
Bank. Set the `DATA_DIR` variable at the top of each script to the folder that
holds the two CSV files.

## Requirements

```
pandas
openpyxl
statsmodels
```

## How to run

```
python appendix_1_correlations.py
python appendix_2_vif.py
```

Each script prints its table to the screen and saves it next to the script as
`.csv` and `.xlsx`.
