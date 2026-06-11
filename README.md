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
| Appendix 2 | [`appendix_2_vif.py`](appendix_2_vif.py) | Variance Inflation Factors (VIF) for the model predictors, per wave, with the interaction term mean-centered (referenced from Section 4.2). |
| Appendix 3 | [`appendix_3_within_country_scatter.py`](appendix_3_within_country_scatter.py) | Within-country scatter of the change in FDI intensity against the change in the AEEI score (referenced from Section 4.5). |
| Appendix 4 | [`appendix_4_regression_output.py`](appendix_4_regression_output.py) | Full regression output (coefficient, robust SE, z, p, R-squared, N) for H1, H2 and the first-difference models (Sections 4.3 to 4.5). |
| Appendix 5 | [`appendix_5_simple_slopes.py`](appendix_5_simple_slopes.py) | Simple slopes of FDI at low, mean and high source concentration, with robust SE, z and p (referenced from Section 4.4). |

## Data

The scripts read the analysis-ready datasets (`aeei1_dataset.csv`,
`aeei2_dataset.csv`, `first_difference_dataset.csv`). Inward FDI and
source-country concentration (HHI) come from the IMF Coordinated Direct
Investment Survey (CDIS), GDP is from the World Bank. Set the `DATA_DIR`
variable at the top of each script to the folder that holds the CSV files.

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
python appendix_3_within_country_scatter.py
python appendix_4_regression_output.py
python appendix_5_simple_slopes.py
```

Each script prints its result to the screen and saves it next to the script
(`.csv`/`.xlsx` for tables, `.png` for the scatter).
