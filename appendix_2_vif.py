import os
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

DATA_DIR = r"C:\Thesis-IMI-AEEI-2026\thesis_data\imf_cdis\analysis_ready"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

PREDICTORS = ["ln_fdi_gdp_c", "hhi_c", "fdi_hhi_c", "resource_exports_pct"]
LABELS = {
    "ln_fdi_gdp_c": "ln(FDI / GDP), centered",
    "hhi_c": "HHI, centered",
    "fdi_hhi_c": "ln(FDI/GDP) x HHI (interaction)",
    "resource_exports_pct": "Resource exports (%)",
}


def vif_table(path, wave):
    df = pd.read_csv(path)
    df["ln_fdi_gdp_c"] = df["ln_fdi_gdp"] - df["ln_fdi_gdp"].mean()
    df["hhi_c"] = df["hhi"] - df["hhi"].mean()
    df["fdi_hhi_c"] = df["ln_fdi_gdp_c"] * df["hhi_c"]
    X = add_constant(df[PREDICTORS].apply(pd.to_numeric, errors="coerce").dropna())
    rows = [
        {"Predictor": LABELS[col], f"VIF ({wave})": round(variance_inflation_factor(X.values, i), 3)}
        for i, col in enumerate(X.columns) if col != "const"
    ]
    return pd.DataFrame(rows)


vif1 = vif_table(os.path.join(DATA_DIR, "aeei1_dataset.csv"), "Wave 1")
vif2 = vif_table(os.path.join(DATA_DIR, "aeei2_dataset.csv"), "Wave 2")
vif = vif1.merge(vif2, on="Predictor")
vif.to_csv(os.path.join(OUT_DIR, "appendix_2_vif.csv"), index=False)
vif.to_excel(os.path.join(OUT_DIR, "appendix_2_vif.xlsx"), index=False)
print(vif.to_string(index=False))
