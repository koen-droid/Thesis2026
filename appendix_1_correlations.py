import os
import pandas as pd

DATA_DIR = r"C:\Thesis-IMI-AEEI-2026\thesis_data\imf_cdis\analysis_ready"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

LABELS = {
    "aeei": "AEEI score",
    "ln_fdi_gdp": "ln(FDI / GDP)",
    "hhi": "Source concentration (HHI)",
    "resource_exports_pct": "Resource exports (%)",
}


def correlation_matrix(path):
    df = pd.read_csv(path)[list(LABELS)].apply(pd.to_numeric, errors="coerce")
    return df.rename(columns=LABELS).corr(method="pearson").round(3)


for filename, wave in [("aeei1_dataset.csv", "wave1"), ("aeei2_dataset.csv", "wave2")]:
    corr = correlation_matrix(os.path.join(DATA_DIR, filename))
    corr.to_csv(os.path.join(OUT_DIR, f"appendix_1_correlations_{wave}.csv"))
    corr.to_excel(os.path.join(OUT_DIR, f"appendix_1_correlations_{wave}.xlsx"))
    print(f"{wave}:\n{corr.to_string()}\n")
