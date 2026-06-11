import os
import numpy as np
import pandas as pd
import statsmodels.api as sm

DATA_DIR = r"C:\Thesis-IMI-AEEI-2026\thesis_data\imf_cdis\analysis_ready"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

LEVELS = [("Low concentration (-1 SD)", -1), ("Mean concentration", 0), ("High concentration (+1 SD)", 1)]


def simple_slopes(name, label):
    d = pd.read_csv(os.path.join(DATA_DIR, name))[["aeei", "ln_fdi_gdp", "hhi", "resource_exports_pct"]]
    d = d.apply(pd.to_numeric, errors="coerce").dropna().copy()
    d["fdi_c"] = d["ln_fdi_gdp"] - d["ln_fdi_gdp"].mean()
    d["hhi_c"] = d["hhi"] - d["hhi"].mean()
    d["int_c"] = d["fdi_c"] * d["hhi_c"]
    model = sm.OLS(d["aeei"], sm.add_constant(d[["fdi_c", "hhi_c", "int_c", "resource_exports_pct"]])).fit(cov_type="HC1")
    sd = d["hhi"].std()
    rows = []
    for name_level, k in LEVELS:
        contrast = model.t_test([0, 1, 0, k * sd, 0])
        lo, hi = np.ravel(contrast.conf_int())[:2]
        rows.append({
            "Wave": label, "Concentration": name_level,
            "Slope": round(float(np.ravel(contrast.effect)[0]), 3),
            "Robust SE": round(float(np.ravel(contrast.sd)[0]), 3),
            "z": round(float(np.ravel(contrast.tvalue)[0]), 2),
            "p": round(float(np.ravel(contrast.pvalue)[0]), 3),
            "CI 2.5%": round(lo, 3), "CI 97.5%": round(hi, 3),
        })
    return rows


rows = simple_slopes("aeei1_dataset.csv", "AEEI1 (2019)") + simple_slopes("aeei2_dataset.csv", "AEEI2 (2022)")
out = pd.DataFrame(rows)
out.to_csv(os.path.join(OUT_DIR, "appendix_5_simple_slopes.csv"), index=False)
print(out.to_string(index=False))
