import os
import pandas as pd
import statsmodels.api as sm

DATA_DIR = r"C:\Thesis-IMI-AEEI-2026\thesis_data\imf_cdis\analysis_ready"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

NICE = {
    "const": "Constant", "ln_fdi_gdp": "ln(FDI / GDP)",
    "hhi": "Source concentration (HHI)", "resource_exports_pct": "Resource exports (%)",
    "fdi_c": "ln(FDI / GDP)", "hhi_c": "Source concentration (HHI)", "int_c": "ln(FDI / GDP) x HHI",
    "delta_ln_fdi_gdp": "Change in ln(FDI / GDP)", "delta_hhi": "Change in HHI",
    "delta_fdi_hhi": "Change in ln(FDI / GDP) x HHI",
    "delta_resource_exports_pct": "Change in resource exports (%)",
}


def fit_h1(name):
    d = pd.read_csv(os.path.join(DATA_DIR, name))[["aeei", "ln_fdi_gdp", "hhi", "resource_exports_pct"]]
    d = d.apply(pd.to_numeric, errors="coerce").dropna()
    return sm.OLS(d["aeei"], sm.add_constant(d[["ln_fdi_gdp", "hhi", "resource_exports_pct"]])).fit(cov_type="HC1")


def fit_h2(name):
    d = pd.read_csv(os.path.join(DATA_DIR, name))[["aeei", "ln_fdi_gdp", "hhi", "resource_exports_pct"]]
    d = d.apply(pd.to_numeric, errors="coerce").dropna().copy()
    d["fdi_c"] = d["ln_fdi_gdp"] - d["ln_fdi_gdp"].mean()
    d["hhi_c"] = d["hhi"] - d["hhi"].mean()
    d["int_c"] = d["fdi_c"] * d["hhi_c"]
    return sm.OLS(d["aeei"], sm.add_constant(d[["fdi_c", "hhi_c", "int_c", "resource_exports_pct"]])).fit(cov_type="HC1")


def fit_fd(cols):
    d = pd.read_csv(os.path.join(DATA_DIR, "first_difference_dataset.csv"))[["delta_aeei"] + cols]
    d = d.apply(pd.to_numeric, errors="coerce").dropna()
    return sm.OLS(d["delta_aeei"], sm.add_constant(d[cols])).fit(cov_type="HC1")


def block(model, label):
    ci = model.conf_int()
    rows = []
    for term in model.params.index:
        rows.append({
            "Model": label, "Variable": NICE.get(term, term),
            "Coef.": round(model.params[term], 3), "Robust SE": round(model.bse[term], 3),
            "z": round(model.tvalues[term], 2), "p": round(model.pvalues[term], 3),
            "CI 2.5%": round(ci.loc[term, 0], 3), "CI 97.5%": round(ci.loc[term, 1], 3),
        })
    rows.append({"Model": label, "Variable": "R-squared", "Coef.": round(model.rsquared, 3),
                 "Robust SE": "", "z": "", "p": "", "CI 2.5%": "", "CI 97.5%": ""})
    rows.append({"Model": label, "Variable": "Observations", "Coef.": int(model.nobs),
                 "Robust SE": "", "z": "", "p": "", "CI 2.5%": "", "CI 97.5%": ""})
    return rows


rows = []
rows += block(fit_h1("aeei1_dataset.csv"), "H1 baseline, AEEI1 (2019)")
rows += block(fit_h1("aeei2_dataset.csv"), "H1 baseline, AEEI2 (2022)")
rows += block(fit_h2("aeei1_dataset.csv"), "H2 interaction, AEEI1 (2019)")
rows += block(fit_h2("aeei2_dataset.csv"), "H2 interaction, AEEI2 (2022)")
rows += block(fit_fd(["delta_ln_fdi_gdp", "delta_hhi", "delta_resource_exports_pct"]), "First difference, change model 1")
rows += block(fit_fd(["delta_ln_fdi_gdp", "delta_hhi", "delta_fdi_hhi", "delta_resource_exports_pct"]), "First difference, change model 2")

out = pd.DataFrame(rows)
out.to_csv(os.path.join(OUT_DIR, "appendix_4_full_output.csv"), index=False)
print(out.to_string(index=False))
