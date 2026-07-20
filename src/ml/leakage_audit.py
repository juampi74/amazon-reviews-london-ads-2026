"""
leakage_audit.py — Auditoría de fuga de datos (tarea de la Persona 2 / Científico de datos)

Re-evalúa el modelo con validación cruzada SIN fuga: la ingeniería de features que usa
información cruzada entre filas (TF-IDF y las estadísticas de price_fit por subcategoría) se
ajusta SOLO con el fold de entrenamiento y se aplica al de test. Compara las métricas
out-of-fold "limpias" contra la versión previa (donde TF-IDF y price_fit se ajustaron sobre
todo el dataset), para cuantificar cuánto optimismo introducía la fuga.

Decisión documentada: el TARGET de éxito se define con percentiles poblacionales (regla de
negocio fija, no una cantidad aprendida) → se mantiene como verdad de terreno fija. Lo que se
corrige aquí es la fuga en las FEATURES.

Salida: output/metrics/leakage_audit.json + actualiza analysis_results.json (part_V_VI.leakage_audit).
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

import metrics as M
import run_pipeline as P

warnings.filterwarnings("ignore")
RNG = 42
REAL_SUBCATS = P.REAL_SUBCATS
DETAIL_FLAGS = ["has_brand", "has_item_form", "has_color",
                "has_scent", "has_skin_type", "has_hair_type"]


def price_fit_from_train(train_df, apply_df):
    """price_fit calculado con mediana/IQR por subcategoría SOLO del train."""
    g = train_df.groupby("subcategory")["price"]
    med = g.median()
    iqr = (g.quantile(0.75) - g.quantile(0.25)).replace(0, np.nan)
    m = apply_df["subcategory"].map(med)
    i = apply_df["subcategory"].map(iqr)
    return ((apply_df["price"] - m) / i).fillna(0.0).values


def assemble(df_part, pf_vals, txt_matrix):
    num = np.column_stack(
        [pf_vals, df_part["price"].isna().astype(int).values]
        + [df_part[c].values for c in DETAIL_FLAGS]
    )
    ohe = (pd.get_dummies(df_part["subcategory"])
           .reindex(columns=REAL_SUBCATS, fill_value=0).values)
    return np.hstack([num, ohe, txt_matrix]).astype(np.float32)


def leakfree_oof(df, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG)
    proba = np.zeros(len(df), dtype=float)
    for i, (tr, te) in enumerate(skf.split(np.arange(len(df)), y), 1):
        dtr, dte = df.iloc[tr], df.iloc[te]
        # TF-IDF: fit SOLO en train
        vec = TfidfVectorizer(max_features=300, stop_words="english",
                              ngram_range=(1, 2), sublinear_tf=True)
        Xtr_txt = vec.fit_transform(dtr["text"].fillna("")).toarray()
        Xte_txt = vec.transform(dte["text"].fillna("")).toarray()
        # price_fit: stats SOLO de train
        Xtr = assemble(dtr, price_fit_from_train(dtr, dtr), Xtr_txt)
        Xte = assemble(dte, price_fit_from_train(dtr, dte), Xte_txt)
        clf = M.make_success_classifier(random_state=RNG)
        clf.fit(Xtr, y[tr])
        proba[te] = clf.predict_proba(Xte)[:, 1]
        print(f"  fold {i}/5 listo")
    return proba


def honest_ece(proba, y):
    """ECE isotónica out-of-fold sobre estas OOF."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG)
    iso_oof = np.zeros_like(proba)
    for tr, te in skf.split(proba, y):
        iso_oof[te] = IsotonicRegression(out_of_bounds="clip").fit(
            proba[tr], y[tr]).predict(proba[te])
    return (float(M.expected_calibration_error(y, proba)),
            float(M.expected_calibration_error(y, iso_oof)))


def main():
    print("Reconstruyendo df + target (verdad fija poblacional)...")
    df = P.load_clean()
    thr = M.subcategory_thresholds(df)
    df["success"] = M.label_success(df, thr)
    y = df["success"].values

    print("\n== CV SIN fuga (TF-IDF + price_fit dentro del fold) ==")
    proba_clean = leakfree_oof(df, y)
    rep_clean = M.classification_report_full(y, proba_clean)
    ece_raw_c, ece_cal_c = honest_ece(proba_clean, y)

    # versión previa (con fuga) para comparar
    proba_leaky = np.load(P.OUT_PRED / "proba_oof.npy")
    rep_leaky = M.classification_report_full(y, proba_leaky)
    ece_raw_l, ece_cal_l = honest_ece(proba_leaky, y)

    def row(rep):
        return {k: round(float(rep[k]), 4) for k in
                ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier"]}

    audit = {
        "note": ("CV sin fuga: TF-IDF y price_fit ajustados SOLO en train por fold. "
                 "El target se mantiene como definición poblacional fija (no es feature)."),
        "leaky_full_fit": {**row(rep_leaky), "ece_uncal": round(ece_raw_l, 4),
                           "ece_isotonic": round(ece_cal_l, 4)},
        "leakfree_infold": {**row(rep_clean), "ece_uncal": round(ece_raw_c, 4),
                            "ece_isotonic": round(ece_cal_c, 4)},
    }
    audit["delta_roc_auc"] = round(audit["leakfree_infold"]["roc_auc"]
                                   - audit["leaky_full_fit"]["roc_auc"], 4)
    audit["delta_pr_auc"] = round(audit["leakfree_infold"]["pr_auc"]
                                  - audit["leaky_full_fit"]["pr_auc"], 4)

    print("\n== RESULTADO DE LA AUDITORÍA ==")
    print(json.dumps(audit, indent=2, ensure_ascii=False))

    (P.OUT_METRICS / "leakage_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False))
    np.save(P.OUT_PRED / "proba_oof_leakfree.npy", proba_clean)

    res_path = P.OUT_METRICS / "analysis_results.json"
    res = json.loads(res_path.read_text())
    res["part_V_VI"]["leakage_audit"] = audit
    res_path.write_text(json.dumps(P.jsonable(res), indent=2, ensure_ascii=False))
    print(f"\nGuardado -> {P.OUT_METRICS/'leakage_audit.json'}")
    print("OK.")


if __name__ == "__main__":
    main()
