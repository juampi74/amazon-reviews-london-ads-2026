"""
model_validation.py — Batería de validación de veracidad del modelo (Persona 2).

Va más allá de re-imprimir métricas: aplica pruebas que demuestran (o refutan) que el
modelo aprendió señal REAL y no un artefacto.

  T1  Generalización en holdout (train/test estratificado, features SIN fuga)
  T2  Comparación contra baselines (Dummy, regresión logística, un solo feature)
  T3  Test de PERMUTACIÓN de etiquetas (control negativo): al barajar y, el AUC
      debe colapsar a ~0.5. Prueba de que la señal es real.
  T4  Estabilidad entre semillas (varias particiones) -> AUC media ± std
  T5  Desempeño por subcategoría (¿funciona across categorías?)

Todas las features de texto/price_fit se ajustan SOLO en train (sin fuga), reutilizando
los helpers de leakage_audit.py.

Salida: output/metrics/model_validation.json
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

import metrics as M
import run_pipeline as P
import leakage_audit as LA

warnings.filterwarnings("ignore")
RNG = 42


def build_split(dtr, dte):
    """Features leak-free: TF-IDF y price_fit ajustados SOLO en train."""
    vec = TfidfVectorizer(max_features=300, stop_words="english",
                          ngram_range=(1, 2), sublinear_tf=True)
    Xtr_txt = vec.fit_transform(dtr["text"].fillna("")).toarray()
    Xte_txt = vec.transform(dte["text"].fillna("")).toarray()
    Xtr = LA.assemble(dtr, LA.price_fit_from_train(dtr, dtr), Xtr_txt)
    Xte = LA.assemble(dte, LA.price_fit_from_train(dtr, dte), Xte_txt)
    return Xtr, Xte


def success_model(seed=RNG):
    return M.make_success_classifier(random_state=seed)


def main():
    print("Cargando datos + target fijo...")
    df = P.load_clean().reset_index(drop=True)
    thr = M.subcategory_thresholds(df)
    df["success"] = M.label_success(df, thr)
    y = df["success"].values
    results = {}

    # ---------------- T1: holdout ----------------
    print("\n[T1] Generalización en holdout 80/20 (sin fuga)")
    idx = np.arange(len(df))
    tr, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=RNG)
    dtr, dte = df.iloc[tr], df.iloc[te]
    Xtr, Xte = build_split(dtr, dte)
    model = success_model().fit(Xtr, y[tr])
    proba_te = model.predict_proba(Xte)[:, 1]
    # calibración HONESTA: isotónica ajustada sobre predicciones OUT-OF-FOLD del train
    # (no sobre las de entrenamiento, que quedan sobre-confiadas), aplicada al test.
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    proba_tr_oof = cross_val_predict(
        success_model(), Xtr, y[tr], method="predict_proba",
        cv=StratifiedKFold(5, shuffle=True, random_state=RNG))[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(proba_tr_oof, y[tr])
    proba_te_cal = iso.predict(proba_te)
    t1 = {
        "test_n": int(len(te)),
        "roc_auc": round(float(roc_auc_score(y[te], proba_te)), 4),
        "pr_auc": round(float(average_precision_score(y[te], proba_te)), 4),
        "brier_raw": round(float(brier_score_loss(y[te], proba_te)), 4),
        "brier_calibrated": round(float(brier_score_loss(y[te], proba_te_cal)), 4),
        "ece_raw": round(float(M.expected_calibration_error(y[te], proba_te)), 4),
        "ece_calibrated": round(float(M.expected_calibration_error(y[te], proba_te_cal)), 4),
        "base_rate": round(float(y[te].mean()), 4),
    }
    results["T1_holdout"] = t1
    print("  ", t1)

    # ---------------- T2: baselines ----------------
    print("\n[T2] Baselines en el mismo holdout")
    base = {}
    for name, clf in [("dummy_stratified", DummyClassifier(strategy="stratified", random_state=RNG)),
                      ("dummy_most_frequent", DummyClassifier(strategy="most_frequent"))]:
        clf.fit(Xtr, y[tr])
        p = clf.predict_proba(Xte)[:, 1]
        base[name] = round(float(roc_auc_score(y[te], p)), 4)
    # logística solo con features estructuradas (num + one-hot, sin texto): ¿el texto aporta?
    n_struct = 2 + len(LA.DETAIL_FLAGS) + len(P.REAL_SUBCATS)   # price_fit,is_missing,flags,onehot
    lr = LogisticRegression(max_iter=1000, class_weight="balanced")
    lr.fit(Xtr[:, :n_struct], y[tr])
    base["logreg_structured_only"] = round(float(roc_auc_score(
        y[te], lr.predict_proba(Xte[:, :n_struct])[:, 1])), 4)
    # un solo feature: price_fit (columna 0)
    lr1 = LogisticRegression(max_iter=1000, class_weight="balanced")
    lr1.fit(Xtr[:, [0]], y[tr])
    base["logreg_price_fit_only"] = round(float(roc_auc_score(
        y[te], lr1.predict_proba(Xte[:, [0]])[:, 1])), 4)
    base["catboost_full"] = t1["roc_auc"]
    results["T2_baselines_roc_auc"] = base
    print("  ", base)

    # ---------------- T3: permutación de etiquetas (control negativo) ----------------
    print("\n[T3] Test de permutación de etiquetas (debe dar AUC ~0.5)")
    perm_aucs = []
    rng = np.random.default_rng(RNG)
    for i in range(3):
        y_shuf = y[tr].copy()
        rng.shuffle(y_shuf)
        m = success_model(seed=100 + i).fit(Xtr, y_shuf)
        perm_aucs.append(float(roc_auc_score(y[te], m.predict_proba(Xte)[:, 1])))
        print(f"  permutación {i+1}: AUC={perm_aucs[-1]:.4f}")
    results["T3_label_permutation"] = {
        "permuted_auc_mean": round(float(np.mean(perm_aucs)), 4),
        "permuted_auc_std": round(float(np.std(perm_aucs)), 4),
        "real_auc": t1["roc_auc"],
        "verdict": ("PASA: la señal es real (AUC real >> AUC permutado ~0.5)"
                    if t1["roc_auc"] - np.mean(perm_aucs) > 0.15 else
                    "REVISAR: la brecha real vs permutado es pequeña"),
    }
    print("  ", results["T3_label_permutation"]["verdict"])

    # ---------------- T4: estabilidad entre semillas ----------------
    print("\n[T4] Estabilidad: holdout AUC con 5 particiones distintas")
    seed_aucs = [t1["roc_auc"]]
    for s in [1, 7, 21, 99]:
        tr2, te2 = train_test_split(idx, test_size=0.2, stratify=y, random_state=s)
        Xtr2, Xte2 = build_split(df.iloc[tr2], df.iloc[te2])
        m = success_model(seed=s).fit(Xtr2, y[tr2])
        a = float(roc_auc_score(y[te2], m.predict_proba(Xte2)[:, 1]))
        seed_aucs.append(round(a, 4))
        print(f"  seed {s}: AUC={a:.4f}")
    results["T4_stability"] = {
        "aucs": seed_aucs,
        "mean": round(float(np.mean(seed_aucs)), 4),
        "std": round(float(np.std(seed_aucs)), 4),
    }
    print("  media±std:", results["T4_stability"]["mean"], "±", results["T4_stability"]["std"])

    # ---------------- T5: por subcategoría ----------------
    print("\n[T5] AUC por subcategoría (holdout T1)")
    per_sub = {}
    sub_te = dte["subcategory"].values
    for sc in P.REAL_SUBCATS:
        mask = sub_te == sc
        if mask.sum() > 30 and 0 < y[te][mask].mean() < 1:
            per_sub[sc] = {
                "n": int(mask.sum()),
                "roc_auc": round(float(roc_auc_score(y[te][mask], proba_te[mask])), 4),
                "base_rate": round(float(y[te][mask].mean()), 4),
            }
    results["T5_per_subcategory"] = per_sub
    for sc, v in per_sub.items():
        print(f"  {sc:26s} n={v['n']:4d} AUC={v['roc_auc']:.3f} base={v['base_rate']:.2f}")

    (P.OUT_METRICS / "model_validation.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nGuardado -> {P.OUT_METRICS/'model_validation.json'}")
    print("OK.")


if __name__ == "__main__":
    main()
