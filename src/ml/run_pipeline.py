"""
run_pipeline.py — Pipeline completo de análisis de datos
Product Success Predictor · Beauty & Personal Care

Ejecuta de principio a fin sobre Master_Beauty_Dataset.csv las Partes I–VIII
descritas en FORMULAS_AND_METRICS.md y PROJECT_CONTEXT.md:

  Fase 2  Limpieza / taxonomía (categories[1]) + parseo de details
  Parte I    Estadística descriptiva, skew/kurtosis, outliers, nulos
  Parte II   Correlaciones (Pearson/Spearman), normalidad, hipótesis, chi²/V
  Parte III  Target de éxito (percentiles por subcategoría) + sensibilidad
  Parte IV   Features: price_fit, TF-IDF(title+features), one-hot, flags details
  Parte V    CatBoost + validación estratificada k-fold + probabilidades out-of-fold
  Parte VI   Matriz de confusión, F1/prec/recall, ROC-AUC, PR-AUC, Brier, ECE
  Parte VII  Calibración (isotónica) + importancia (Gini + permutación) + SHAP
  Parte VIII score 0-100, incertidumbre (boosting stages), k-NN/coseno, saturación, riesgo,
             precio sugerido, IC de Wilson

Artefactos (para la app Streamlit) → output/models/
Métricas y tablas → output/metrics/  ·  Figuras → output/figures/
"""
from __future__ import annotations

import ast
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer

import metrics as M

warnings.filterwarnings("ignore")
RNG = 42
np.random.seed(RNG)

# ---- Rutas -----------------------------------------------------------------
HERE = Path(__file__).resolve()
REPO = HERE.parents[2]                 # .../amazon-reviews-london-ads-2026
CSV = REPO / "Master_Beauty_Dataset.csv"

OUT_MODELS = REPO / "output" / "models"
OUT_METRICS = REPO / "output" / "metrics"
OUT_PRED = REPO / "output" / "predictions"
OUT_FIG = REPO / "output" / "figures"
for d in (OUT_MODELS, OUT_METRICS, OUT_PRED, OUT_FIG):
    d.mkdir(parents=True, exist_ok=True)

REAL_SUBCATS = [
    "Hair Care", "Skin Care", "Foot, Hand & Nail Care", "Makeup",
    "Tools & Accessories", "Fragrance", "Shave & Hair Removal", "Personal Care",
]
RESULTS: dict = {}      # se serializa a metrics/analysis_results.json


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    return o


def banner(t):
    print("\n" + "=" * 72 + f"\n{t}\n" + "=" * 72)


# ============================================================
# FASE 2 — Carga y limpieza
# ============================================================

def parse_categories(val):
    """categories es una lista serializada; el nivel 2 (índice 1) es la subcategoría real."""
    try:
        lst = ast.literal_eval(val) if isinstance(val, str) else val
        if isinstance(lst, (list, tuple)) and len(lst) > 1:
            return str(lst[1]).strip()
    except (ValueError, SyntaxError):
        pass
    return None


def parse_details(val):
    try:
        d = ast.literal_eval(val) if isinstance(val, str) else val
        return d if isinstance(d, dict) else {}
    except (ValueError, SyntaxError):
        return {}


def join_list_text(val):
    """features/description vienen como listas serializadas de bullets."""
    try:
        lst = ast.literal_eval(val) if isinstance(val, str) else val
        if isinstance(lst, (list, tuple)):
            return " ".join(str(x) for x in lst)
    except (ValueError, SyntaxError):
        pass
    return "" if val is None or (isinstance(val, float) and np.isnan(val)) else str(val)


def load_clean():
    banner("FASE 2 · Carga y limpieza")
    df = pd.read_csv(CSV, low_memory=False)
    print(f"Filas crudas: {len(df):,} · columnas: {df.shape[1]}")

    # precio: numeric_price ya viene tipado; usarlo como 'price'
    df["price"] = pd.to_numeric(df["numeric_price"], errors="coerce")
    df["average_rating"] = pd.to_numeric(df["average_rating"], errors="coerce")
    df["rating_number"] = pd.to_numeric(df["rating_number"], errors="coerce")

    # taxonomía real (categories[1]); main_category se descarta (es ruido)
    df["subcategory"] = df["categories"].apply(parse_categories)
    before = len(df)
    df = df[df["subcategory"].isin(REAL_SUBCATS)].copy()
    print(f"Filtrado a 8 subcategorías reales: {len(df):,} "
          f"({len(df)/before*100:.1f}% de las filas)")

    # texto para TF-IDF (title + features)
    df["features_text"] = df["features"].apply(join_list_text)
    df["desc_text"] = df["description"].apply(join_list_text)
    df["text"] = (df["title"].fillna("") + " " + df["features_text"]).str.strip()

    # details -> flags de presencia (features sparse, Parte IV)
    det = df["details"].apply(parse_details)
    for key in ["Brand", "Item Form", "Color", "Scent", "Skin Type", "Hair Type"]:
        df[f"has_{key.replace(' ', '_').lower()}"] = det.apply(
            lambda d: int(key in d and str(d.get(key)).strip() not in ("", "nan"))
        )

    df = df.reset_index(drop=True)
    RESULTS["dataset"] = {
        "rows": len(df),
        "subcategory_counts": df["subcategory"].value_counts().to_dict(),
        "price_null_rate": float(df["price"].isna().mean()),
    }
    print("Conteo por subcategoría:")
    print(df["subcategory"].value_counts())
    print(f"price nulo: {df['price'].isna().mean()*100:.2f}%")
    return df


# ============================================================
# PARTE I — Descriptiva y limpieza
# ============================================================

def part_I(df):
    banner("PARTE I · Estadística descriptiva, forma y outliers")
    desc = {}
    for col in ["average_rating", "rating_number", "price"]:
        d = M.descriptive_stats(df[col])
        desc[col] = d
        print(f"\n[{col}]  n={d['n']:,}  media={d['mean']:.3f}  mediana={d['median']:.3f}"
              f"  std={d['std']:.3f}")
        print(f"    Q1={d['Q1']:.3f}  Q3={d['Q3']:.3f}  IQR={d['IQR']:.3f}"
              f"  CV={d['cv']:.3f}")
        print(f"    skew(g1)={d['skewness']:.3f}  kurtosis_exc(g2)={d['kurtosis_excess']:.3f}"
              f"  min={d['min']:.3f}  max={d['max']:.3f}")

    # outliers (tres reglas) sobre rating_number y price
    outliers = {}
    for col in ["rating_number", "price"]:
        s = df[col].dropna()
        outliers[col] = {
            "tukey_1.5": int(M.tukey_outliers(s, 1.5).sum()),
            "tukey_3.0": int(M.tukey_outliers(s, 3.0).sum()),
            "zscore_3": int(M.zscore_outliers(s, 3.0).sum()),
            "mad_3.5": int(M.modified_zscore_outliers(s, 3.5).sum()),
        }
        print(f"\noutliers [{col}]: {outliers[col]}")

    # log1p reduce el sesgo
    logskew = {c: {"raw_skew": float(stats.skew(df[c].dropna())),
                   "log1p_skew": float(stats.skew(np.log1p(df[c].dropna())))}
               for c in ["rating_number", "price"]}
    print("\nEfecto de log1p sobre el sesgo:", logskew)

    miss = M.missing_rate(df[["average_rating", "rating_number", "price",
                              "features_text", "desc_text", "title"]])
    print("\nTasa de nulos:\n", miss)

    RESULTS["part_I"] = {"descriptive": desc, "outliers": outliers,
                         "log1p_skew": logskew,
                         "missing_rate": miss.to_dict()}
    _fig_distributions(df)


def _fig_distributions(df):
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    ax[0, 0].hist(df["average_rating"].dropna(), bins=40, color="#4C72B0")
    ax[0, 0].set_title("average_rating (sesgo negativo)")
    ax[0, 1].hist(df["rating_number"].dropna(), bins=60, color="#DD8452")
    ax[0, 1].set_title("rating_number (crudo, cola larga)")
    ax[0, 2].hist(np.log1p(df["rating_number"].dropna()), bins=60, color="#DD8452")
    ax[0, 2].set_title("log1p(rating_number)")
    ax[1, 0].hist(df["price"].dropna().clip(upper=df["price"].quantile(0.99)),
                  bins=60, color="#55A868")
    ax[1, 0].set_title("price (recortado p99)")
    ax[1, 1].hist(np.log1p(df["price"].dropna()), bins=60, color="#55A868")
    ax[1, 1].set_title("log1p(price)")
    df["subcategory"].value_counts().plot.barh(ax=ax[1, 2], color="#8172B3")
    ax[1, 2].set_title("Productos por subcategoría")
    plt.tight_layout()
    plt.savefig(OUT_FIG / "01_distributions.png", dpi=130)
    plt.close()
    print(f"[fig] {OUT_FIG/'01_distributions.png'}")


# ============================================================
# PARTE II — Relaciones e inferencia
# ============================================================

def part_II(df):
    banner("PARTE II · Correlaciones, normalidad, hipótesis")
    log_vol = np.log1p(df["rating_number"])
    pairs = {
        "price~rating": M.correlations(df["price"], df["average_rating"]),
        "price~log_volume": M.correlations(df["price"], log_vol),
        "rating~log_volume": M.correlations(df["average_rating"], log_vol),
        "review_count~rating_number": M.correlations(df["review_count"], df["rating_number"]),
    }
    for k, v in pairs.items():
        print(f"[{k}]  Pearson r={v['pearson_r']:.3f} (p={v['pearson_p']:.1e})"
              f"  Spearman ρ={v['spearman_rho']:.3f} (p={v['spearman_p']:.1e})")

    norm = {c: M.normality_test(df[c].sample(min(5000, len(df)), random_state=RNG))
            for c in ["average_rating", "rating_number", "price"]}
    print("\nnormalidad (D'Agostino, muestra 5k):",
          {k: v["is_normal"] for k, v in norm.items()})

    RESULTS["part_II"] = {"correlations": pairs, "normality": norm}
    _fig_corr(df, log_vol)


def _fig_corr(df, log_vol):
    sub = df.sample(min(6000, len(df)), random_state=RNG)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].scatter(sub["price"].clip(upper=df["price"].quantile(0.99)),
                  sub["average_rating"], alpha=0.15, s=8, color="#4C72B0")
    ax[0].set(xlabel="price (USD)", ylabel="average_rating",
              title="Precio vs rating (señal débil)")
    ax[1].scatter(np.log1p(sub["rating_number"]), sub["average_rating"],
                  alpha=0.15, s=8, color="#DD8452")
    ax[1].set(xlabel="log1p(rating_number)", ylabel="average_rating",
              title="Volumen vs rating")
    plt.tight_layout()
    plt.savefig(OUT_FIG / "02_correlations.png", dpi=130)
    plt.close()
    print(f"[fig] {OUT_FIG/'02_correlations.png'}")


# ============================================================
# PARTE III — Target de éxito
# ============================================================

def part_III(df):
    banner("PARTE III · Target de éxito + balance + sensibilidad")
    thr = M.subcategory_thresholds(df)
    print("Umbrales por subcategoría:\n", thr.round(3))
    df["success"] = M.label_success(df, thr)
    bal = M.class_balance(df["success"])
    print("\nBalance de clases:", bal)

    # sensibilidad del umbral (3 definiciones alternativas)
    base = df["success"].values
    sens = {}
    for name, (rq, vq) in {"p50/p60(base)": (0.50, 0.60),
                           "p50/p50": (0.50, 0.50),
                           "p50/p75": (0.50, 0.75),
                           "p60/p60": (0.60, 0.60)}.items():
        alt = M.threshold_sensitivity(df, "subcategory", "average_rating",
                                      "rating_number", rq, vq).values
        flip = float((alt != base).mean())
        sens[name] = {"positive_rate": float(alt.mean()), "flip_vs_base": flip}
    print("\nSensibilidad del umbral:", json.dumps(jsonable(sens), indent=2))

    # comparar precio y volumen entre éxito y no-éxito
    comp = {}
    for col in ["price", "rating_number", "average_rating"]:
        a = df.loc[df.success == 1, col].dropna().values
        b = df.loc[df.success == 0, col].dropna().values
        comp[col] = M.compare_groups(a, b)
        print(f"\n[{col}] éxito vs no-éxito: Welch p={comp[col]['welch_p']:.2e}"
              f"  d={comp[col]['cohens_d']:.3f}  MWU p={comp[col]['mannwhitney_p']:.2e}"
              f"  rank-biserial={comp[col]['rank_biserial']:.3f}")

    # asociación subcategoría ↔ éxito (chi² / Cramér's V)
    assoc = M.categorical_association(df["subcategory"], df["success"])
    print(f"\nsubcategoría ↔ éxito: χ²={assoc['chi2']:.1f} (p={assoc['p']:.1e})"
          f"  Cramér's V={assoc['cramers_v']:.3f}")

    RESULTS["part_III"] = {"thresholds": thr.round(4).to_dict(),
                           "class_balance": bal, "sensitivity": sens,
                           "group_compare": comp, "subcat_assoc": assoc}
    return thr


# ============================================================
# PARTE IV — Features
# ============================================================

def part_IV(df, thr):
    banner("PARTE IV · Ingeniería de features")
    df = M.add_price_fit(df, thr)
    print(f"price_fit: media={df['price_fit'].mean():.3f} "
          f"mediana={df['price_fit'].median():.3f} | "
          f"price_is_missing suma={int(df['price_is_missing'].sum())}")

    X_text, vec = M.build_text_features(df["text"], max_features=300)
    print(f"TF-IDF: {X_text.shape[1]} términos sobre {X_text.shape[0]:,} docs")

    detail_flags = ["has_brand", "has_item_form", "has_color",
                    "has_scent", "has_skin_type", "has_hair_type"]
    num_cols = ["price_fit", "price_is_missing"] + detail_flags
    X_num = df[num_cols].fillna(0.0).values
    subcat_ohe = pd.get_dummies(df["subcategory"])[REAL_SUBCATS]
    X = np.hstack([X_num, subcat_ohe.values, X_text.toarray()]).astype(np.float32)
    feat_names = (num_cols + [f"subcat::{c}" for c in REAL_SUBCATS]
                  + [f"tfidf::{t}" for t in vec.get_feature_names_out()])
    print(f"Matriz de features X: {X.shape}")

    RESULTS["part_IV"] = {"n_features": X.shape[1], "num_cols": num_cols,
                          "tfidf_terms": int(X_text.shape[1])}
    joblib.dump(vec, OUT_MODELS / "tfidf_vectorizer.pkl")
    (OUT_MODELS / "feature_names.json").write_text(json.dumps(feat_names))
    return df, X, feat_names, vec, num_cols


# ============================================================
# PARTE V + VI — Modelo, validación, evaluación
# ============================================================

def part_V_VI(df, X):
    banner("PARTE V–VI · RF + validación estratificada + métricas")
    y = df["success"].values
    clf, proba_oof = M.train_and_crossval(X, y, n_splits=5, random_state=RNG)
    rep = M.classification_report_full(y, proba_oof)
    ece = M.expected_calibration_error(y, proba_oof)
    print("Evaluación out-of-fold:")
    for k, v in rep.items():
        print(f"  {k}: {v}")
    print(f"  ECE (sin calibrar): {ece:.4f}")

    RESULTS["part_V_VI"] = {"report_oof": rep, "ece_uncalibrated": ece}
    np.save(OUT_PRED / "proba_oof.npy", proba_oof)
    _fig_eval(y, proba_oof)
    return clf, proba_oof, y


def _fig_eval(y, proba):
    from sklearn.metrics import roc_curve, precision_recall_curve
    from sklearn.calibration import calibration_curve
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    fpr, tpr, _ = roc_curve(y, proba)
    ax[0].plot(fpr, tpr); ax[0].plot([0, 1], [0, 1], "--", color="gray")
    ax[0].set(title=f"ROC (AUC={M.roc_auc_score(y, proba):.3f})",
              xlabel="FPR", ylabel="TPR")
    pr, rc, _ = precision_recall_curve(y, proba)
    ax[1].plot(rc, pr)
    ax[1].set(title=f"PR (AP={M.average_precision_score(y, proba):.3f})",
              xlabel="Recall", ylabel="Precision")
    frac, mean_pred = calibration_curve(y, proba, n_bins=10)
    ax[2].plot(mean_pred, frac, "o-"); ax[2].plot([0, 1], [0, 1], "--", color="gray")
    ax[2].set(title="Curva de calibración (sin calibrar)",
              xlabel="prob. predicha", ylabel="frac. real")
    plt.tight_layout()
    plt.savefig(OUT_FIG / "03_evaluation.png", dpi=130)
    plt.close()
    print(f"[fig] {OUT_FIG/'03_evaluation.png'}")


# ============================================================
# PARTE VII — Calibración e interpretabilidad
# ============================================================

def _honest_calibration(proba_oof, y):
    """Calibra 1D sobre las OOF por CV (isotónica y Platt) -> ECE/Brier honestos.
    Los MD exigen calibrar sobre predicciones out-of-fold, no in-sample."""
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import brier_score_loss
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG)
    iso_oof = np.zeros_like(proba_oof, dtype=float)
    sig_oof = np.zeros_like(proba_oof, dtype=float)
    for tr, te in skf.split(proba_oof, y):
        iso_oof[te] = IsotonicRegression(out_of_bounds="clip").fit(
            proba_oof[tr], y[tr]).predict(proba_oof[te])
        lr = LogisticRegression().fit(proba_oof[tr].reshape(-1, 1), y[tr])
        sig_oof[te] = lr.predict_proba(proba_oof[te].reshape(-1, 1))[:, 1]
    m = lambda p: {"ece": float(M.expected_calibration_error(y, p)),
                   "brier": float(brier_score_loss(y, p))}
    rep = {"uncalibrated": m(proba_oof), "isotonic_oof": m(iso_oof),
           "platt_sigmoid_oof": m(sig_oof)}
    rep["best_method"] = min(("isotonic_oof", "platt_sigmoid_oof"),
                             key=lambda k: rep[k]["ece"])
    return rep, iso_oof


def part_VII(clf, X, y, feat_names, proba_oof):
    banner("PARTE VII · Calibración honesta + importancia + SHAP")

    # calibración HONESTA sobre las probabilidades out-of-fold
    cal_report, _ = _honest_calibration(proba_oof, y)
    print("Calibración honesta (OOF):", json.dumps(jsonable(cal_report), indent=2))
    # artefacto desplegable: wrapper que calibra RF.predict_proba(features)
    calibrated = M.calibrate_model(clf, X, y, method="isotonic", cv=5)
    p_cal = calibrated.predict_proba(X)[:, 1]

    # importancia Gini (MDI) — todas las features, instantáneo
    mdi = pd.Series(clf.feature_importances_, index=feat_names).sort_values(ascending=False)
    print("\nTop-15 importancia Gini/MDI:")
    print(mdi.head(15).round(4))

    # importancia por permutación (método preferido por los MD) — subsample
    idx = np.random.RandomState(RNG).choice(len(X), size=min(6000, len(X)), replace=False)
    perm = M.permutation_feature_importance(clf, X[idx], y[idx], feat_names,
                                            n_repeats=5, random_state=RNG)
    print("\nTop-15 importancia por permutación (ROC-AUC drop):")
    print(perm.head(15).round(5))

    # SHAP — sanity check de aditividad; shap 0.50 + sklearn 1.7 falla numéricamente.
    # Los MD indican correr SHAP en Colab, no localmente/en la app.
    shap_summary, shap_status = {}, "no ejecutado"
    try:
        import shap
        s_idx = np.random.RandomState(RNG).choice(len(X), size=300, replace=False)
        Xs = np.ascontiguousarray(X[s_idx], dtype=np.float64)
        explainer = shap.TreeExplainer(clf, feature_perturbation="tree_path_dependent")
        sv = explainer.shap_values(Xs, check_additivity=False)
        sv1 = sv[1] if isinstance(sv, list) else (sv[..., 1] if sv.ndim == 3 else sv)
        base = explainer.expected_value
        base1 = base[1] if hasattr(base, "__len__") else base
        pred = clf.predict_proba(Xs)[:, 1]
        recon_err = float(np.abs(sv1.sum(1) + base1 - pred).max())
        if recon_err > 0.05:        # aditividad rota -> valores no confiables
            shap_status = (f"OMITIDO: additivity check falla (err={recon_err:.1e}); "
                           "incompatibilidad shap 0.50 + sklearn 1.7 con RF. "
                           "Correr en Colab (ver metrics.py). Se usa permutación.")
            print(f"\n[SHAP {shap_status}]")
        else:
            mean_abs = pd.Series(np.abs(sv1).mean(0), index=feat_names).sort_values(ascending=False)
            shap_summary = mean_abs.head(20).round(5).to_dict()
            shap_status = "ok"
            print("\nTop-15 SHAP (|valor| medio):")
            print(mean_abs.head(15).round(5))
    except Exception as e:
        shap_status = f"OMITIDO: {e}"
        print(f"\n[SHAP {shap_status}]")

    RESULTS["part_VII"] = {
        "honest_calibration_oof": cal_report,
        "mdi_top20": mdi.head(20).round(5).to_dict(),
        "permutation_top20": perm.head(20).round(6).to_dict(),
        "shap_status": shap_status,
        "shap_top20": jsonable(shap_summary),
    }
    _fig_importance(mdi, perm)
    return calibrated, p_cal


def _fig_importance(mdi, perm):
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    mdi.head(15)[::-1].plot.barh(ax=ax[0], color="#4C72B0")
    ax[0].set_title("Importancia Gini/MDI (top 15)")
    perm.head(15)[::-1].plot.barh(ax=ax[1], color="#C44E52")
    ax[1].set_title("Importancia por permutación (top 15)")
    plt.tight_layout()
    plt.savefig(OUT_FIG / "04_importance.png", dpi=130)
    plt.close()
    print(f"[fig] {OUT_FIG/'04_importance.png'}")


# ============================================================
# PARTE VIII — Métricas del dashboard + artefactos
# ============================================================

def part_VIII(df, X, clf, calibrated, p_cal, thr, num_cols):
    banner("PARTE VIII · Métricas del dashboard + artefactos")
    # incertidumbre RF
    unc = M.model_uncertainty(clf, X[np.random.RandomState(RNG).choice(len(X), 2000, replace=False)])
    print(f"Incertidumbre RF (muestra): media={unc.mean():.4f} max={unc.max():.4f}")

    # índice k-NN coseno (comparables) — sobre matriz de features
    nn = M.build_comparables_index(X)

    # referencia de densidad para saturación (submuestra del catálogo)
    ref_idx = np.arange(0, len(X), max(1, len(X) // 1500))
    dens_ref = np.array([(1 - nn.kneighbors(X[j].reshape(1, -1), n_neighbors=10)[0].ravel()).mean()
                         for j in ref_idx])

    # demostración sobre un producto de ejemplo
    ex = 0
    x0 = X[ex]
    idx, sims = M.find_comparables(nn, x0, k=5)
    sat = M.market_saturation(x0, nn, dens_ref)
    risk = M.risk_index(p_cal[ex], sat / 100, float(unc.mean()))
    score = M.success_score(p_cal[ex])
    print(f"\nProducto ejemplo: '{df['title'].iloc[ex][:60]}'")
    print(f"  score={score}  saturación={sat:.1f}  riesgo={risk}")
    print(f"  comparables idx={idx.tolist()} sims={np.round(sims,3).tolist()}")

    # precio sugerido: barrido sobre price_fit (columna 0)
    price_idx = num_cols.index("price_fit")
    grid = np.linspace(-2, 2, 17)
    best, curve = M.suggested_price(calibrated, x0, price_idx, grid)
    print(f"  price_fit óptimo={best:.2f} (curva {len(curve)} puntos)")

    # Wilson: tasa de éxito de los comparables
    succ = int(df["success"].iloc[idx].sum())
    lo, hi = M.wilson_interval(succ, len(idx))
    print(f"  éxito comparables={succ}/{len(idx)}  IC Wilson 95%=({lo:.2f}, {hi:.2f})")

    # ---- Artefactos para la app ----
    joblib.dump(clf, OUT_MODELS / "model.pkl")
    joblib.dump(calibrated, OUT_MODELS / "calibrator.pkl")
    joblib.dump(nn, OUT_MODELS / "knn_index.pkl")
    np.save(OUT_MODELS / "density_reference.npy", dens_ref)
    thr.to_json(OUT_MODELS / "subcategory_stats.json", indent=2)
    df[["parent_asin", "title", "subcategory", "price", "average_rating",
        "rating_number", "success"]].assign(p_cal=p_cal, score=(p_cal*100).round(1)) \
        .to_csv(OUT_PRED / "scored_catalog.csv", index=False)
    print("\nArtefactos guardados en output/models/:")
    for f in sorted(OUT_MODELS.glob("*")):
        print("  ", f.name)

    RESULTS["part_VIII"] = {
        "model_uncertainty_mean": float(unc.mean()),
        "rf_uncertainty_mean": float(unc.mean()),  # alias legacy for downstream readers
        "example_product": {
            "title": df["title"].iloc[ex][:80], "score": score,
            "saturation": sat, "risk": risk,
            "comparables_success": f"{succ}/{len(idx)}",
            "wilson_ci": [lo, hi], "best_price_fit": best,
        },
        "score_distribution": {
            "mean": float((p_cal * 100).mean()),
            "median": float(np.median(p_cal * 100)),
            "p10": float(np.percentile(p_cal * 100, 10)),
            "p90": float(np.percentile(p_cal * 100, 90)),
        },
    }
    _fig_pricecurve(curve)


def _fig_pricecurve(curve):
    xs, ys = zip(*curve)
    plt.figure(figsize=(8, 5))
    plt.plot(xs, ys, "o-", color="#4C72B0")
    plt.xlabel("price_fit (desviación robusta del precio)")
    plt.ylabel("success score (0-100)")
    plt.title("Barrido de precio → curva éxito vs precio (producto ejemplo)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_FIG / "05_price_curve.png", dpi=130)
    plt.close()
    print(f"[fig] {OUT_FIG/'05_price_curve.png'}")


# ============================================================
# MAIN
# ============================================================

def main():
    df = load_clean()
    part_I(df)
    part_II(df)
    thr = part_III(df)
    df, X, feat_names, vec, num_cols = part_IV(df, thr)
    clf, proba_oof, y = part_V_VI(df, X)
    calibrated, p_cal = part_VII(clf, X, y, feat_names, proba_oof)
    part_VIII(df, X, clf, calibrated, p_cal, thr, num_cols)

    (OUT_METRICS / "analysis_results.json").write_text(
        json.dumps(jsonable(RESULTS), indent=2, ensure_ascii=False))
    banner("PIPELINE COMPLETO")
    print(f"Resultados → {OUT_METRICS/'analysis_results.json'}")


if __name__ == "__main__":
    main()
