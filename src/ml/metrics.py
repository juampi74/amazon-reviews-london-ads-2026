"""
metrics.py — Módulo consolidado de fórmulas y métricas
Product Success Predictor · Beauty & Personal Care (Amazon Reviews'23)

Reúne TODAS las funciones descritas en FORMULAS_AND_METRICS.md, Partes I–VIII:
  I    Estadística descriptiva y limpieza
  II   Relaciones e inferencia (EDA)
  III  Diseño del target ("éxito")
  IV   Ingeniería de features
  V    Modelado y validación
  VI   Métricas de evaluación
  VII  Calibración e interpretabilidad
  VIII Métricas derivadas del dashboard

Ejecutar `python metrics.py` corre la demo end-to-end con datos sintéticos
(al final del archivo) para verificar que cada fórmula funciona.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    accuracy_score, roc_auc_score, average_precision_score,
    brier_score_loss, log_loss,
)

try:
    from catboost import CatBoostClassifier
except ImportError as e:  # pragma: no cover
    CatBoostClassifier = None
    _CATBOOST_IMPORT_ERROR = e
else:
    _CATBOOST_IMPORT_ERROR = None


# ============================================================
# PARTE I — Estadística descriptiva y limpieza
# ============================================================

def descriptive_stats(x: pd.Series) -> dict:
    """Estadística descriptiva completa de una variable numérica."""
    x = x.dropna()
    q1, q3 = x.quantile(0.25), x.quantile(0.75)
    mean = x.mean()
    return {
        "n": int(x.size),
        "mean": mean,
        "median": x.median(),
        "std": x.std(ddof=1),                 # ddof=1 -> corrección de Bessel
        "var": x.var(ddof=1),
        "min": x.min(), "max": x.max(),
        "Q1": q1, "Q3": q3, "IQR": q3 - q1,
        "cv": x.std(ddof=1) / mean if mean else np.nan,   # coef. de variación
        "skewness": stats.skew(x),            # g1 (Fisher)
        "kurtosis_excess": stats.kurtosis(x), # g2 (exceso, ya resta 3)
        "p60": x.quantile(0.60),
    }


def log1p_transform(x: pd.Series) -> pd.Series:
    """ln(1+x): normaliza colas pesadas y admite ceros."""
    return np.log1p(x)


def tukey_outliers(x: pd.Series, k: float = 1.5) -> pd.Series:
    """Cercas de Tukey (robusto). k=1.5 leve, k=3 extremo. Máscara booleana."""
    q1, q3 = x.quantile(0.25), x.quantile(0.75)
    iqr = q3 - q1
    return (x < q1 - k * iqr) | (x > q3 + k * iqr)


def zscore_outliers(x: pd.Series, thresh: float = 3.0) -> pd.Series:
    """Z-score clásico. Solo apropiado si la variable es aprox. normal."""
    z = (x - x.mean()) / x.std(ddof=1)
    return z.abs() > thresh


def modified_zscore_outliers(x: pd.Series, thresh: float = 3.5) -> pd.Series:
    """Z-score modificado por MAD: robusto a outliers en media y std."""
    med = x.median()
    mad = (x - med).abs().median()
    if mad == 0:
        return pd.Series(False, index=x.index)
    m = 0.6745 * (x - med) / mad
    return m.abs() > thresh


def missing_rate(df: pd.DataFrame) -> pd.Series:
    """Fracción de nulos por columna."""
    return df.isna().mean().sort_values(ascending=False)


# ============================================================
# PARTE II — Relaciones e inferencia (EDA)
# ============================================================

def correlations(x: pd.Series, y: pd.Series) -> dict:
    """Pearson (lineal) y Spearman (monótona, robusta) con sus p-valores."""
    x = pd.Series(np.asarray(x)); y = pd.Series(np.asarray(y))
    mask = x.notna() & y.notna()
    xr, yr = x[mask], y[mask]
    r_p, p_p = stats.pearsonr(xr, yr)
    r_s, p_s = stats.spearmanr(xr, yr)
    return {"pearson_r": r_p, "pearson_p": p_p,
            "spearman_rho": r_s, "spearman_p": p_s}


def point_biserial(binary: pd.Series, continuous: pd.Series) -> dict:
    """Asociación entre el target binario y una variable continua."""
    binary = pd.Series(np.asarray(binary)); continuous = pd.Series(np.asarray(continuous))
    mask = binary.notna() & continuous.notna()
    r, p = stats.pointbiserialr(binary[mask], continuous[mask])
    return {"r_pb": r, "p": p}


def normality_test(x: pd.Series) -> dict:
    """D'Agostino-Pearson. H0: normal. p<0.05 => NO normal => usar no paramétrico."""
    x = pd.Series(np.asarray(x)).dropna()
    stat, p = stats.normaltest(x)
    return {"statistic": stat, "p": p, "is_normal": p >= 0.05}


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Tamaño del efecto para diferencia de medias (t-test)."""
    a, b = np.asarray(a), np.asarray(b)
    n1, n2 = len(a), len(b)
    sp = np.sqrt(((n1-1)*a.var(ddof=1) + (n2-1)*b.var(ddof=1)) / (n1+n2-2))
    return (a.mean() - b.mean()) / sp


def compare_groups(success_group: np.ndarray, fail_group: np.ndarray) -> dict:
    """Compara una variable entre exitosos vs no exitosos.
    Reporta test paramétrico y no paramétrico + tamaños de efecto."""
    t_stat, t_p = stats.ttest_ind(success_group, fail_group, equal_var=False)  # Welch
    u_stat, u_p = stats.mannwhitneyu(success_group, fail_group, alternative="two-sided")
    n1, n2 = len(success_group), len(fail_group)
    rank_biserial = 1 - (2 * u_stat) / (n1 * n2)
    return {
        "welch_t": t_stat, "welch_p": t_p,
        "cohens_d": cohens_d(success_group, fail_group),
        "mannwhitney_U": u_stat, "mannwhitney_p": u_p,
        "rank_biserial": rank_biserial,
    }


def categorical_association(cat1: pd.Series, cat2: pd.Series) -> dict:
    """Chi-cuadrado + Cramér's V para dos variables categóricas."""
    table = pd.crosstab(cat1, cat2)
    chi2, p, dof, _ = stats.chi2_contingency(table)
    n = table.values.sum()
    r, k = table.shape
    cramers_v = np.sqrt(chi2 / (n * (min(r, k) - 1)))
    return {"chi2": chi2, "p": p, "dof": dof, "cramers_v": cramers_v}


# ============================================================
# PARTE III — Diseño del target: "éxito"
# ============================================================

def subcategory_thresholds(df: pd.DataFrame,
                           subcat="subcategory",
                           rating="average_rating",
                           volume="rating_number",
                           price="price") -> pd.DataFrame:
    """Umbrales poblacionales por subcategoría: se calculan UNA vez sobre el
    dataset completo y se exportan como artefacto (subcategory_stats.json)."""
    g = df.groupby(subcat)
    stats_tbl = pd.DataFrame({
        "median_rating":  g[rating].median(),
        "p60_log_volume": g[volume].apply(lambda s: np.log1p(s).quantile(0.60)),
        # stats de precio SOLO sobre el subconjunto con precio (~37% en crudo)
        "median_price":   g[price].median(),
        "p25_price":      g[price].quantile(0.25),
        "p75_price":      g[price].quantile(0.75),
    })
    stats_tbl["iqr_price"] = stats_tbl["p75_price"] - stats_tbl["p25_price"]
    return stats_tbl


def label_success(df: pd.DataFrame, thresholds: pd.DataFrame,
                  subcat="subcategory", rating="average_rating",
                  volume="rating_number") -> pd.Series:
    """Aplica la fórmula de éxito fila por fila usando el umbral de su subcategoría."""
    med = df[subcat].map(thresholds["median_rating"])
    p60 = df[subcat].map(thresholds["p60_log_volume"])
    quality  = df[rating] >= med
    traction = np.log1p(df[volume]) >= p60
    return (quality & traction).astype(int)


def class_balance(y: pd.Series) -> dict:
    y = pd.Series(np.asarray(y))
    counts = y.value_counts().to_dict()
    return {"positive_rate": y.mean(),
            "counts": {int(k): int(v) for k, v in counts.items()},
            "imbalance_ratio": counts.get(0, 0) / max(counts.get(1, 1), 1)}


def threshold_sensitivity(df, subcat, rating, volume,
                          rating_q=0.50, volume_q=0.60) -> pd.Series:
    """Etiqueta con un umbral alternativo (para el análisis de sensibilidad)."""
    g = df.groupby(subcat)
    med = df[subcat].map(g[rating].quantile(rating_q))
    p_v = df[subcat].map(g[volume].apply(lambda s: np.log1p(s).quantile(volume_q)))
    return ((df[rating] >= med) & (np.log1p(df[volume]) >= p_v)).astype(int)


# ============================================================
# PARTE IV — Ingeniería de features
# ============================================================

def price_fit(price, median_c, iqr_c):
    """Desviación robusta del precio respecto a su subcategoría."""
    return (price - median_c) / iqr_c if iqr_c else 0.0


def add_price_fit(df, thresholds, subcat="subcategory", price="price"):
    med = df[subcat].map(thresholds["median_price"])
    iqr = df[subcat].map(thresholds["iqr_price"]).replace(0, np.nan)
    out = df.copy()
    out["price_fit"] = (df[price] - med) / iqr
    out["price_is_missing"] = df[price].isna().astype(int)   # bandera de nulo
    return out


def build_text_features(text_series, max_features=300):
    """TF-IDF de title+features. Devuelve matriz dispersa y el vectorizador."""
    vec = TfidfVectorizer(max_features=max_features, stop_words="english",
                          ngram_range=(1, 2), sublinear_tf=True)
    X_text = vec.fit_transform(text_series.fillna(""))
    return X_text, vec


def scale_numeric(X_train, X_test):
    """StandardScaler ajustado SOLO en train (evita leakage)."""
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test), scaler


# ============================================================
# PARTE V — Modelado y validación
# ============================================================

def make_success_classifier(random_state=42, iterations=500):
    """CatBoost product classifier (replaces Random Forest for Phase 1)."""
    if CatBoostClassifier is None:
        raise ImportError(
            "catboost is required. Install with: pip install catboost>=1.2"
        ) from _CATBOOST_IMPORT_ERROR
    return CatBoostClassifier(
        iterations=iterations,
        depth=6,
        learning_rate=0.05,
        loss_function="Logloss",
        auto_class_weights="Balanced",
        random_seed=random_state,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )


def train_and_crossval(X, y, n_splits=5, random_state=42):
    """CatBoost with stratified CV. Returns fitted model + out-of-fold probs."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    clf = make_success_classifier(random_state=random_state)
    proba_oof = cross_val_predict(clf, X, y, cv=skf, method="predict_proba")[:, 1]
    clf.fit(X, y)                 # modelo final sobre todo el set (a serializar)
    return clf, proba_oof


# ============================================================
# PARTE VI — Métricas de evaluación
# ============================================================

def classification_report_full(y_true, proba, threshold=0.5) -> dict:
    """Todas las métricas de clasificación en un punto de corte."""
    y_pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "confusion": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_true, proba),
        "pr_auc":    average_precision_score(y_true, proba),
        "brier":     brier_score_loss(y_true, proba),
        "log_loss":  log_loss(y_true, np.clip(proba, 1e-9, 1 - 1e-9)),
    }


def expected_calibration_error(y_true, proba, n_bins=10) -> float:
    """ECE: promedio ponderado |precisión - confianza| por bin de probabilidad."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    bins = np.linspace(0, 1, n_bins + 1)
    ece, n = 0.0, len(proba)
    for i in range(n_bins):
        mask = (proba > bins[i]) & (proba <= bins[i + 1])
        if mask.sum() == 0:
            continue
        acc  = y_true[mask].mean()
        conf = proba[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return ece


# ============================================================
# PARTE VII — Calibración e interpretabilidad
# ============================================================

def calibrate_model(clf, X, y, method="isotonic", cv=5):
    """Envuelve el modelo con calibración (Platt='sigmoid' o 'isotonic')."""
    calibrated = CalibratedClassifierCV(clf, method=method, cv=cv)
    calibrated.fit(X, y)
    return calibrated


def permutation_feature_importance(clf, X, y, feature_names, n_repeats=10, random_state=42):
    """Importancia por permutación: caída de la métrica al barajar cada feature."""
    r = permutation_importance(clf, X, y, n_repeats=n_repeats,
                               random_state=random_state, scoring="roc_auc")
    return (pd.Series(r.importances_mean, index=feature_names)
            .sort_values(ascending=False))


def shap_values(clf, X_sample):
    """Valores SHAP (interpretabilidad local). Requiere `pip install shap`."""
    import shap
    explainer = shap.TreeExplainer(clf)
    return explainer.shap_values(X_sample)


# ============================================================
# PARTE VIII — Métricas derivadas del dashboard
# ============================================================

def success_score(p_cal) -> int:
    """Probabilidad calibrada -> score 0-100."""
    return int(round(100 * float(p_cal)))


def model_uncertainty(clf, X) -> np.ndarray:
    """Per-row prediction uncertainty in ~[0, 1].

    - RandomForest (legacy): std across trees.
    - CatBoost: probability margin ``1 - |2p - 1|`` (1 at p=0.5, 0 at extremes).
      Late-stage boost std is near-zero for this setup and underweights risk.
    """
    X = np.asarray(X)
    if hasattr(clf, "estimators_"):
        per_tree = np.stack(
            [est.predict_proba(X)[:, 1] for est in clf.estimators_], axis=0)
        return per_tree.std(axis=0)

    p = clf.predict_proba(X)[:, 1]
    return 1.0 - np.abs(2.0 * p - 1.0)


def rf_uncertainty(clf, X) -> np.ndarray:
    """Alias kept for callers; prefer model_uncertainty."""
    return model_uncertainty(clf, X)


def build_comparables_index(V_catalog):
    """Índice k-NN por similitud coseno (a serializar como artefacto)."""
    nn = NearestNeighbors(n_neighbors=10, metric="cosine")
    nn.fit(V_catalog)
    return nn


def find_comparables(nn, x_vec, k=4):
    """(índices, similitudes) de los k productos más parecidos. sim = 1 - dist coseno."""
    dist, idx = nn.kneighbors(np.asarray(x_vec).reshape(1, -1), n_neighbors=k)
    sims = 1 - dist.ravel()
    return idx.ravel(), sims


def market_saturation(x_vec, nn, density_reference, k=10) -> float:
    """Percentil (0-100) de la densidad local del producto vs. el catálogo."""
    dist, _ = nn.kneighbors(np.asarray(x_vec).reshape(1, -1), n_neighbors=k)
    local_density = (1 - dist.ravel()).mean()
    return float((density_reference < local_density).mean() * 100)


def risk_index(p_cal, saturation_0_1, uncertainty_0_1, weights=(0.5, 0.3, 0.2)) -> int:
    """Índice de riesgo compuesto 0-100 (decisión de diseño, no probabilidad)."""
    w1, w2, w3 = weights
    r = w1 * (1 - p_cal) + w2 * saturation_0_1 + w3 * uncertainty_0_1
    return int(round(100 * float(np.clip(r, 0, 1))))


def suggested_price(model, base_row, price_col_idx, price_grid):
    """Barrido de precio: recalcula el score variando solo el precio.
    Devuelve (precio_optimo, curva[(precio, score)])."""
    curve = []
    for p in price_grid:
        row = np.asarray(base_row, dtype=float).copy()
        row[price_col_idx] = p
        score = 100 * model.predict_proba(row.reshape(1, -1))[0, 1]
        curve.append((float(p), float(score)))
    best_price = max(curve, key=lambda t: t[1])[0]
    return best_price, curve


def wilson_interval(successes: int, n: int, z: float = 1.96):
    """IC de Wilson para una proporción (tasa de éxito de comparables)."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0, centre - half), min(1, centre + half))


# ============================================================
# DEMO END-TO-END (datos sintéticos) — verificación
# ============================================================

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    N = 4000
    subcats = ["Hair Care", "Skin Care", "Makeup", "Fragrance"]

    df = pd.DataFrame({
        "subcategory":    rng.choice(subcats, N, p=[0.45, 0.30, 0.15, 0.10]),
        "average_rating": np.clip(rng.normal(4.06, 0.6, N), 1, 5),
        "rating_number":  rng.lognormal(3.0, 1.5, N).astype(int),
    })
    price = rng.lognormal(2.8, 0.7, N)
    price[rng.random(N) < 0.63] = np.nan
    df["price"] = price
    df["text"] = rng.choice(
        ["hydrating serum frizz control", "anti aging cream vitamin c",
         "matte lipstick long lasting", "floral fragrance eau de parfum"], N)

    print("== I. Descriptiva (rating_number, sesgado) ==")
    print(descriptive_stats(df["rating_number"]))
    print("outliers Tukey:", int(tukey_outliers(df["rating_number"]).sum()))

    print("\n== II. Relaciones ==")
    print(correlations(np.log1p(df["rating_number"]), df["average_rating"]))
    print("normalidad rating_number:", normality_test(df["rating_number"])["is_normal"])

    print("\n== III. Target de éxito ==")
    thr = subcategory_thresholds(df)
    df["success"] = label_success(df, thr)
    print(class_balance(df["success"]))

    priced = df.dropna(subset=["price"])
    print("comparación de precio éxito vs no éxito:",
          compare_groups(priced.loc[priced.success == 1, "price"].values,
                         priced.loc[priced.success == 0, "price"].values))

    print("\n== IV-V. Features + modelo ==")
    df = add_price_fit(df, thr)
    X_text, vec = build_text_features(df["text"], max_features=20)
    X_num = df[["price_fit", "price_is_missing"]].fillna(0).values
    subcat_ohe = pd.get_dummies(df["subcategory"]).values
    X = np.hstack([X_num, subcat_ohe, X_text.toarray()])
    y = df["success"].values
    clf, proba_oof = train_and_crossval(X, y)

    print("\n== VI. Evaluación (out-of-fold) ==")
    for k, v in classification_report_full(y, proba_oof).items():
        print(f"  {k}: {v}")
    print("  ECE:", round(expected_calibration_error(y, proba_oof), 4))

    print("\n== VII. Calibración + importancia ==")
    calibrated = calibrate_model(clf, X, y, method="isotonic")
    p_cal = calibrated.predict_proba(X)[:, 1]
    print("  ECE tras calibrar:", round(expected_calibration_error(y, p_cal), 4))

    print("\n== VIII. Métricas del dashboard (primer producto) ==")
    x0 = X[0]
    print("  success_score:", success_score(p_cal[0]))
    u = rf_uncertainty(clf, X)[0]
    nn = build_comparables_index(X)
    idx, sims = find_comparables(nn, x0, k=4)
    sat = market_saturation(x0, nn,
            density_reference=np.array([(1 - nn.kneighbors(X[j].reshape(1, -1),
                                         n_neighbors=10)[0].ravel()).mean()
                                        for j in range(0, N, 50)]))
    print("  comparables idx:", idx, "sims:", np.round(sims, 3))
    print("  saturación:", round(sat, 1),
          "| riesgo:", risk_index(p_cal[0], sat/100, u))
    best, curve = suggested_price(calibrated, x0, 0, price_grid=np.linspace(-2, 2, 9))
    print("  precio óptimo (en unidades de price_fit):", round(best, 2))
    print("  IC Wilson de 'éxito de comparables' (ej. 3/4):", wilson_interval(3, 4))

    print("\nOK — todas las fórmulas corrieron.")
