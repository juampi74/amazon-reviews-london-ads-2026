"""
inference.py — Motor de inferencia que reconstruye EXACTAMENTE el vector de features
del pipeline (316 dims) y produce todas las métricas del dashboard a partir de los
artefactos entrenados (Persona 2 → sirve a Persona 3 backend).

Feature order (feature_names.json):
    [price_fit, price_is_missing, has_brand, has_item_form, has_color,
     has_scent, has_skin_type, has_hair_type] + subcat one-hot (8) + tfidf (300)
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# artefactos en REPO/output/models  (../../output desde src/api)
ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "output" / "models"
PRED = ROOT / "output" / "predictions"
METRICS = ROOT / "output" / "metrics"

REAL_SUBCATS = [
    "Hair Care", "Skin Care", "Foot, Hand & Nail Care", "Makeup",
    "Tools & Accessories", "Fragrance", "Shave & Hair Removal", "Personal Care",
]
DETAIL_FLAGS = ["has_brand", "has_item_form", "has_color",
                "has_scent", "has_skin_type", "has_hair_type"]

MODEL_VERSION = "success-catboost-0.2.0"
DATASET_VERSION = "beauty-master-2026-07"


class Artifacts:
    """Carga perezosa y única de todos los artefactos."""
    def __init__(self):
        self.model = joblib.load(MODELS / "model.pkl")
        cal = joblib.load(MODELS / "calibrator_1d.pkl")
        self.cal_kind = cal["kind"]
        self.calibrator = cal["calibrator"]        # IsotonicRegression sobre RF proba
        self.knn = joblib.load(MODELS / "knn_index.pkl")
        self.vectorizer = joblib.load(MODELS / "tfidf_vectorizer.pkl")
        self.density_ref = np.load(MODELS / "density_reference.npy")
        self.feature_names = json.loads((MODELS / "feature_names.json").read_text())
        self.stats = json.loads((MODELS / "subcategory_stats.json").read_text())
        self.catalog = pd.read_csv(PRED / "scored_catalog.csv")
        try:
            self.metrics = json.loads((METRICS / "analysis_results.json").read_text())
        except FileNotFoundError:
            self.metrics = {}
        assert len(self.feature_names) == 316, "feature_names inesperado"

    # ---------- construcción del vector ----------
    def price_fit(self, subcat: str, price: float) -> float:
        med = self.stats["median_price"].get(subcat)
        iqr = self.stats["iqr_price"].get(subcat)
        if not iqr:
            return 0.0
        return (price - med) / iqr

    def build_vector(self, subcat: str, text: str, price: float,
                     detail_flags: dict | None = None) -> np.ndarray:
        flags = detail_flags or {}
        num = [self.price_fit(subcat, price), 0.0] + [
            float(flags.get(f, 0)) for f in DETAIL_FLAGS]
        ohe = [1.0 if subcat == s else 0.0 for s in REAL_SUBCATS]
        tfidf = self.vectorizer.transform([text or ""]).toarray()[0]
        vec = np.array(num + ohe + list(tfidf), dtype=np.float32)
        assert vec.shape[0] == 316
        return vec

    # ---------- predicciones ----------
    def calibrated_proba(self, vec: np.ndarray) -> tuple[float, float]:
        raw_p = float(self.model.predict_proba(vec.reshape(1, -1))[0, 1])
        p_cal = float(self.calibrator.predict([raw_p])[0])
        return p_cal, raw_p

    def uncertainty(self, vec: np.ndarray) -> float:
        """RF: tree std. CatBoost: probability margin ``1 - |2p - 1|``."""
        X = vec.reshape(1, -1)
        if hasattr(self.model, "estimators_"):
            per = np.array([est.predict_proba(X)[0, 1]
                            for est in self.model.estimators_])
            return float(per.std())
        p = float(self.model.predict_proba(X)[0, 1])
        return float(1.0 - abs(2.0 * p - 1.0))

    def comparables(self, vec: np.ndarray, k: int = 5) -> list[dict]:
        dist, idx = self.knn.kneighbors(vec.reshape(1, -1), n_neighbors=k)
        sims = 1 - dist.ravel()
        rows = self.catalog.iloc[idx.ravel()]
        out = []
        for (_, r), s in zip(rows.iterrows(), sims):
            out.append({
                "parent_asin": r["parent_asin"], "title": r["title"],
                "subcategory": r["subcategory"], "price": _num(r["price"]),
                "rating": _num(r["average_rating"]), "reviews": int(r["rating_number"]),
                "success": int(r["success"]), "similarity": round(float(s), 4),
            })
        return out

    def saturation(self, vec: np.ndarray, k: int = 10) -> float:
        dist, _ = self.knn.kneighbors(vec.reshape(1, -1), n_neighbors=k)
        local = (1 - dist.ravel()).mean()
        return float((self.density_ref < local).mean() * 100)

    def price_curve(self, subcat: str, text: str, flags: dict | None,
                    n: int = 25) -> list[dict]:
        p25 = self.stats["p25_price"].get(subcat, 5)
        p75 = self.stats["p75_price"].get(subcat, 40)
        lo = max(1.0, p25 * 0.5)
        hi = max(lo + 1, p75 * 1.8)
        grid = np.linspace(lo, hi, n)
        curve = []
        for p in grid:
            vec = self.build_vector(subcat, text, float(p), flags)
            p_cal, _ = self.calibrated_proba(vec)
            curve.append({"price": round(float(p), 2), "score": int(round(100 * p_cal))})
        return curve


def _num(x):
    try:
        return round(float(x), 2)
    except (TypeError, ValueError):
        return None


def wilson_interval(successes: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (round(float(max(0.0, centre - half)), 3),
            round(float(min(1.0, centre + half)), 3))


def risk_index(p_cal, sat01, unc01, weights=(0.5, 0.3, 0.2)) -> int:
    w1, w2, w3 = weights
    r = w1 * (1 - p_cal) + w2 * sat01 + w3 * unc01
    return int(round(100 * float(np.clip(r, 0, 1))))


@lru_cache(maxsize=1)
def get_artifacts() -> Artifacts:
    return Artifacts()


# ---------- función de alto nivel: análisis completo ----------

def analyze(subcategory: str, title: str, description: str, price: float,
            risk_preference: str = "balanced",
            detail_flags: dict | None = None) -> dict:
    if subcategory not in REAL_SUBCATS:
        raise ValueError(f"subcategory debe ser una de {REAL_SUBCATS}")
    A = get_artifacts()
    text = f"{title or ''} {description or ''}".strip()
    vec = A.build_vector(subcategory, text, price, detail_flags)

    p_cal, _raw_p = A.calibrated_proba(vec)
    unc = A.uncertainty(vec)
    sat = A.saturation(vec)
    comps = A.comparables(vec, k=5)
    curve = A.price_curve(subcategory, text, detail_flags)

    score = int(round(100 * p_cal))
    risk = risk_index(p_cal, sat / 100, unc)
    succ_comp = sum(c["success"] for c in comps)
    wlo, whi = wilson_interval(succ_comp, len(comps))
    best = max(curve, key=lambda c: c["score"])
    p25 = A.stats["p25_price"].get(subcategory)
    p75 = A.stats["p75_price"].get(subcategory)

    confidence = "high" if unc < 0.20 else "medium" if unc < 0.30 else "low"
    return {
        "success": {"score": score, "probability": round(p_cal, 4),
                    "uncertainty": round(unc, 4), "confidence": confidence,
                    "source_type": "model"},
        "risk": {"index": risk,
                 "components": {"downside": int(round(100 * (1 - p_cal))),
                                "saturation": int(round(sat)),
                                "uncertainty": int(round(100 * unc))},
                 "source_type": "model"},
        "saturation": {"value": round(sat, 1), "source_type": "model"},
        "recommended_price": best["price"],
        "price_range": [_num(p25), _num(p75)],
        "price_curve": curve,
        "comparables": comps,
        "comparables_success": {"successes": succ_comp, "n": len(comps),
                                "wilson_ci_95": [wlo, whi]},
        "model_version": MODEL_VERSION,
        "dataset_version": DATASET_VERSION,
        "limitations": [
            "El 'éxito' es un proxy (rating+volumen), no ventas reales.",
            "Señal modesta: ROC-AUC ~0.71. Los scores son probabilidades con incertidumbre.",
            "Beneficio/forecast/demografía no salen del modelo (capa de supuestos/externa).",
        ],
    }
