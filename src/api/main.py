"""
main.py — FastAPI que expone el modelo real al dashboard (Launchly / AmazonProject.html).

Endpoints (contrato de docs/09_API_DATABASE.md, P0):
    GET  /health                  liveness
    GET  /v1/models/current       versión del modelo + métricas de validación
    POST /v1/predict/success      score calibrado + incertidumbre
    POST /v1/comparables          k-NN + saturación
    POST /v1/price-scenarios      curva de precio (barrido)
    POST /v1/analyses             análisis completo (lo que consume la página)

Ejecutar:
    cd REPO/src/api
    uvicorn main:app --reload --port 8000
Docs interactivas: http://localhost:8000/docs
"""
from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import inference as I

app = FastAPI(
    title="Product Success Predictor API",
    version=I.MODEL_VERSION,
    description="Sirve el modelo calibrado de éxito de producto (Beauty & Personal Care).",
)

# CORS: permite que el HTML (file://, localhost, o Streamlit) llame a la API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # en producción, restringir al dominio del front
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ----------------------- Esquemas -----------------------

class DetailFlags(BaseModel):
    has_brand: int = 0
    has_item_form: int = 0
    has_color: int = 0
    has_scent: int = 0
    has_skin_type: int = 0
    has_hair_type: int = 0


class ProductInput(BaseModel):
    subcategory: str = Field(..., examples=["Skin Care"])
    title: str = Field("", examples=["Hydrating Vitamin C Serum"])
    description: str = Field("", examples=["Vegan, lightweight, fast absorbing"])
    price: float = Field(..., gt=0, examples=[30])
    risk_preference: str = Field("balanced", examples=["cautious", "balanced", "bold"])
    detail_flags: Optional[DetailFlags] = None

    def flags(self):
        return self.detail_flags.model_dump() if self.detail_flags else None


# ----------------------- Startup -----------------------

@app.on_event("startup")
def _warm():
    I.get_artifacts()          # carga los artefactos una sola vez


def _guard_subcat(subcat: str):
    if subcat not in I.REAL_SUBCATS:
        raise HTTPException(
            status_code=422,
            detail=f"subcategory inválida. Debe ser una de: {I.REAL_SUBCATS}")


# ----------------------- Endpoints -----------------------

@app.get("/health")
def health():
    return {"status": "ok", "model_version": I.MODEL_VERSION}


@app.get("/v1/models/current")
def models_current():
    A = I.get_artifacts()
    m = A.metrics.get("part_V_VI", {})
    val = {}
    try:
        import json
        vpath = I.METRICS / "model_validation.json"
        if vpath.exists():
            val = json.loads(vpath.read_text())
    except Exception:
        val = {}
    return {
        "model_version": I.MODEL_VERSION,
        "dataset_version": I.DATASET_VERSION,
        "algorithm": "CatBoost (500) + isotonic calibration",
        "features": len(A.feature_names),
        "subcategories": I.REAL_SUBCATS,
        "cv_metrics": m.get("report_oof", {}),
        "calibration": A.metrics.get("part_VII", {}).get("honest_calibration_oof", {}),
        "validation": {k: val.get(k) for k in ("T1_holdout", "T3_label_permutation",
                                               "T4_stability") if k in val},
    }


@app.post("/v1/predict/success")
def predict_success(inp: ProductInput):
    _guard_subcat(inp.subcategory)
    A = I.get_artifacts()
    text = f"{inp.title} {inp.description}".strip()
    vec = A.build_vector(inp.subcategory, text, inp.price, inp.flags())
    p_cal, rf_p = A.calibrated_proba(vec)
    unc = A.uncertainty(vec)
    return {
        "score": int(round(100 * p_cal)),
        "probability": round(p_cal, 4),
        "raw_probability": round(rf_p, 4),
        "uncertainty": round(unc, 4),
        "confidence": "high" if unc < 0.20 else "medium" if unc < 0.30 else "low",
        "model_version": I.MODEL_VERSION,
        "source_type": "model",
    }


@app.post("/v1/comparables")
def comparables(inp: ProductInput):
    _guard_subcat(inp.subcategory)
    A = I.get_artifacts()
    text = f"{inp.title} {inp.description}".strip()
    vec = A.build_vector(inp.subcategory, text, inp.price, inp.flags())
    comps = A.comparables(vec, k=5)
    sat = A.saturation(vec)
    succ = sum(c["success"] for c in comps)
    return {
        "comparables": comps,
        "saturation": round(sat, 1),
        "comparables_success": {
            "successes": succ, "n": len(comps),
            "wilson_ci_95": I.wilson_interval(succ, len(comps))},
        "source_type": "model",
    }


@app.post("/v1/price-scenarios")
def price_scenarios(inp: ProductInput):
    _guard_subcat(inp.subcategory)
    A = I.get_artifacts()
    text = f"{inp.title} {inp.description}".strip()
    curve = A.price_curve(inp.subcategory, text, inp.flags())
    best = max(curve, key=lambda c: c["score"])
    return {
        "price_curve": curve,
        "recommended_price": best["price"],
        "recommended_score": best["score"],
        "current_price": inp.price,
        "source_type": "model",
    }


@app.post("/v1/analyses")
def analyses(inp: ProductInput):
    _guard_subcat(inp.subcategory)
    try:
        return I.analyze(inp.subcategory, inp.title, inp.description,
                         inp.price, inp.risk_preference, inp.flags())
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
