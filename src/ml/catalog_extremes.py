"""
catalog_extremes.py — Tetos de sucesso e produtos top: RF vs CatBoost.

Para cada modelo reporta:
  • máx / percentis do score (OOF honesto e calibrado tipo dashboard)
  • produto(s) com maior taxa de sucesso prevista
  • top-N global e por subcategoria
  • quantos produtos batem score ≥ 70/80/90/100

Treina RF no mesmo X/y do pipeline; CatBoost reutiliza model.pkl se existir,
senão treina de novo.

Uso:
  cd REPO/src/ml && python3 catalog_extremes.py
  python3 catalog_extremes.py --top 15
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

import metrics as M
import run_pipeline as P

warnings.filterwarnings("ignore")
RNG = 42
REPO = Path(__file__).resolve().parents[2]
OUT_PRED = REPO / "output" / "predictions"
OUT_METRICS = REPO / "output" / "metrics"
MODEL_PKL = REPO / "output" / "models" / "model.pkl"


def _rf():
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=RNG,
        n_jobs=-1,
    )


def _oof_and_fit(clf, X, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG)
    proba_oof = cross_val_predict(clf, X, y, cv=skf, method="predict_proba")[:, 1]
    clf.fit(X, y)
    return clf, proba_oof


def _isotonic_oof(proba_oof, y):
    """Calibração isotónica honesta fold-a-fold → probs calibradas OOF."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG)
    out = np.zeros_like(proba_oof, dtype=float)
    for tr, te in skf.split(proba_oof, y):
        iso = IsotonicRegression(out_of_bounds="clip")
        out[te] = iso.fit(proba_oof[tr], y[tr]).predict(proba_oof[te])
    return out


def _score_stats(scores: np.ndarray) -> dict:
    s = np.asarray(scores, dtype=float)
    return {
        "min": float(s.min()),
        "max": float(s.max()),
        "mean": float(s.mean()),
        "median": float(np.median(s)),
        "p90": float(np.percentile(s, 90)),
        "p99": float(np.percentile(s, 99)),
        "n_ge_70": int((s >= 70).sum()),
        "n_ge_80": int((s >= 80).sum()),
        "n_ge_90": int((s >= 90).sum()),
        "n_ge_100": int((s >= 99.5).sum()),
    }


def _top_rows(df: pd.DataFrame, col: str, n: int) -> list[dict]:
    cols = [
        "parent_asin", "title", "subcategory", "price",
        "average_rating", "rating_number", "success", col,
    ]
    top = df.nlargest(n, col)[cols].copy()
    top["title"] = top["title"].astype(str).str.slice(0, 80)
    return top.round(4).to_dict(orient="records")


def _print_block(name: str, stats: dict, tops: list[dict], score_col: str) -> None:
    print(f"\n{'=' * 72}")
    print(f"{name}")
    print("=" * 72)
    print(
        f"  máx={stats['max']:.1f}  p99={stats['p99']:.1f}  "
        f"p90={stats['p90']:.1f}  mediana={stats['median']:.1f}  média={stats['mean']:.1f}"
    )
    print(
        f"  n≥70={stats['n_ge_70']}  ≥80={stats['n_ge_80']}  "
        f"≥90={stats['n_ge_90']}  ≈100={stats['n_ge_100']}"
    )
    print(f"\n  Top produtos por {score_col}:")
    for i, r in enumerate(tops, 1):
        print(
            f"  {i:2d}. score={r[score_col]:5.1f}  success={r['success']}  "
            f"[{r['subcategory']}]  {r['title']}"
        )
        print(
            f"      asin={r['parent_asin']}  price=${r['price']:.2f}  "
            f"rating={r['average_rating']}  n_reviews={int(r['rating_number'])}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=10, help="Top-N produtos")
    args = ap.parse_args()

    print("Carregando dados + features (mesmo pipeline)…")
    df = P.load_clean()
    thr = P.part_III(df)
    df, X, _feat_names, _vec, _num = P.part_IV(df, thr)
    y = df["success"].values

    # ---- CatBoost ----
    print("\nCatBoost: OOF + modelo final…")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG)
    cb_fresh = M.make_success_classifier(random_state=RNG)
    cb_oof = cross_val_predict(
        cb_fresh, X, y, cv=skf, method="predict_proba"
    )[:, 1]
    cb = None
    if MODEL_PKL.exists():
        loaded = joblib.load(MODEL_PKL)
        if loaded.__class__.__name__ == "CatBoostClassifier":
            cb = loaded
    if cb is None:
        cb = M.make_success_classifier(random_state=RNG)
        cb.fit(X, y)

    cb_oof_cal = _isotonic_oof(cb_oof, y)
    cb_insample = cb.predict_proba(X)[:, 1]
    # calibrador 1D treinado nos OOF (como dashboard honesto)
    cb_iso = IsotonicRegression(out_of_bounds="clip").fit(cb_oof, y)
    cb_dash = cb_iso.predict(cb_insample)

    # ---- Random Forest ----
    print("Random Forest: OOF + modelo final (pode demorar alguns minutos)…")
    rf = _rf()
    rf, rf_oof = _oof_and_fit(rf, X, y)
    rf_oof_cal = _isotonic_oof(rf_oof, y)
    rf_insample = rf.predict_proba(X)[:, 1]
    rf_iso = IsotonicRegression(out_of_bounds="clip").fit(rf_oof, y)
    rf_dash = rf_iso.predict(rf_insample)

    # scores 0–100
    df = df.copy()
    df["score_cb_oof"] = (cb_oof_cal * 100).round(1)
    df["score_rf_oof"] = (rf_oof_cal * 100).round(1)
    df["score_cb_dash"] = (cb_dash * 100).round(1)
    df["score_rf_dash"] = (rf_dash * 100).round(1)
    df["p_cb_oof"] = cb_oof_cal
    df["p_rf_oof"] = rf_oof_cal

    catalog_cols = [
        "parent_asin", "title", "subcategory", "price", "average_rating",
        "rating_number", "success",
        "score_cb_oof", "score_rf_oof", "score_cb_dash", "score_rf_dash",
        "p_cb_oof", "p_rf_oof",
    ]
    out_csv = OUT_PRED / "catalog_extremes_rf_vs_catboost.csv"
    df[catalog_cols].to_csv(out_csv, index=False)

    report = {
        "protocol": {
            "oof": "5-fold estratificado + isotónica fold-a-fold (honesto)",
            "dashboard": "predict in-sample + isotónica fit nos OOF (tipo app)",
            "note": (
                "OOF = teto honesto do que o modelo generaliza; "
                "dashboard pode ir a 100 por sobreajuste in-sample."
            ),
        },
        "catboost": {
            "oof_calibrated": _score_stats(df["score_cb_oof"].values),
            "dashboard": _score_stats(df["score_cb_dash"].values),
            "top_oof": _top_rows(df, "score_cb_oof", args.top),
            "top_dashboard": _top_rows(df, "score_cb_dash", args.top),
        },
        "random_forest": {
            "oof_calibrated": _score_stats(df["score_rf_oof"].values),
            "dashboard": _score_stats(df["score_rf_dash"].values),
            "top_oof": _top_rows(df, "score_rf_oof", args.top),
            "top_dashboard": _top_rows(df, "score_rf_dash", args.top),
        },
        "per_subcategory_max_oof": {},
    }

    for sub, g in df.groupby("subcategory"):
        report["per_subcategory_max_oof"][sub] = {
            "catboost_max": float(g["score_cb_oof"].max()),
            "rf_max": float(g["score_rf_oof"].max()),
            "catboost_best": g.loc[g["score_cb_oof"].idxmax(), "title"][:80],
            "rf_best": g.loc[g["score_rf_oof"].idxmax(), "title"][:80],
        }

    # produto com maior score em cada modelo (OOF)
    cb_best = df.loc[df["score_cb_oof"].idxmax()]
    rf_best = df.loc[df["score_rf_oof"].idxmax()]
    report["absolute_best_oof"] = {
        "catboost": {
            "score": float(cb_best["score_cb_oof"]),
            "title": str(cb_best["title"])[:100],
            "parent_asin": cb_best["parent_asin"],
            "subcategory": cb_best["subcategory"],
            "success_label": int(cb_best["success"]),
        },
        "random_forest": {
            "score": float(rf_best["score_rf_oof"]),
            "title": str(rf_best["title"])[:100],
            "parent_asin": rf_best["parent_asin"],
            "subcategory": rf_best["subcategory"],
            "success_label": int(rf_best["success"]),
        },
    }

    out_json = OUT_METRICS / "catalog_extremes.json"
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    _print_block(
        "CatBoost · scores OOF calibrados (honesto)",
        report["catboost"]["oof_calibrated"],
        report["catboost"]["top_oof"],
        "score_cb_oof",
    )
    _print_block(
        "Random Forest · scores OOF calibrados (honesto)",
        report["random_forest"]["oof_calibrated"],
        report["random_forest"]["top_oof"],
        "score_rf_oof",
    )
    _print_block(
        "CatBoost · scores tipo dashboard (in-sample + cal OOF)",
        report["catboost"]["dashboard"],
        report["catboost"]["top_dashboard"],
        "score_cb_dash",
    )
    _print_block(
        "Random Forest · scores tipo dashboard (in-sample + cal OOF)",
        report["random_forest"]["dashboard"],
        report["random_forest"]["top_dashboard"],
        "score_rf_dash",
    )

    print(f"\n{'=' * 72}")
    print("Máximo absoluto (OOF calibrado)")
    print("=" * 72)
    a, b = report["absolute_best_oof"]["catboost"], report["absolute_best_oof"]["random_forest"]
    print(f"  CatBoost: score={a['score']:.1f}  [{a['subcategory']}]  {a['title']}")
    print(f"  RF:       score={b['score']:.1f}  [{b['subcategory']}]  {b['title']}")
    print("\nMáx OOF por subcategoria:")
    for sub, v in sorted(report["per_subcategory_max_oof"].items()):
        print(
            f"  {sub:<28}  CB={v['catboost_max']:5.1f}  RF={v['rf_max']:5.1f}"
        )
    print(f"\nCSV  → {out_csv}")
    print(f"JSON → {out_json}")


if __name__ == "__main__":
    main()
