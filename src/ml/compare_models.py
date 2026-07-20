"""
compare_models.py — RF (baseline congelado) vs CatBoost (último pipeline).

Lee:
  output/metrics/rf_baseline_oof.json      → Random Forest documentado
  output/metrics/analysis_results.json    → Corrida actual (CatBoost)

Imprime tabla lado a lado y guarda:
  output/metrics/model_comparison.json

Uso:
  cd REPO/src/ml && python3 compare_models.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
METRICS = REPO / "output" / "metrics"
RF_PATH = METRICS / "rf_baseline_oof.json"
CUR_PATH = METRICS / "analysis_results.json"
OUT_PATH = METRICS / "model_comparison.json"

# Métricas onde menor é melhor
LOWER_IS_BETTER = {
    "brier",
    "brier_isotonic",
    "log_loss",
    "ece_uncalibrated",
    "ece_isotonic",
}

CORE_KEYS = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "brier",
    "log_loss",
)


def _load(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"No encontrado: {path}")
    return json.loads(path.read_text())


def _catboost_from_analysis(raw: dict) -> dict:
    vvi = raw["part_V_VI"]
    vii = raw.get("part_VII", {})
    cal = vii.get("honest_calibration_oof", {})
    return {
        "model": "CatBoost",
        "source": str(CUR_PATH.relative_to(REPO)),
        "report_oof": vvi["report_oof"],
        "ece_uncalibrated": vvi.get("ece_uncalibrated"),
        "honest_calibration_oof": cal,
    }


def _flat(block: dict) -> dict:
    rep = block["report_oof"]
    cal = block.get("honest_calibration_oof") or {}
    iso = cal.get("isotonic_oof") or {}
    out = {k: rep.get(k) for k in CORE_KEYS}
    out["ece_uncalibrated"] = block.get("ece_uncalibrated") or (
        cal.get("uncalibrated") or {}
    ).get("ece")
    out["ece_isotonic"] = iso.get("ece")
    out["brier_isotonic"] = iso.get("brier")
    return out


def _delta(a, b):
    if a is None or b is None:
        return None
    return float(b) - float(a)


def _winner(key: str, rf_v, cb_v) -> str:
    if rf_v is None or cb_v is None:
        return "—"
    if abs(float(rf_v) - float(cb_v)) < 1e-6:
        return "empate"
    lower = key in LOWER_IS_BETTER
    if lower:
        return "RF" if float(rf_v) < float(cb_v) else "CatBoost"
    return "RF" if float(rf_v) > float(cb_v) else "CatBoost"


def _fmt(v, nd=3):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def compare(rf: dict, cb: dict) -> dict:
    rf_f, cb_f = _flat(rf), _flat(cb)
    rows = []
    for key in list(CORE_KEYS) + ["ece_uncalibrated", "ece_isotonic", "brier_isotonic"]:
        rv, cv = rf_f.get(key), cb_f.get(key)
        rows.append({
            "metric": key,
            "rf": rv,
            "catboost": cv,
            "delta_cb_minus_rf": _delta(rv, cv),
            "winner": _winner(key, rv, cv),
        })

    # Contagem de vitórias (métricas de ranking/prob; F1 separado)
    ranking = {"roc_auc", "pr_auc", "brier", "log_loss", "ece_isotonic"}
    rf_w = sum(1 for r in rows if r["metric"] in ranking and r["winner"] == "RF")
    cb_w = sum(1 for r in rows if r["metric"] in ranking and r["winner"] == "CatBoost")

    return {
        "protocol": "OOF stratified 5-fold, threshold=0.5, same dataset Master_Beauty_Dataset",
        "rf": {
            "model": rf.get("model", "RandomForest"),
            "source": rf.get("source"),
            "hparams": rf.get("hparams"),
            "confusion": rf["report_oof"].get("confusion"),
        },
        "catboost": {
            "model": cb.get("model", "CatBoost"),
            "source": cb.get("source"),
            "confusion": cb["report_oof"].get("confusion"),
        },
        "rows": rows,
        "summary": {
            "ranking_wins_rf": rf_w,
            "ranking_wins_catboost": cb_w,
            "note": (
                "CatBoost usa auto_class_weights=Balanced → mais recall / menos precision. "
                "Comparar F1 e ROC/PR-AUC juntos; não só accuracy."
            ),
        },
    }


def print_table(comp: dict) -> None:
    print("=" * 72)
    print("Comparação · Random Forest (baseline) vs CatBoost (atual)")
    print("=" * 72)
    print(f"Protocolo: {comp['protocol']}")
    print(f"RF source:       {comp['rf']['source']}")
    print(f"CatBoost source: {comp['catboost']['source']}")
    print()
    hdr = f"{'Métrica':<18} {'RF':>10} {'CatBoost':>10} {'Δ (CB−RF)':>12} {'Melhor':>10}"
    print(hdr)
    print("-" * len(hdr))
    for r in comp["rows"]:
        d = r["delta_cb_minus_rf"]
        d_s = "—" if d is None else f"{d:+.3f}"
        print(
            f"{r['metric']:<18} {_fmt(r['rf']):>10} {_fmt(r['catboost']):>10} "
            f"{d_s:>12} {r['winner']:>10}"
        )
    print()
    print("Matriz de confusão (OOF):")
    print(f"  RF:       {comp['rf']['confusion']}")
    print(f"  CatBoost: {comp['catboost']['confusion']}")
    print()
    s = comp["summary"]
    print(
        f"Vitórias em métricas de ranking/calibração: "
        f"RF={s['ranking_wins_rf']} · CatBoost={s['ranking_wins_catboost']}"
    )
    print(f"Nota: {s['note']}")
    print(f"\nJSON → {OUT_PATH}")


def main() -> None:
    rf = _load(RF_PATH)
    cb = _catboost_from_analysis(_load(CUR_PATH))
    comp = compare(rf, cb)
    OUT_PATH.write_text(json.dumps(comp, indent=2, ensure_ascii=False) + "\n")
    print_table(comp)


if __name__ == "__main__":
    main()
