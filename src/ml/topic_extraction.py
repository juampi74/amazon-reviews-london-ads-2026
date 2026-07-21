"""
topic_extraction.py — Parte IX del pipeline: temas, keywords y sentimiento de reviews
Product Success Predictor · Beauty & Personal Care (Amazon Reviews'23)

Implementa FR-015 ("The system extracts review topics and keywords", P1) y la
mitad de FR-014 (sentiment) descritas en docs/02_REQUIREMENTS.md, siguiendo el
diseño de docs/06_ML_MODELS.md:
    Sentiment: VADER baseline.
    Topics:    TF-IDF/NMF por subcategoría.
    Keywords:  frecuencia ponderada por polaridad.

Corre SOBRE Dataset C (`reviews_nlp_subset`, ver docs/05_DATA_PIPELINE.md §4),
NO sobre el metadata de producto que ya usa run_pipeline.py. Es decir, este
script necesita el archivo de reviews individuales (texto real de clientes),
que a la fecha de este commit todavía no está descargado — ver
docs/PROJECT_CONTEXT.md §6 ("Reviews file: not yet downloaded"). El script
está listo para correr en cuanto ese archivo exista; hasta entonces
`review_topics.json` simplemente no se genera y `inference.py` cae de vuelta
a los topics estáticos del frontend (ver AnalysisResults.tsx).

Input esperado (esquema estándar McAuley Lab Amazon Reviews'23, reviews file):
    parent_asin, asin, rating, title, text, verified_purchase, timestamp, ...
Se une contra Master_Beauty_Dataset (parent_asin -> subcategory) para poder
agregar por subcategoría real (categories[1]), igual que run_pipeline.py.

Salida:
    output/models/review_topics.json   (consumido por inference.py, Artifacts.topics())
    output/metrics/topic_extraction_report.json  (para auditoría/QA, mismo patrón
                                                    que analysis_results.json)

Uso:
    python topic_extraction.py \
        --reviews /path/to/Master_Reviews_Dataset.csv \
        --catalog /path/to/Master_Beauty_Dataset.csv \
        --min-reviews 50
"""
from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings("ignore")
RNG = 42
np.random.seed(RNG)

# ---- Rutas (mismo layout que run_pipeline.py) --------------------------------
HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
OUT_MODELS = REPO / "output" / "models"
OUT_METRICS = REPO / "output" / "metrics"
for d in (OUT_MODELS, OUT_METRICS):
    d.mkdir(parents=True, exist_ok=True)

REAL_SUBCATS = [
    "Hair Care", "Skin Care", "Foot, Hand & Nail Care", "Makeup",
    "Tools & Accessories", "Fragrance", "Shave & Hair Removal", "Personal Care",
]
DATASET_VERSION = "beauty-reviews-nlp-2026-07"
N_TOPICS = 4              # 4 temas por subcategoría, para calzar con el UI actual
N_KEYWORDS_PER_SIDE = 6   # top-N keywords positivas / negativas por subcategoría
MIN_REVIEWS_DEFAULT = 50  # Dataset C, ver docs/05_DATA_PIPELINE.md §4


def banner(t: str) -> None:
    print("\n" + "=" * 72 + f"\n{t}\n" + "=" * 72)


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    return o


# ============================================================
# Carga y limpieza (Dataset C)
# ============================================================

def load_reviews_nlp_subset(reviews_path: Path, catalog_path: Path,
                             min_reviews: int) -> pd.DataFrame:
    """Construye Dataset C: reviews unidas a subcategoría real, filtradas por volumen mínimo.

    No reemplaza Dataset A/B (catalog_training_all / catalog_priced_subset) — es
    un subset derivado exclusivamente para NLP, igual que documenta
    docs/05_DATA_PIPELINE.md §4 (`reviews_nlp_subset`).
    """
    banner("PARTE IX · Carga Dataset C (reviews_nlp_subset)")
    reviews = pd.read_csv(reviews_path, low_memory=False)
    catalog = pd.read_csv(catalog_path, low_memory=False, usecols=lambda c: c in {
        "parent_asin", "subcategory", "categories",
    })

    # subcategoría real ya resuelta en el catálogo (categories[1], ver run_pipeline.py);
    # si el catálogo no trae 'subcategory' calculada, se asume que sí (artefacto de Fase 2).
    if "subcategory" not in catalog.columns:
        raise ValueError(
            "catalog_path debe traer la columna 'subcategory' ya resuelta "
            "(categories[1]) — correr Fase 2 de run_pipeline.py primero."
        )

    df = reviews.merge(
        catalog[["parent_asin", "subcategory"]].drop_duplicates("parent_asin"),
        on="parent_asin", how="inner",
    )
    df = df[df["subcategory"].isin(REAL_SUBCATS)].copy()

    df["text"] = (df.get("title", "").fillna("") + " " + df.get("text", "").fillna("")).str.strip()
    df = df[df["text"].str.len() > 0]
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["rating"])
    df["verified_purchase"] = df.get("verified_purchase", False).astype(bool)

    counts = df.groupby("subcategory")["text"].count()
    keep = counts[counts >= min_reviews].index
    dropped = counts[~counts.index.isin(keep)]
    if len(dropped):
        print(f"Subcategorías con menos de {min_reviews} reviews (excluidas del "
              f"artefacto, quedan en fallback estático): {dropped.to_dict()}")
    df = df[df["subcategory"].isin(keep)].reset_index(drop=True)

    print(f"Reviews retenidas para NLP: {len(df):,} en {df['subcategory'].nunique()} subcategorías")
    return df


# ============================================================
# Sentimiento (VADER baseline, per docs/06_ML_MODELS.md)
# ============================================================

def score_sentiment(texts: pd.Series) -> np.ndarray:
    """Devuelve compound score VADER en [-1, 1] por review.

    VADER es la baseline documentada (léxico, sin GPU, corre offline como
    batch job — ver docs/PROJECT_CONTEXT.md §"Constraint to respect": nunca
    cargar un transformer dentro de la app servida, solo aquí).
    """
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except ImportError as exc:
        raise ImportError(
            "Falta vaderSentiment. Agregar a requirements.txt / src/ml y "
            "`pip install vaderSentiment` antes de correr este script."
        ) from exc
    analyzer = SentimentIntensityAnalyzer()
    return np.array([analyzer.polarity_scores(str(t))["compound"] for t in texts])


# ============================================================
# Topics (TF-IDF + NMF por subcategoría, per docs/06_ML_MODELS.md)
# ============================================================

STOPWORDS_EXTRA = {
    "product", "amazon", "use", "used", "using", "just", "really", "get",
    "got", "one", "would", "also", "like", "good", "great", "bought", "buy",
}


def _clean_for_topics(text: str) -> str:
    text = re.sub(r"[^a-zA-Z\s]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def extract_topics_for_subcategory(texts: pd.Series, n_topics: int = N_TOPICS) -> list[dict]:
    """TF-IDF -> NMF: cada componente es un 'tema' recurrente en las reviews.

    El 'share' de cada tema es la proporción de reviews donde ese tema es el
    dominante (argmax de la matriz W) — no una probabilidad de tópico LDA,
    sino una asignación dura simple, más fácil de explicar en el pitch/tesis.
    """
    cleaned = texts.map(_clean_for_topics)
    vec = TfidfVectorizer(max_features=800, stop_words="english", min_df=5, ngram_range=(1, 2))
    X = vec.fit_transform(cleaned)
    if X.shape[0] < n_topics * 5:
        return []

    vocab = np.array(vec.get_feature_names_out())
    nmf = NMF(n_components=n_topics, random_state=RNG, init="nndsvda", max_iter=400)
    W = nmf.fit_transform(X)
    H = nmf.components_

    dominant = W.argmax(axis=1)
    n = len(cleaned)
    topics = []
    for k in range(n_topics):
        top_terms = vocab[np.argsort(H[k])[::-1][:6]]
        top_terms = [t for t in top_terms if t not in STOPWORDS_EXTRA][:3]
        share = float((dominant == k).mean() * 100)
        topics.append({
            "name": ", ".join(top_terms) if top_terms else f"Topic {k+1}",
            "share": round(share, 1),
        })
    topics.sort(key=lambda t: t["share"], reverse=True)
    # normalizar shares para que sumen ~100 (quedan comparables sin importar n)
    total = sum(t["share"] for t in topics) or 1.0
    for t in topics:
        t["share"] = round(t["share"] / total * 100, 1)
    return topics


def extract_keywords_by_polarity(texts: pd.Series, sentiment: np.ndarray,
                                  n_per_side: int = N_KEYWORDS_PER_SIDE) -> tuple[list[str], list[str]]:
    """Keywords que más empujan el promedio de sentimiento hacia positivo o negativo.

    Para cada n-grama del vocabulario TF-IDF: correlación simple entre su
    presencia (binaria) y el compound score VADER de la review. Es una
    aproximación barata a 'frequency weighted by polarity' (docs/06_ML_MODELS.md);
    la alternativa más rigurosa (regresión con regularización L1 sobre
    presencia de n-gramas -> sentiment) queda como mejora futura si el
    equipo necesita más precisión para el paper.
    """
    cleaned = texts.map(_clean_for_topics)
    vec = TfidfVectorizer(max_features=600, stop_words="english", min_df=8,
                          ngram_range=(1, 2), binary=True)
    X = vec.fit_transform(cleaned).toarray()
    if X.shape[0] < 20:
        return [], []
    vocab = vec.get_feature_names_out()

    s = sentiment - sentiment.mean()
    s_std = s.std() or 1.0
    scores = (X * s[:, None]).sum(axis=0) / (X.sum(axis=0) + 1e-6) / s_std

    order = np.argsort(scores)
    neg_idx = [i for i in order if vocab[i] not in STOPWORDS_EXTRA][:n_per_side]
    pos_idx = [i for i in order[::-1] if vocab[i] not in STOPWORDS_EXTRA][:n_per_side]
    return list(vocab[pos_idx]), list(vocab[neg_idx])


# ============================================================
# Orquestación por subcategoría
# ============================================================

def build_subcategory_insight(group: pd.DataFrame) -> dict:
    sentiment = score_sentiment(group["text"])
    pos_rate = float((sentiment >= 0.05).mean() * 100)
    neg_rate = float((sentiment <= -0.05).mean() * 100)
    neu_rate = max(0.0, 100.0 - pos_rate - neg_rate)

    topics = extract_topics_for_subcategory(group["text"])
    pos_kw, neg_kw = extract_keywords_by_polarity(group["text"], sentiment)

    rating_dist = (group["rating"].round().clip(1, 5).astype(int)
                   .value_counts(normalize=True).reindex(range(1, 6), fill_value=0.0) * 100)

    return {
        "sentiment": {
            "positive": round(pos_rate, 1),
            "neutral": round(neu_rate, 1),
            "negative": round(neg_rate, 1),
        },
        "topics": topics,
        "positive_keywords": pos_kw,
        "negative_keywords": neg_kw,
        "rating_distribution": {str(s): round(float(p), 1) for s, p in rating_dist.items()},
        "verified_purchase_pct": round(float(group["verified_purchase"].mean() * 100), 1),
        "sample_size": int(len(group)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviews", type=Path, required=True,
                        help="CSV de Master_Reviews_Dataset (texto individual de reviews)")
    parser.add_argument("--catalog", type=Path, required=True,
                        help="CSV de Master_Beauty_Dataset con 'subcategory' ya resuelta")
    parser.add_argument("--min-reviews", type=int, default=MIN_REVIEWS_DEFAULT,
                        help="Volumen mínimo de reviews por subcategoría (Dataset C)")
    args = parser.parse_args()

    df = load_reviews_nlp_subset(args.reviews, args.catalog, args.min_reviews)
    if df.empty:
        print("Dataset C vacío tras los filtros — no se genera review_topics.json.")
        return

    banner("PARTE IX · Sentimiento + topics + keywords por subcategoría")
    result: dict[str, dict] = {}
    report: dict[str, dict] = {}
    for subcat, group in df.groupby("subcategory"):
        print(f"\n[{subcat}]  n={len(group):,}")
        insight = build_subcategory_insight(group)
        insight["dataset_version"] = DATASET_VERSION
        insight["source_type"] = "model"
        result[subcat] = insight
        report[subcat] = {"sample_size": insight["sample_size"],
                          "sentiment": insight["sentiment"],
                          "top_topic": insight["topics"][0] if insight["topics"] else None}
        print(f"  sentiment: {insight['sentiment']}")
        print(f"  top topic: {report[subcat]['top_topic']}")
        print(f"  keywords+: {insight['positive_keywords']}")
        print(f"  keywords-: {insight['negative_keywords']}")

    out_path = OUT_MODELS / "review_topics.json"
    out_path.write_text(json.dumps(jsonable(result), indent=2, ensure_ascii=False))
    (OUT_METRICS / "topic_extraction_report.json").write_text(
        json.dumps(jsonable(report), indent=2, ensure_ascii=False))

    banner("PARTE IX COMPLETA")
    print(f"Artefacto (consumido por inference.py) → {out_path}")
    print(f"Reporte de auditoría                    → {OUT_METRICS / 'topic_extraction_report.json'}")


if __name__ == "__main__":
    main()
