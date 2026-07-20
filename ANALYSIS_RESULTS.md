# Resultados del análisis de datos — Product Success Predictor
### Beauty & Personal Care (Amazon Reviews'23) · ejecutado sobre `Master_Beauty_Dataset.csv`

Este documento reporta **los números reales** producidos al ejecutar de principio a fin el
pipeline descrito en `PROJECT_CONTEXT.md`, `FORMULAS_AND_METRICS.md` (Partes I–VIII) y
`METHODS_TO_MOCKUP.md`. Todo es reproducible con:

```bash
cd REPO/src/ml
python run_pipeline.py            # pipeline completo (I–VIII) + artefactos + figuras
python fix_calibration_shap.py    # calibración honesta OOF + diagnóstico SHAP
python metrics.py                 # auto-test de todas las fórmulas (datos sintéticos)
```

**Salidas:** métricas → [`REPO/output/metrics/analysis_results.json`](REPO/output/metrics/analysis_results.json) ·
artefactos → `REPO/output/models/` · figuras → `REPO/output/figures/` ·
catálogo puntuado → `REPO/output/predictions/scored_catalog.csv`.

---

## 0. Dataset (medido, no asumido)

| Propiedad | Valor |
|---|---|
| Filas crudas | 47,755 |
| Filas usadas (8 subcategorías reales, `categories[1]`) | **46,435** (97.2%) |
| `price` nulo | **0.00%** (el CSV ya viene depurado) |
| `title` nulo | 0.002% (1 fila) |

**Distribución por subcategoría:**

| Subcategoría | n |
|---|---|
| Hair Care | 14,062 |
| Skin Care | 9,225 |
| Foot, Hand & Nail Care | 5,800 |
| Makeup | 5,039 |
| Tools & Accessories | 4,497 |
| Shave & Hair Removal | 3,175 |
| Fragrance | 2,369 |
| Personal Care | 2,268 |

---

## Parte I — Estadística descriptiva, forma y outliers

| Variable | n | media | mediana | std | Q1 | Q3 | IQR | skew (g1) | kurtosis exc. (g2) | máx |
|---|---|---|---|---|---|---|---|---|---|---|
| `average_rating` | 46,435 | 4.362 | 4.400 | 0.310 | 4.20 | 4.60 | 0.40 | **−1.436** | 3.558 | 5.00 |
| `rating_number` | 46,435 | 2,366.3 | 876.0 | 6,018.9 | 422 | 2,041.5 | 1,619.5 | **+14.856** | 516.97 | 340,182 |
| `price` (USD) | 46,435 | 24.02 | 15.99 | 31.21 | 9.96 | 26.21 | 16.25 | **+14.091** | 761.86 | 2,398.91 |

**Lectura (confirma los MD):** el rating tiene **sesgo negativo** (casi todo puntúa alto → poca
capacidad discriminante), mientras que volumen y precio están **fuertemente sesgados a la
derecha** → se usa mediana, no media, y se justifica `log1p`.

**Efecto de `log1p` sobre el sesgo:**

| Variable | skew crudo | skew tras log1p |
|---|---|---|
| `rating_number` | 14.86 | **0.11** |
| `price` | 14.09 | **0.75** |

**Detección de outliers (nº de productos marcados):**

| Variable | Tukey k=1.5 | Tukey k=3 | Z-score >3 | MAD >3.5 |
|---|---|---|---|---|
| `rating_number` | 5,328 | 3,185 | 679 | 6,194 |
| `price` | 3,798 | 2,077 | 817 | 3,661 |

El z-score clásico detecta muchos menos (la cola ya distorsiona media y std) → con datos
sesgados manda Tukey/MAD, como indican los MD. Figura: `output/figures/01_distributions.png`.

---

## Parte II — Relaciones e inferencia (EDA)

| Par | Pearson r | Spearman ρ | Lectura |
|---|---|---|---|
| `price` ↔ `average_rating` | 0.013 (p=5e-3) | **0.070** | precio casi no predice el rating |
| `price` ↔ log(volumen) | −0.096 | **−0.134** | lo barato vende un poco más |
| `average_rating` ↔ log(volumen) | 0.191 | **0.185** | lo popular puntúa levemente mejor |
| `review_count` ↔ `rating_number` | 0.699 | **0.651** | miden lo mismo (popularidad) |

**Normalidad (D'Agostino-Pearson, muestra 5k):** `average_rating`, `rating_number` y `price`
son **todas NO normales** (p<0.05) → se usan pruebas **no paramétricas** (Spearman,
Mann-Whitney). Figura: `output/figures/02_correlations.png`.

> **La verdad incómoda de los MD queda confirmada con datos:** la señal producto→éxito es
> **débil**. Es un resultado válido y publicable (la hipótesis lo contempla explícitamente).

---

## Parte III — Diseño del target "éxito"

**Regla:** `success = 1` si `average_rating ≥ mediana_subcat` **Y** `log1p(rating_number) ≥ p60_subcat`.

**Umbrales poblacionales por subcategoría** (artefacto `subcategory_stats.json`):

| Subcategoría | mediana rating | p60 log-volumen | mediana precio | p25 | p75 | IQR precio |
|---|---|---|---|---|---|---|
| Foot, Hand & Nail Care | 4.4 | 7.136 | ~13 | 7.99 | 18.00 | 10.01 |
| Fragrance | 4.5 | 7.042 | ~30 | 19.50 | 60.00 | 40.50 |
| Hair Care | 4.4 | 7.082 | ~17 | 9.99 | 29.95 | 19.96 |
| Makeup | 4.3 | 7.058 | ~13 | 8.29 | 21.00 | 12.72 |
| Personal Care | 4.4 | 6.990 | ~14 | 9.97 | 21.00 | 11.03 |
| Shave & Hair Removal | 4.4 | 7.146 | ~17 | 11.42 | 29.95 | 18.54 |
| Skin Care | 4.5 | 7.009 | ~16 | 11.97 | 28.00 | 16.03 |
| Tools & Accessories | 4.5 | 7.156 | ~13 | 8.29 | 19.99 | 11.70 |

**Balance de clases:** clase positiva = **26.08%** (12,112 éxitos / 34,323 no-éxitos),
razón de desbalance 2.83. Desbalance moderado → se reporta F1/AUC (no accuracy) y
`class_weight="balanced"`.

**Análisis de sensibilidad del umbral** (¿las conclusiones dependen frágilmente de la definición?):

| Definición (rating_q / volumen_q) | tasa positiva | % etiquetas que cambian vs base |
|---|---|---|
| **p50 / p60 (base)** | 26.1% | — |
| p50 / p50 | 32.1% | 6.1% |
| p50 / p75 | 16.9% | 9.2% |
| p60 / p60 | 21.2% | 4.9% |

Solo **5–9%** de las etiquetas cambian de bando → la definición de éxito es **robusta**, no frágil.

**¿Difieren "exitosos" vs "no exitosos"?** (Welch + Mann-Whitney + tamaño de efecto):

| Variable | p-valor | Cohen's d | Lectura |
|---|---|---|---|
| `average_rating` | ~0 | **0.98** (grande) | separa mucho |
| `rating_number` | ~0 | **0.75** (medio-grande) | separa bien |
| `price` | 1.3e-55 | **−0.14** (trivial) | apenas separa (aunque p sea "significativo") |

> El precio es estadísticamente "significativo" por el enorme n, pero su **tamaño de efecto es
> trivial**: no es una palanca real de éxito. Justo la lección de honestidad de los MD.

**Asociación subcategoría ↔ éxito:** χ² = 72.1 (p=5.5e-13), **Cramér's V = 0.039** (asociación
muy débil): la subcategoría casi no determina el éxito por sí sola.

---

## Parte IV — Ingeniería de features

Matriz final **X: 46,435 × 316 features**:

- `price_fit` (desviación robusta del precio vs su subcategoría) + `price_is_missing`
  (=0 en todo el CSV, porque price ya está 0% nulo)
- 6 flags de presencia de `details`: `has_brand`, `has_item_form`, `has_color`, `has_scent`,
  `has_skin_type`, `has_hair_type`
- 8 columnas one-hot de subcategoría
- **300 términos TF-IDF** de `title + features` (1-2 gramas, sublinear_tf, stopwords EN)

Artefactos: `tfidf_vectorizer.pkl`, `feature_names.json`.

---

## Parte V–VI — Modelo, validación y evaluación

**Modelo actual (producto):** CatBoost (500 iteraciones, `depth=6`, `learning_rate=0.05`,
`auto_class_weights="Balanced"`), **validación estratificada 5-fold**, probabilidades
**out-of-fold** (sin fuga). Artefacto: `success-catboost-0.2.0`.

**Baseline documentado (fase anterior):** Random Forest (300 árboles, `min_samples_leaf=5`,
`class_weight="balanced"`), mismo protocolo OOF. Congelado en
`output/metrics/rf_baseline_oof.json`.

### Comparación RF vs CatBoost (OOF, threshold 0.5)

Reproducible con:

```bash
cd REPO/src/ml && python3 compare_models.py
```

Salida máquina: `output/metrics/model_comparison.json`.

| Métrica | RF (baseline) | CatBoost (actual) | Δ (CB − RF) | Mejor |
|---|---:|---:|---:|---|
| Accuracy | 0.748 | 0.642 | −0.106 | RF |
| Precisión | 0.529 | 0.385 | −0.144 | RF |
| Recall | 0.298 | 0.622 | +0.324 | CatBoost |
| **F1** | 0.381 | **0.476** | +0.095 | **CatBoost** |
| **ROC-AUC** | **0.715** | 0.692 | −0.023 | **RF** |
| **PR-AUC** | **0.472** | 0.432 | −0.040 | **RF** |
| Brier ↓ | **0.186** | 0.218 | +0.032 | **RF** |
| Log-loss ↓ | **0.555** | 0.622 | +0.067 | **RF** |
| ECE sin calibrar ↓ | **0.119** | 0.202 | +0.083 | RF |
| ECE isotónica OOF ↓ | 0.003 | **0.002** | −0.001 | CatBoost (≈empate) |

**Matrices de confusión (OOF):**

| | TN | FP | FN | TP |
|---|---:|---:|---:|---:|
| RF | 31,105 | 3,218 | 8,503 | 3,609 |
| CatBoost | 22,295 | 12,028 | 4,578 | 7,534 |

> **Lectura:** CatBoost (con pesos balanceados) es **más agresivo**: sube mucho el recall
> (+0.32) y el F1 (+0.10), a costa de precisión y de un AUC/PR-AUC algo peores. RF era
> **conservador** (preciso cuando dice éxito, pero deja muchos éxitos sin detectar). Tras
> calibración isotónica OOF, ambos quedan honestos (ECE ≈ 0.002–0.003). Para ranking puro
> manda RF; para capturar más productos exitosos en el dashboard, CatBoost equilibra mejor.
> Figura actual: `output/figures/03_evaluation.png`.

### Auditoría de fuga de datos (data leakage)

Tarea explícita de la Persona 2. Se re-evaluó con validación cruzada **sin fuga**: TF-IDF y las
estadísticas de `price_fit` por subcategoría se ajustan **solo con el fold de entrenamiento** y
se aplican al de test (antes se ajustaban sobre todo el dataset). El target se mantiene como
definición poblacional fija (regla de negocio, no es feature). Script: `leakage_audit.py`.

| Métrica | Con fuga (fit global) | **Sin fuga (fit por fold)** | Δ |
|---|---|---|---|
| ROC-AUC | 0.7151 | **0.7149** | −0.0002 |
| PR-AUC | 0.4717 | **0.4713** | −0.0004 |
| F1 | 0.3811 | 0.3815 | +0.0004 |
| Brier | 0.1862 | 0.1862 | 0 |
| ECE (isotónica OOF) | 0.0029 | **0.0017** | mejor |

> **Conclusión de la auditoría:** la fuga era **despreciable** (ROC-AUC cae 0.0002). Las métricas
> reportadas **no estaban infladas**; el 0.715 es señal genuina. Datos: `output/metrics/leakage_audit.json`.

---

## Parte VII — Calibración e interpretabilidad

**Calibración (medida HONESTAMENTE sobre las predicciones out-of-fold):**

| Método | ECE | Brier |
|---|---|---|
| Sin calibrar | 0.1187 | 0.1862 |
| **Isotónica (OOF)** | **0.0029** | **0.1704** |
| Platt / sigmoide (OOF) | 0.0043 | 0.1703 |

La calibración **isotónica** reduce el ECE de 0.119 a **0.003** → un "score 80" del dashboard
significa de verdad ~80% de éxito. Artefacto: `calibrator_1d.pkl` (+ `calibrator.pkl` como
wrapper desplegable). Figura: `output/figures/06_calibration_honest.png`.

> **Nota metodológica:** una primera corrida comparó por error el ECE *out-of-fold* contra el
> ECE *in-sample*, dando la falsa impresión de que la calibración "empeoraba". Medido como
> exigen los MD (sobre OOF), la calibración funciona muy bien. Corregido.

**Importancia por permutación (método preferido por los MD — caída de ROC-AUC al barajar):**

| Feature | Importancia |
|---|---|
| **`price_fit`** | 0.0223 |
| `has_scent` | 0.0087 |
| `tfidf::natural` | 0.0058 |
| `tfidf::oz` | 0.0037 |
| `tfidf::pack` | 0.0036 |
| `tfidf::skin` | 0.0029 |
| `tfidf::hair` | 0.0024 |
| `tfidf::color` | 0.0019 |

El **ajuste de precio** es el predictor estructurado más fuerte, seguido de la presencia de
`Scent` y de términos de texto ("natural", tamaño "oz", "pack"). También se exportó la
importancia Gini/MDI (top: `price_fit`, luego términos TF-IDF). Figura: `output/figures/04_importance.png`.

**SHAP:** omitido localmente — `shap 0.50` + `sklearn 1.7` fallan el *additivity check* con
RandomForest (los valores divergen a ~1e14; ranking no confiable). Los MD ya indican **correr
SHAP en Google Colab, no en la app**; el código queda en `metrics.py::shap_values`. La
interpretabilidad local queda cubierta con la importancia por permutación.

---

## Parte VIII — Métricas del dashboard (Launchly)

Todas construidas sobre el modelo calibrado + índice k-NN coseno (`knn_index.pkl`,
`density_reference.npy`).

**Distribución del `success_score` (0–100) en el catálogo:** media 27.7 · mediana 19.9 ·
p10 8.5 · p90 60.8.

**Incertidumbre RF media** (std entre árboles): 0.297.

**Producto de ejemplo — REVLON One-Step Volumizer (Hair Care):**

| Métrica del mockup | Valor | Método |
|---|---|---|
| Success chance (`msSuccess`) | **82** | prob. calibrada × 100 |
| Market crowding / saturación (`satBar`) | 95.0 | densidad k-NN (percentil) |
| Decision risk (`msRisk`) | 43 | índice `0.5(1−p)+0.3·sat+0.2·u` |
| Similar products (`discoverGrid`) | 5 vecinos, sim 0.99–1.00 | coseno k-NN |
| Éxito de comparables | 4/5, **IC Wilson 95% = [0.38, 0.96]** | proporción + Wilson |
| Sweet-spot price (barrido) | `price_fit*` = +1.25 | barrido de precio sobre el modelo |

Cada elemento del mockup que los MD clasifican como **✅ Nivel A** (núcleo predictivo y
comparativo) queda producido con datos reales. Los niveles **🟡 B** (beneficio en $ con
supuestos) y **🔴 C** (forecast, sentiment, demografía) siguen requiriendo supuestos declarados
o fuentes externas, como documenta `METHODS_TO_MOCKUP.md`. Figura del barrido de precio:
`output/figures/05_price_curve.png`.

---

## Artefactos generados (para la app Streamlit)

| Archivo | Contenido |
|---|---|
| `output/models/model.pkl` | CatBoost final (`success-catboost-0.2.0`) |
| `output/metrics/rf_baseline_oof.json` | Baseline RF congelado (comparación) |
| `output/metrics/model_comparison.json` | Tabla RF vs CatBoost (`compare_models.py`) |
| `output/models/calibrator.pkl` | Calibrador desplegable (features → prob) |
| `output/models/calibrator_1d.pkl` | Calibrador isotónico honesto (OOF) |
| `output/models/knn_index.pkl` | Índice k-NN coseno (comparables) |
| `output/models/density_reference.npy` | Referencia de densidad (saturación) |
| `output/models/subcategory_stats.json` | Umbrales/percentiles por subcategoría |
| `output/models/tfidf_vectorizer.pkl` | Vectorizador TF-IDF |
| `output/models/feature_names.json` | Nombres de las 316 features |
| `output/predictions/scored_catalog.csv` | Catálogo con `p_cal` y `score` por producto |
| `output/predictions/proba_oof.npy` | Probabilidades out-of-fold |

---

## Conclusión (honestidad estadística)

1. **La hipótesis se resuelve del lado "señal débil pero real":** el CatBoost actual alcanza
   **ROC-AUC 0.692 / PR-AUC 0.432** (RF baseline: 0.715 / 0.472) — mejor que el azar y útil,
   pero lejos de determinista. El éxito en este mercado depende en buena parte de factores
   fuera de los datos (marketing, marca), tal como anticipa `PROJECT_CONTEXT.md §3`.
2. **RF vs CatBoost:** CatBoost gana en F1/recall (perfil más equilibrado para producto);
   RF gana en ROC/PR-AUC y Brier. Ver tabla en Parte V–VI y `compare_models.py`.
3. **El precio casi no predice el éxito** (d=−0.14, ρ=0.07): el `price_fit` es el feature
   estructurado más importante, pero su efecto absoluto es pequeño → precio como **rango**, no
   como número exacto.
4. **La definición de éxito es robusta** (5–9% de flip ante umbrales alternativos).
5. **El modelo queda bien calibrado** (ECE isotónica ≈ 0.002) → los scores del dashboard son
   honestos. La **auditoría de fuga** (sobre RF) confirma que el AUC no estaba inflado
   (Δ ROC-AUC −0.0002).
6. **Limitaciones declaradas:** éxito es un proxy (rating+volumen), no ventas reales; sin serie
   temporal (no hay forecast real); sin texto de reseñas (no hay sentiment); SHAP debe correrse
   en Colab. El framework fallaría en una subcategoría genuinamente nueva sin comparables.
