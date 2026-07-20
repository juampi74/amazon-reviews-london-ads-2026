# RF vs CatBoost — comparación práctica

**Producto:** Launchly · Product Success Predictor  
**Dataset:** Beauty & Personal Care, 46.435 productos · 8 subcategorías  
**Fecha:** 2026-07-20  

---

## 1. Qué predice el modelo (importante)

El score **no** es “probabilidad de vender”. Es la probabilidad calibrada de caer en nuestro **proxy de éxito**:

> En su subcategoría, el producto tiene buen rating (≥ mediana) **y** buen volumen de reviews (≥ percentil 60).

Es un atajo de “productos que ya se ven fuertes en Amazon”, no ventas reales ni garantía de lanzamiento.

| Score | Lectura práctica |
|---|---|
| **0–40** | Perfil poco parecido a éxitos históricos |
| **40–70** | Zona intermedia; mirar comparables y riesgo |
| **70–90** | Se parece bastante a productos exitosos del catálogo |
| **~100** | Máxima confianza en el **proxy** (no certeza de venta) |

Un score 80 bien calibrado significa: *entre productos con score ~80, ~80% eran éxito en el histórico*. No significa “va a vender el 80% de las unidades”.

---

## 2. Cómo comparamos

Misma data, mismas features, validación **out-of-fold** (5 folds estratificados). Así nadie se evalúa a sí mismo.

Scripts:

```bash
cd src/ml
python3 compare_models.py      # métricas RF vs CatBoost
python3 catalog_extremes.py    # tops y techos de score
```

---

## 3. Números clave

| Métrica | Random Forest | CatBoost | Quién gana |
|---|---:|---:|---|
| ROC-AUC (ranking) | **0.715** | 0.692 | RF |
| PR-AUC | **0.472** | 0.432 | RF |
| **F1** | 0.381 | **0.476** | **CatBoost** |
| Precision | **0.529** | 0.385 | RF |
| Recall | 0.298 | **0.622** | **CatBoost** |
| Brier ↓ | **0.186** | 0.218 | RF |
| ECE tras calibrar ↓ | ~0.003 | **~0.002** | Empate (ambos honestos) |

**En criollo:**

- **RF** es más **conservador**: cuando dice “éxito”, acierta más a menudo, pero **se pierde** muchos éxitos reales (recall bajo ~30%).
- **CatBoost** es más **agresivo**: detecta muchos más éxitos (recall ~62%), con más falsos positivos. El F1 (equilibrio) es mejor.
- Ambos quedan **bien calibrados** después de la calibración isotónica → el score del dashboard se puede leer como probabilidad del proxy.

### Matrices (threshold 0.5)

| | TN | FP | FN | TP |
|---|---:|---:|---:|---:|
| RF | 31.105 | 3.218 | 8.503 | 3.609 |
| CatBoost | 22.295 | 12.028 | 4.578 | 7.534 |

CatBoost encuentra **el doble** de verdaderos éxitos (TP), a costa de más falsas alarmas (FP).

---

## 4. Techos de score y productos top

Ambos pueden llegar a **score 100** en el proxy. Eso es normal: hay productos cuyo perfil (texto + precio relativo + señales) se parece muchísimo a los éxitos históricos.

**Ejemplos frecuentes en el top:** líneas NIVEA, OGX, Johnson’s — marcas/textos muy asociados a productos ya exitosos en Skin/Hair Care.

| | CatBoost (OOF) | RF (OOF) |
|---|---:|---:|
| Score máximo | 100 | 100 |
| Productos con score ≥ 90 | 35 | 17 |
| Mediana del catálogo | ~24 | ~25 |

La mediana ~26 refleja la tasa base de éxito (~26%): la mayoría del catálogo no es “éxito”.

---

## 5. ¿Cuál modelo conviene para nuestra práctica?

### Recomendación: **CatBoost** (modelo de producto actual)

Para Launchly / el dashboard de “chance de éxito” pre-lanzamiento, priorizamos:

1. **No perder oportunidades** — un seller quiere ver si su idea *puede* parecerse a un éxito; recall bajo es caro.
2. **Score usable y honesto** — ambos calibran bien; CatBoost ya está en producción (`success-catboost-0.2.0`).
3. **Equilibrio (F1)** — mejor que RF para un umbral de decisión único.

| Si el objetivo es… | Mejor modelo |
|---|---|
| Dashboard / product score para sellers | **CatBoost** |
| Ranking puro o menos falsas alarmas | RF (baseline) |
| Explicar “hay señal real en los datos” | Ambos (AUC ~0.70) |

**RF no se tira:** queda como baseline documentado (`output/metrics/rf_baseline_oof.json`) para comparar regresiones futuras.

---

## 6. Cómo usarlo con el equipo (reglas simples)

1. Mostrar el score como **rango + incertidumbre**, no como certeza.
2. Decir siempre: *“probabilidad de parecerse a productos exitosos del histórico”*, no *“probabilidad de venta”*.
3. Mirar también **comparables, saturación y riesgo** — el score solo no decide el lanzamiento.
4. Un 100 no garantiza venta; un 30 no prohíbe lanzar — solo indica poco parecido a éxitos pasados.

---

## 7. Archivos de soporte

| Archivo | Contenido |
|---|---|
| `output/metrics/model_comparison.json` | Tabla métricas RF vs CatBoost |
| `output/metrics/catalog_extremes.json` | Tops y techos de score |
| `output/predictions/catalog_extremes_rf_vs_catboost.csv` | Score por producto (ambos) |
| `ANALYSIS_RESULTS.md` | Resultados detallados del pipeline |

---

**TL;DR:** Hay señal real, pero modesta. **CatBoost es el mejor para la práctica del producto** (más equilibrado, detecta más éxitos, score calibrado). RF ordena un poco mejor, pero es demasiado conservador para el uso en el dashboard.
