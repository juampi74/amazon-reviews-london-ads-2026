# Fórmulas y Métricas — Product Success Predictor
### Documento técnico completo · Beauty & Personal Care (Amazon Reviews'23)

Este documento reúne **todas** las fórmulas, métricas y reglas estadísticas que usa el
proyecto, explicadas una por una, y al final de cada sección el **código Python** que las
calcula. Al final hay un **módulo consolidado y ejecutable** (`metrics.py`) que corre de
principio a fin sobre datos sintéticos para que puedas verificar cada función.

**Notación:** $x_i$ es el valor del producto $i$; $n$ el número de productos; $c$ la
subcategoría (Hair Care, Skin Care, …); $\bar{x}$ la media; $\tilde{x}$ la mediana.

**Reglas de ciencia de datos que atraviesan todo el documento:**

1. **Con distribuciones sesgadas se usa la mediana, no la media.** El precio y el número de
   reseñas están fuertemente sesgados a la derecha (§7 del contexto), así que la media
   engaña.
2. **Antes de correlacionar o testear, se mira la forma de la distribución.** Si no es
   normal → pruebas no paramétricas (Spearman, Mann-Whitney).
3. **Bajo desbalance de clases, la exactitud (accuracy) miente.** Se reporta F1, precisión,
   recall y AUC.
4. **Una probabilidad solo significa lo que dice si está calibrada.** Un "score 80" debe
   corresponder a ~80% de éxito real.
5. **Todo umbral es una decisión de diseño** y se somete a análisis de sensibilidad.

---

## Dependencias

```python
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    accuracy_score, roc_auc_score, average_precision_score,
    brier_score_loss, log_loss,
)
```

---

# PARTE I — Estadística descriptiva y limpieza

## 1.1 Medidas de tendencia central y dispersión

| Medida | Fórmula | Para qué sirve |
|---|---|---|
| Media | $\bar{x} = \frac{1}{n}\sum_i x_i$ | Centro; **sensible a outliers** |
| Mediana | valor central ordenado | Centro **robusto**; se usa con sesgo |
| Varianza | $s^2 = \frac{1}{n-1}\sum_i (x_i-\bar{x})^2$ | Dispersión al cuadrado |
| Desv. estándar | $s = \sqrt{s^2}$ | Dispersión en unidades originales |
| Rango intercuartílico | $\text{IQR} = Q_3 - Q_1$ | Dispersión robusta (50% central) |
| Coef. de variación | $\text{CV} = s/\bar{x}$ | Dispersión **relativa**, comparable entre variables |

> **Regla:** usamos $n-1$ (corrección de Bessel) porque estimamos la varianza a partir de
> una **muestra**, no de la población completa; divide entre $n-1$ para que el estimador sea
> insesgado.

## 1.2 Forma de la distribución: asimetría y curtosis

**Asimetría (skewness, Fisher-Pearson):**
$$
g_1 = \frac{\frac{1}{n}\sum_i (x_i-\bar{x})^3}{s^3}
$$
- $g_1 > 0$: cola larga a la derecha (caso de `price` y `rating_number`).
- $g_1 \approx 0$: simétrica.

**Curtosis en exceso:**
$$
g_2 = \frac{\frac{1}{n}\sum_i (x_i-\bar{x})^4}{s^4} - 3
$$
- $g_2 > 0$: colas pesadas / muchos valores extremos.

> **Por qué importa:** si $g_1$ es grande, la media deja de representar al "producto típico"
> y **cualquier modelo lineal o test que asuma normalidad falla**. Es lo que justifica la
> transformación logarítmica de la sección 1.4.

## 1.3 Cuantiles y percentiles

El percentil $p$ es el valor $Q_p$ tal que una fracción $p$ de los datos queda por debajo.
Se usan $Q_1$ (p25), $Q_2$ (p50 = mediana), $Q_3$ (p75), y **p60 del volumen** para definir
éxito (Parte III). Estos son la base de las comparaciones "dentro de la subcategoría".

## 1.4 Transformación logarítmica

$$
x' = \ln(1 + x) \quad (\texttt{log1p})
$$

> **Regla:** `rating_number` va de 0 a 99,553 con mediana 19 y media 233 → sesgo extremo.
> `log1p` comprime la cola y hace la variable aproximadamente simétrica, condición necesaria
> para que el modelo no quede dominado por unos pocos best-sellers gigantes. Se usa `log1p`
> (y no `log`) para admitir el valor 0 sin producir $-\infty$.

## 1.5 Detección de outliers (tres reglas)

**(a) Regla de Tukey / cercas del IQR** (robusta, la preferida con datos sesgados):
$$
\text{outlier si } x < Q_1 - k\cdot\text{IQR} \;\text{ o }\; x > Q_3 + k\cdot\text{IQR},
\quad k=1.5 \text{ (leve)},\; k=3 \text{ (extremo)}
$$

**(b) Z-score** (solo válido si la variable es aprox. normal):
$$
z_i = \frac{x_i - \bar{x}}{s}, \qquad \text{outlier si } |z_i| > 3
$$

**(c) Z-score modificado (basado en MAD)** — robusto, recomendado cuando hay outliers que
contaminan $\bar{x}$ y $s$:
$$
\text{MAD} = \text{mediana}(|x_i - \tilde{x}|), \qquad
M_i = \frac{0.6745\,(x_i - \tilde{x})}{\text{MAD}}, \qquad \text{outlier si } |M_i| > 3.5
$$
El factor 0.6745 hace que MAD sea comparable a la desviación estándar bajo normalidad.

> **Regla:** con `price` (63% nulo, muy sesgado) se prefiere Tukey o MAD sobre el z-score
> clásico, porque la media y la desviación estándar ya están distorsionadas por la cola.

## 1.6 Datos faltantes

- `price`: **63% nulo**. Regla del proyecto: **no imputar a ciegas**. Se preserva el nulo y
  se añade una bandera `price_is_missing` (0/1), porque la ausencia puede ser informativa.
  Las partes que dependen de precio usan solo el subconjunto ~37% con precio real.
- Tasa de faltantes por columna: $\text{missing}_j = \frac{\#\text{nulos en } j}{n}$.

### Código — Parte I

```python
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
    """Cercas de Tukey (robusto). k=1.5 leve, k=3 extremo. Devuelve máscara booleana."""
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
```

---

# PARTE II — Relaciones e inferencia (EDA)

## 2.1 Correlación de Pearson (lineal)

$$
r = \frac{\sum_i (x_i-\bar{x})(y_i-\bar{y})}{\sqrt{\sum_i (x_i-\bar{x})^2}\sqrt{\sum_i (y_i-\bar{y})^2}}
= \frac{\text{cov}(x,y)}{s_x\,s_y}, \qquad r\in[-1,1]
$$
Mide relación **lineal**. Asume normalidad y sensible a outliers.

## 2.2 Correlación de Spearman (monótona, robusta)

Es Pearson calculado sobre los **rangos** de los datos:
$$
\rho = 1 - \frac{6\sum_i d_i^2}{n(n^2-1)}, \qquad d_i = \text{rango}(x_i)-\text{rango}(y_i)
$$

> **Regla:** con variables sesgadas (precio, volumen) **Spearman es la opción por defecto**;
> Pearson solo si ambas variables son aprox. normales. Reportar ambas y comparar.

## 2.3 Correlación punto-biserial (continua vs. binaria)

Para medir cuánto se asocia una variable continua (p. ej. `price`) con el target binario
`success`. Es matemáticamente Pearson entre una continua y una 0/1:
$$
r_{pb} = \frac{\bar{x}_1 - \bar{x}_0}{s_x}\sqrt{\frac{n_1 n_0}{n^2}}
$$

## 2.4 Prueba de normalidad

Antes de elegir test paramétrico vs. no paramétrico:
- **Shapiro-Wilk** (n < ~5000) o **D'Agostino-Pearson** (n grande). $H_0$: los datos son
  normales. Si $p < 0.05$ → se rechaza normalidad → usar pruebas no paramétricas.

## 2.5 Pruebas de hipótesis: ¿"exitosos" vs. "no exitosos" difieren?

Comparamos un atributo (p. ej. precio) entre el grupo exitoso y el no exitoso.

**(a) t de Welch** (paramétrica, no asume varianzas iguales):
$$
t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{s_1^2/n_1 + s_2^2/n_2}}
$$

**(b) Mann-Whitney U** (no paramétrica — la preferida aquí por el sesgo). Contrasta si un
grupo tiende a tener valores mayores que el otro sin asumir normalidad.

**Tamaño del efecto** (siempre reportarlo junto al p-valor — un p pequeño con n grande puede
ser trivial):
- **Cohen's d** (para t):
$$
d = \frac{\bar{x}_1 - \bar{x}_2}{s_p}, \quad
s_p = \sqrt{\frac{(n_1-1)s_1^2 + (n_2-1)s_2^2}{n_1+n_2-2}}
$$
Interpretación: 0.2 pequeño, 0.5 medio, 0.8 grande.
- **Rank-biserial** (para Mann-Whitney): $r = 1 - \dfrac{2U}{n_1 n_2}$.

## 2.6 Asociación entre categóricas: Chi-cuadrado y Cramér's V

$$
\chi^2 = \sum \frac{(O - E)^2}{E}, \qquad
V = \sqrt{\frac{\chi^2}{n\,(\min(r,k)-1)}}
$$
$O$ observado, $E$ esperado bajo independencia. Cramér's $V\in[0,1]$ normaliza el $\chi^2$
para poder comparar la fuerza de asociación (p. ej. subcategoría ↔ éxito).

### Código — Parte II

```python
def correlations(x: pd.Series, y: pd.Series) -> dict:
    """Pearson (lineal) y Spearman (monótona, robusta) con sus p-valores."""
    mask = x.notna() & y.notna()
    xr, yr = x[mask], y[mask]
    r_p, p_p = stats.pearsonr(xr, yr)
    r_s, p_s = stats.spearmanr(xr, yr)
    return {"pearson_r": r_p, "pearson_p": p_p,
            "spearman_rho": r_s, "spearman_p": p_s}


def point_biserial(binary: pd.Series, continuous: pd.Series) -> dict:
    """Asociación entre el target binario y una variable continua."""
    mask = binary.notna() & continuous.notna()
    r, p = stats.pointbiserialr(binary[mask], continuous[mask])
    return {"r_pb": r, "p": p}


def normality_test(x: pd.Series) -> dict:
    """D'Agostino-Pearson. H0: normal. p<0.05 => NO normal => usar no paramétrico."""
    x = x.dropna()
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
```

---

# PARTE III — Diseño del target: "éxito"

## 3.1 Definición del éxito (proxy por percentiles dentro de subcategoría)

**No** hay datos reales de ventas; el éxito se define como un producto que logra **calidad**
(rating alto) **y tracción** (volumen alto) relativas a su propia subcategoría:

$$
\text{success}_i =
\begin{cases}
1 & \text{si } \text{avg\_rating}_i \ge \text{mediana\_rating}_c \;\wedge\; \ln(1+\text{rating\_number}_i) \ge p60_c \\
0 & \text{en otro caso}
\end{cases}
$$

donde $\text{mediana\_rating}_c$ y $p60_c$ (percentil 60 del volumen log) se calculan sobre la
**población completa** de cada subcategoría $c$.

**Justificación estadística:**
- **Se exigen las dos condiciones** (rating **AND** volumen) porque cada una sola tiene un
  modo de fallo: rating alto con 3 reseñas no es éxito; mucho volumen con rating malo tampoco.
- **Relativo a la subcategoría:** un buen volumen en Fragrance (5% del mercado) ≠ buen
  volumen en Hair Care (30%). Comparar dentro de $c$ elimina ese sesgo.
- **`log1p` en volumen:** por el sesgo extremo (§1.4).

## 3.2 Balance de clases

$$
\text{tasa positiva} = \frac{\#\{success=1\}}{n}
$$
Con rating > mediana (top ~50%) **y** volumen > p60 (top ~40%), y dado que están
correlacionados, la clase positiva ronda **25–35%**. Es un desbalance moderado y manejable
(justifica usar F1/AUC en vez de accuracy, y `class_weight="balanced"` en el modelo).

## 3.3 Análisis de sensibilidad del umbral (Fase 7)

El umbral (mediana + p60) es una **decisión de diseño**. Se prueban 2–3 definiciones
alternativas (p. ej. p50/p50, p50/p75) y se verifica que las conclusiones **no cambien
frágilmente** según la definición. Se reporta el % de etiquetas que "cambian de bando".

### Código — Parte III

```python
def subcategory_thresholds(df: pd.DataFrame,
                           subcat="subcategory",
                           rating="average_rating",
                           volume="rating_number",
                           price="price") -> pd.DataFrame:
    """Umbrales poblacionales por subcategoría: se calculan UNA vez en Colab
    sobre el dataset completo y se exportan como artefacto (subcategory_stats.json)."""
    g = df.groupby(subcat)
    stats_tbl = pd.DataFrame({
        "median_rating":  g[rating].median(),
        "p60_log_volume": g[volume].apply(lambda s: np.log1p(s).quantile(0.60)),
        # stats de precio SOLO sobre el subconjunto con precio (~37%)
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
    counts = y.value_counts().to_dict()
    return {"positive_rate": y.mean(),
            "counts": counts,
            "imbalance_ratio": counts.get(0, 0) / max(counts.get(1, 1), 1)}


def threshold_sensitivity(df, subcat, rating, volume,
                          rating_q=0.50, volume_q=0.60) -> pd.Series:
    """Etiqueta con un umbral alternativo (para el análisis de sensibilidad)."""
    g = df.groupby(subcat)
    med = df[subcat].map(g[rating].quantile(rating_q))
    p_v = df[subcat].map(g[volume].apply(lambda s: np.log1p(s).quantile(volume_q)))
    return ((df[rating] >= med) & (np.log1p(df[volume]) >= p_v)).astype(int)
```

---

# PARTE IV — Ingeniería de features

## 4.1 Ajuste de precio (price-fit)

En lugar del precio crudo, se mide qué tan lejos está del precio típico de su categoría,
estandarizado por el IQR (robusto):
$$
\text{price\_fit}_i = \frac{\text{price}_i - \text{mediana\_price}_c}{\text{IQR\_price}_c}
$$
0 = precio típico; positivo = caro para la categoría; negativo = barato.

## 4.2 Estandarización (z-score)

Para modelos y distancias, las features numéricas se llevan a media 0 y desv. 1:
$$
z_j = \frac{x_j - \mu_j}{\sigma_j}
$$
> **Regla:** el escalado se ajusta **solo con el set de entrenamiento** y se aplica al de
> prueba, para evitar *data leakage*.

## 4.3 TF-IDF sobre texto (`title` + `features`)

Convierte texto en números. Para el término $t$ en el documento $d$ de un corpus de $N$
documentos:
$$
\text{tf}(t,d) = \frac{\#(t \in d)}{|d|}, \qquad
\text{idf}(t) = \ln\frac{N}{1 + \text{df}(t)} + 1, \qquad
\text{tfidf}(t,d) = \text{tf}(t,d)\cdot\text{idf}(t)
$$
Luego se normaliza L2 cada vector. Da más peso a palabras distintivas ("hydration") y menos a
las comunes ("the"). El `df(t)` es en cuántos documentos aparece $t$.

## 4.4 One-hot encoding de la subcategoría

Cada una de las 8 subcategorías se convierte en una columna binaria 0/1. Es un feature
categórico limpio (a diferencia de `main_category`, que es ruido y se descarta).

### Código — Parte IV

```python
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
    """TF-IDF de title+features. Devuelve matriz dispersa y el vectorizador (a serializar)."""
    vec = TfidfVectorizer(max_features=max_features, stop_words="english",
                          ngram_range=(1, 2), sublinear_tf=True)
    X_text = vec.fit_transform(text_series.fillna(""))
    return X_text, vec


def scale_numeric(X_train, X_test):
    """StandardScaler ajustado SOLO en train (evita leakage)."""
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test), scaler
```

---

# PARTE V — Modelado y validación

## 5.1 Modelo

Clasificador de éxito: **Random Forest** o **XGBoost** (no lineales, capturan
interacciones, robustos a escalas mixtas). Con desbalance moderado se usa
`class_weight="balanced"`, que pondera cada clase por $w_k = \frac{n}{K\, n_k}$.

## 5.2 Validación cruzada estratificada k-fold

Se parte el dataset en $k$ pliegues **manteniendo la proporción de la clase positiva** en
cada uno (estratificado). Se entrena en $k-1$ y se evalúa en el restante, $k$ veces.
> **Regla:** estratificar es obligatorio bajo desbalance; un k-fold aleatorio podría dejar
> un pliegue casi sin positivos y arruinar la métrica.

## 5.3 Predicciones out-of-fold

Para calibrar y evaluar sin leakage, se obtienen probabilidades **out-of-fold**: cada
producto es predicho por un modelo que **no** lo vio en entrenamiento.

### Código — Parte V

```python
def train_and_crossval(X, y, n_splits=5, random_state=42):
    """RF con validación estratificada. Devuelve el modelo entrenado y las
    probabilidades out-of-fold (para calibración/evaluación honesta)."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    clf = RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_leaf=5,
        class_weight="balanced", n_jobs=-1, random_state=random_state)
    # probabilidades out-of-fold, sin que ningún producto se prediga a sí mismo
    proba_oof = cross_val_predict(clf, X, y, cv=skf, method="predict_proba")[:, 1]
    clf.fit(X, y)                 # modelo final sobre todo el set (a serializar)
    return clf, proba_oof
```

---

# PARTE VI — Métricas de evaluación

## 6.1 Matriz de confusión

|  | Predicho 0 | Predicho 1 |
|---|---|---|
| **Real 0** | TN | FP |
| **Real 1** | FN | TP |

## 6.2 Métricas de clasificación

$$
\text{Accuracy} = \frac{TP+TN}{TP+TN+FP+FN}, \qquad
\text{Precisión} = \frac{TP}{TP+FP}, \qquad
\text{Recall} = \frac{TP}{TP+FN}
$$
$$
F_1 = 2\cdot\frac{\text{Precisión}\cdot\text{Recall}}{\text{Precisión}+\text{Recall}}
$$

> **Regla (la más importante de esta parte):** **bajo desbalance NO se reporta accuracy sola.**
> Si 70% son clase 0, predecir siempre "0" da 70% de accuracy y es inútil. Se reportan
> **F1, precisión, recall**. Precisión = "de los que predije exitosos, cuántos lo eran";
> recall = "de los exitosos reales, cuántos detecté".

## 6.3 ROC-AUC y PR-AUC

- **ROC-AUC:** probabilidad de que el modelo ordene un positivo aleatorio por encima de un
  negativo aleatorio. 0.5 = azar, 1.0 = perfecto. Independiente del umbral.
- **PR-AUC** (average precision): área bajo precisión-recall. **Más informativa que ROC bajo
  desbalance**, porque se concentra en la clase positiva (la minoritaria y de interés).

## 6.4 Métricas de calidad probabilística (para la calibración)

- **Brier score** (menor = mejor): error cuadrático medio de la probabilidad.
$$
\text{Brier} = \frac{1}{n}\sum_i (p_i - y_i)^2
$$
- **Log loss** (penaliza fuerte la confianza equivocada):
$$
\text{LogLoss} = -\frac{1}{n}\sum_i \big[y_i \ln p_i + (1-y_i)\ln(1-p_i)\big]
$$
- **Expected Calibration Error (ECE):** mide si "score 80" ≈ 80% real. Se agrupan las
  predicciones en $M$ bins y se compara confianza vs. acierto:
$$
\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{n}\,\big|\,\text{acc}(B_m) - \text{conf}(B_m)\,\big|
$$

### Código — Parte VI

```python
def classification_report_full(y_true, proba, threshold=0.5) -> dict:
    """Todas las métricas de clasificación en un punto de corte + métricas por umbral."""
    y_pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "confusion": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_true, proba),      # independiente del umbral
        "pr_auc":    average_precision_score(y_true, proba),  # mejor bajo desbalance
        "brier":     brier_score_loss(y_true, proba),
        "log_loss":  log_loss(y_true, np.clip(proba, 1e-9, 1 - 1e-9)),
    }


def expected_calibration_error(y_true, proba, n_bins=10) -> float:
    """ECE: promedio ponderado |precisión - confianza| por bin de probabilidad."""
    y_true = np.asarray(y_true)
    bins = np.linspace(0, 1, n_bins + 1)
    ece, n = 0.0, len(proba)
    for i in range(n_bins):
        mask = (proba > bins[i]) & (proba <= bins[i + 1])
        if mask.sum() == 0:
            continue
        acc  = y_true[mask].mean()      # % real de éxito en el bin
        conf = proba[mask].mean()       # confianza media del modelo en el bin
        ece += (mask.sum() / n) * abs(acc - conf)
    return ece
```

---

# PARTE VII — Calibración e interpretabilidad

## 7.1 Calibración de probabilidades

Convierte la salida cruda del modelo en una probabilidad **honesta** (Fase 6):
- **Platt scaling** (sigmoide): ajusta $p_{\text{cal}} = \dfrac{1}{1+e^{A f + B}}$ sobre las
  puntuaciones $f$ del modelo. Bueno con pocos datos.
- **Isotónica:** ajusta una función monótona no paramétrica. Más flexible, necesita más datos.

> **Regla:** se calibra sobre las predicciones **out-of-fold**, nunca sobre las de
> entrenamiento. Sin calibración, el "80" del dashboard no significa "80% de éxito".

## 7.2 Importancia de features

- **Gini / MDI** (impureza media que reduce cada feature en el bosque): rápida pero
  sesgada hacia variables de alta cardinalidad.
- **Importancia por permutación** (preferida): se baraja aleatoriamente una feature y se mide
  cuánto **cae la métrica**; la caída = importancia. Es agnóstica al modelo y más fiable.

## 7.3 Valores SHAP

Reparten la predicción entre las features de forma **local** (por producto) y justa (teoría
de juegos, valores de Shapley). Responden el "¿por qué este producto tiene score 80?".
Contribución de la feature $j$:
$$
\phi_j = \sum_{S \subseteq F\setminus\{j\}} \frac{|S|!\,(|F|-|S|-1)!}{|F|!}\big[f(S\cup\{j\}) - f(S)\big]
$$

### Código — Parte VII

```python
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
    """Valores SHAP (interpretabilidad local). Requiere `pip install shap`.
    Correr en Colab; NO cargar en la app de Streamlit."""
    import shap
    explainer = shap.TreeExplainer(clf)
    return explainer.shap_values(X_sample)
```

---

# PARTE VIII — Métricas derivadas del dashboard

Estas son las que ve el usuario final. Se construyen sobre el modelo calibrado.

## 8.1 Score de éxito (0–100)

$$
\text{success\_score} = \text{round}\big(100 \cdot p_{\text{cal}}\big)
$$
El "80%" del dashboard. Significa: de productos muy similares, ~80% tuvieron éxito.

## 8.2 Incertidumbre del modelo

En Random Forest, la dispersión de las predicciones entre árboles mide cuánta confianza hay:
$$
u_i = \text{std}\big(\{\,p^{(t)}_i : t=1..T\,\}\big)
$$
Alta cuando el producto cae en una zona con pocos ejemplos históricos.

## 8.3 Similitud coseno y comparables (k-NN)

Para el producto de entrada $x$ y cada producto $i$ del catálogo:
$$
\text{sim}(x, i) = \cos(\mathbf{x}, \mathbf{v}_i) = \frac{\mathbf{x}\cdot\mathbf{v}_i}{\|\mathbf{x}\|\,\|\mathbf{v}_i\|} \in [-1, 1]
$$
Los $k$ mayores son los "productos similares" del dashboard. Se usa coseno (no distancia
euclídea) porque compara **dirección/perfil** del vector de features, no magnitud.

## 8.4 Saturación del mercado (0–100)

Densidad local: si el producto tiene muchos vecinos muy parecidos, el mercado está saturado:
$$
\text{saturation} = \text{percentil}\Big(\overline{\text{sim}}_k(x)\Big),
\quad \overline{\text{sim}}_k(x) = \frac{1}{k}\sum_{i \in \text{kNN}(x)} \text{sim}(x, i)
$$
Se normaliza como percentil contra la distribución de densidades del catálogo → 0–100.

## 8.5 Índice de riesgo (0–100)

**Decisión de diseño** (documentarla como tal): combina lo que el score no captura —
downside directo, saturación y la incertidumbre del modelo:
$$
\text{risk} = 100 \cdot \big[\,w_1(1 - p_{\text{cal}}) + w_2\,\text{sat} + w_3\,u\,\big],
\quad (w_1, w_2, w_3) = (0.5,\ 0.3,\ 0.2)
$$
con $\text{sat}, u \in [0,1]$. Los pesos se validan; no es una probabilidad, es un índice.

## 8.6 Precio sugerido (barrido de precio)

Se fija todo excepto el precio y se recalcula el score sobre una grilla; el recomendado es el
que maximiza el éxito dentro de un rango de márgenes razonable:
$$
\text{precio}^{*} = \arg\max_{p \in [p_{10,c},\, p_{90,c}]} \; 100\cdot p_{\text{cal}}(x \mid \text{price}=p)
$$
Esto genera literalmente el "si cambias a \$27.50 → 85%". El **rango sugerido** es
$[p_{25,c}, p_{75,c}]$ (regresión cuantílica opcional).

## 8.7 Intervalo de confianza de la tasa de éxito de comparables — Wilson

Cuando muestras "8 de 10 similares tuvieron éxito", ese 80% tiene incertidumbre. El
**intervalo de Wilson** es correcto para proporciones (mejor que el normal, sobre todo con
$n$ pequeño o $p$ cerca de 0/1):
$$
\hat{p} \pm z\,\sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}} \Big/ \Big(1 + \frac{z^2}{n}\Big)
$$
con $z=1.96$ para 95%. Comunica honestamente la incertidumbre del score al usuario.

### Código — Parte VIII

```python
def success_score(p_cal) -> int:
    """Probabilidad calibrada -> score 0-100."""
    return int(round(100 * float(p_cal)))


def rf_uncertainty(rf: RandomForestClassifier, X) -> np.ndarray:
    """Desviación estándar de la probabilidad entre árboles (incertidumbre por producto)."""
    per_tree = np.stack([est.predict_proba(X)[:, 1] for est in rf.estimators_], axis=0)
    return per_tree.std(axis=0)


def build_comparables_index(V_catalog):
    """Índice k-NN por similitud coseno (a serializar como artefacto)."""
    nn = NearestNeighbors(n_neighbors=10, metric="cosine")
    nn.fit(V_catalog)
    return nn


def find_comparables(nn, x_vec, k=4):
    """Devuelve (índices, similitudes) de los k productos más parecidos.
    similitud coseno = 1 - distancia coseno."""
    dist, idx = nn.kneighbors(x_vec.reshape(1, -1), n_neighbors=k)
    sims = 1 - dist.ravel()
    return idx.ravel(), sims


def market_saturation(x_vec, nn, density_reference, k=10) -> float:
    """Percentil (0-100) de la densidad local del producto vs. el catálogo."""
    dist, _ = nn.kneighbors(x_vec.reshape(1, -1), n_neighbors=k)
    local_density = (1 - dist.ravel()).mean()          # sim media a los k vecinos
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
        row = base_row.copy()
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
```

---

# MÓDULO CONSOLIDADO Y EJECUTABLE (`metrics.py`)

Copia todo lo anterior en un archivo `metrics.py`. El bloque siguiente es una **demo
end-to-end** con datos sintéticos que ejercita cada fórmula, para verificar que todo corre.
Guárdalo al final del archivo.

```python
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    N = 4000
    subcats = ["Hair Care", "Skin Care", "Makeup", "Fragrance"]

    # --- Dataset sintético que imita el perfil real (§7) ---
    df = pd.DataFrame({
        "subcategory":    rng.choice(subcats, N, p=[0.45, 0.30, 0.15, 0.10]),
        "average_rating": np.clip(rng.normal(4.06, 0.6, N), 1, 5),        # media ~4.06
        "rating_number":  rng.lognormal(3.0, 1.5, N).astype(int),         # muy sesgado
    })
    # price: 63% nulo, sesgado a la derecha
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

    # Comparar precio entre exitosos y no exitosos (subconjunto con precio)
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
            density_reference=np.array([ (1 - nn.kneighbors(X[j].reshape(1,-1),
                                          n_neighbors=10)[0].ravel()).mean()
                                         for j in range(0, N, 50) ]))
    print("  comparables idx:", idx, "sims:", np.round(sims, 3))
    print("  saturación:", round(sat, 1),
          "| riesgo:", risk_index(p_cal[0], sat/100, u))
    price_idx = 0  # price_fit está en la columna 0 de X_num
    best, curve = suggested_price(calibrated, x0, price_idx,
                                  price_grid=np.linspace(-2, 2, 9))
    print("  precio óptimo (en unidades de price_fit):", round(best, 2))
    print("  IC Wilson de 'éxito de comparables' (ej. 3/4):", wilson_interval(3, 4))

    print("\nOK — todas las fórmulas corrieron.")
```

---

## Tabla resumen: qué métrica va en qué fase

| Fase del proyecto | Métricas / fórmulas | Parte |
|---|---|---|
| Limpieza | descriptiva, skew/kurtosis, log1p, Tukey/MAD, tasa de nulos | I |
| EDA | Pearson, Spearman, normalidad, Mann-Whitney/Welch, Cohen's d, χ²/Cramér's V | II |
| Target | fórmula de éxito, balance de clases, sensibilidad del umbral | III |
| Features | price_fit, z-score, TF-IDF, one-hot | IV |
| Modelado | RF/XGBoost, k-fold estratificado, out-of-fold | V |
| Evaluación | matriz de confusión, F1/precisión/recall, ROC-AUC, PR-AUC, Brier, LogLoss, ECE | VI |
| Interpretab. | Platt/isotónica, importancia por permutación, SHAP | VII |
| Dashboard | score 0-100, incertidumbre RF, coseno/k-NN, saturación, riesgo, precio sugerido, Wilson | VIII |

## Limitaciones a declarar (honestidad estadística)

- **El éxito es un proxy** (rating + volumen), no ventas reales.
- **63% de precios nulos:** todo lo dependiente de precio usa solo el ~37% con dato real.
- **El riesgo (8.5) es un índice de diseño**, no una probabilidad; sus pesos se validan.
- **La tendencia de demanda "over time" no sale de la metadata** (es una foto fija): requiere
  los timestamps del archivo de reseñas o debe etiquetarse como ilustrativa.
- **El modelo falla en subcategorías genuinamente nuevas** sin comparables históricos.
