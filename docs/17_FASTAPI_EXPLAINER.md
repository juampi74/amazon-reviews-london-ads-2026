# FastAPI Explainer / Explicacion de FastAPI

## 1. What FastAPI is / Que es FastAPI

### English

FastAPI is a Python framework for building APIs. In this project, it is the layer that receives a request from the frontend, runs the machine learning logic, and returns a structured response.

### Español

FastAPI es un framework de Python para construir APIs. En este proyecto es la capa que recibe la solicitud del frontend, ejecuta la logica de machine learning y devuelve una respuesta estructurada.

## 2. Why use it here / Por que usarlo aqui

### English

Use FastAPI because it:

- Works naturally with Python ML code.
- Makes request and response schemas explicit.
- Is fast enough for inference services.
- Integrates well with validation, logging, and async jobs.
- Keeps the frontend and model separated.

### Español

Usen FastAPI porque:

- Funciona de forma natural con codigo Python de ML.
- Hace explicitos los esquemas de request y response.
- Es suficientemente rapido para inference services.
- Se integra bien con validacion, logs y jobs asincronos.
- Separa el frontend del modelo.

## 3. What FastAPI does in Launchly / Que hace FastAPI en Launchly

### English

FastAPI should expose endpoints such as:

- `POST /v1/predict/success`
- `POST /v1/comparables`
- `POST /v1/price-scenarios`
- `POST /v1/profit/calculate`
- `POST /v1/analyses`

The API receives product inputs, loads the model artifact, computes the outputs, and returns JSON for the UI and Supabase.

### Español

FastAPI debe exponer endpoints como:

- `POST /v1/predict/success`
- `POST /v1/comparables`
- `POST /v1/price-scenarios`
- `POST /v1/profit/calculate`
- `POST /v1/analyses`

La API recibe los inputs del producto, carga el artefacto del modelo, calcula las salidas y devuelve JSON para la UI y Supabase.

## 4. Why not put ML directly in the frontend / Por que no poner el ML directo en el frontend

### English

Do not run the model in the browser because:

- The model artifacts can be heavy.
- The logic would be harder to secure.
- The code would be harder to version.
- Reusing the model for Power BI, validation, and batch jobs becomes harder.

### Español

No ejecuten el modelo en el navegador porque:

- Los artefactos del modelo pueden pesar mucho.
- La logica seria mas dificil de asegurar.
- El codigo seria mas dificil de versionar.
- Reutilizar el modelo para Power BI, validacion y batch jobs seria mas dificil.

## 5. Why not only Supabase / Por que no solo Supabase

### English

Supabase is great for authentication, database storage, and row-level security, but it is not the machine learning runtime. It stores data and results; FastAPI performs inference.

### Español

Supabase es excelente para autenticacion, almacenamiento en base de datos y row-level security, pero no es el runtime de machine learning. Supabase guarda datos y resultados; FastAPI hace la inference.

## 6. Simple flow / Flujo simple

```text
Frontend -> FastAPI -> Model / Rules -> Supabase -> Frontend
```

### English

1. The user submits a product.
2. FastAPI validates the payload.
3. FastAPI loads the model and supporting artifacts.
4. The model computes score, risk, profit, and comparables.
5. FastAPI saves or returns the analysis.
6. The frontend displays the result.

### Español

1. El usuario envia un producto.
2. FastAPI valida el payload.
3. FastAPI carga el modelo y los artefactos de apoyo.
4. El modelo calcula score, risk, profit y comparables.
5. FastAPI guarda o devuelve el analisis.
6. El frontend muestra el resultado.

## 7. What to store near the API / Que guardar cerca de la API

### English

The API should load only the minimum artifacts:

- Serialized model or pipeline.
- Calibration table.
- Comparable index.
- Threshold tables.
- Fee lookup tables.

Do not load the full raw dataset into the API.

### Español

La API debe cargar solo los artefactos minimos:

- Modelo o pipeline serializado.
- Tabla de calibracion.
- Indice de comparables.
- Tablas de thresholds.
- Tablas de fees.

No carguen el dataset raw completo dentro de la API.

## 8. Why FastAPI is a good MVP choice / Por que FastAPI es una buena eleccion para el MVP

### English

FastAPI is a good MVP choice because it gives the team a clean boundary between:

- data science,
- business logic,
- and frontend presentation.

That boundary makes the project easier to test, easier to deploy, and easier to explain.

### Español

FastAPI es una buena eleccion para el MVP porque da una frontera clara entre:

- data science,
- logica de negocio,
- y presentacion en frontend.

Esa frontera hace que el proyecto sea mas facil de probar, desplegar y explicar.
