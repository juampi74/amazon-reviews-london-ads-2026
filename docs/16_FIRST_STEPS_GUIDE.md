# First Steps Guide / Guía de Primeros Pasos

## 1. What to do first / Qué hacer primero

### English

Do not start by building the full interface. Start by locking the product logic:

1. Confirm the MVP scope.
2. Freeze the P0 requirements.
3. Verify the cleaned datasets.
4. Define the target.
5. Build one baseline model.
6. Expose the baseline through an API.
7. Connect the frontend after the model is stable.

The frontend can change later. The workflow cannot.

### Español

No empieces construyendo toda la interfaz. Empieza fijando la lógica del producto:

1. Confirmar el alcance del MVP.
2. Congelar los requerimientos P0.
3. Verificar los datasets limpios.
4. Definir el target.
5. Construir un modelo baseline.
6. Exponer el baseline por API.
7. Conectar el frontend cuando el modelo ya sea estable.

El frontend puede cambiar después. El flujo no debe cambiar.

## 2. MVP focus / En qué enfocarse para el MVP

### English

Focus only on the requirements that create the first valuable demo:

- P0 data requirements.
- P0 functional requirements.
- P0 analysis outputs.
- P0 storage and hosting decisions.
- P0 validation and QA.

Do not start with optional P1 or P2 features unless the core path is already working.

### Español

Enfóquense solo en los requerimientos que permiten la primera demo con valor:

- Requerimientos de datos P0.
- Requerimientos funcionales P0.
- Outputs P0.
- Decisiones de storage y hosting P0.
- Validación y QA P0.

No empiecen con funciones opcionales P1 o P2 salvo que el camino principal ya funcione.

## 3. Recommended tools / Herramientas recomendadas

| Area | English | Español |
|---|---|---|
| Data prep | Python, Pandas or Polars, DuckDB | Python, Pandas o Polars, DuckDB |
| Experiments | Google Colab | Google Colab |
| Modeling | scikit-learn, CatBoost, Sentence Transformers | scikit-learn, CatBoost, Sentence Transformers |
| API | FastAPI | FastAPI |
| UI | Next.js or another frontend framework if the team changes later | Next.js o cualquier frontend si luego cambia |
| Storage | Oracle Cloud Infrastructure Object Storage | Oracle Cloud Infrastructure Object Storage |
| App DB | Supabase PostgreSQL and RLS | Supabase PostgreSQL y RLS |
| Validation | Power BI | Power BI |
| Versioning | GitHub | GitHub |

## 4. Why OCI for datasets / Por qué OCI para los datasets

### English

Use Oracle Cloud Infrastructure Object Storage for the heavy datasets because:

- The raw data is larger than Git should hold.
- It separates the data lake from the application database.
- It can store Parquet, embeddings, and model artifacts.
- It fits a production-oriented pipeline better than local storage.

### Español

Usen Oracle Cloud Infrastructure Object Storage para los datasets pesados porque:

- Los datos raw son demasiado grandes para Git.
- Separa el data lake de la base de datos de la aplicación.
- Puede guardar Parquet, embeddings y artefactos del modelo.
- Se adapta mejor a un pipeline productivo que el storage local.

## 5. What to build first / Qué construir primero

### English

Build in this order:

1. Dataset A for training.
2. Target definition and leakage audit.
3. Baseline model.
4. Calibration.
5. Comparable products.
6. Risk and profit formulas.
7. FastAPI endpoints.
8. Frontend screens.
9. Power BI validation.

### Español

Construyan en este orden:

1. Dataset A para entrenamiento.
2. Definición del target y auditoría de leakage.
3. Modelo baseline.
4. Calibración.
5. Productos comparables.
6. Fórmulas de risk y profit.
7. Endpoints de FastAPI.
8. Pantallas del frontend.
9. Validación en Power BI.

## 6. P0 requirements checklist / Checklist de requerimientos P0

| Requirement | Why it matters | Prioridad |
|---|---|---:|
| Clean taxonomy | Keeps comparables consistent | P0 |
| Title and description input | Creates semantic features | P0 |
| Price and cost inputs | Enables profit and price logic | P0 |
| Success Score | Core model output | P0 |
| Decision Risk | Business-readable index | P0 |
| Comparable products | Makes the result credible | P0 |
| Saturation | Shows competition density | P0 |
| Analysis persistence | Needed for history and audit | P0 |
| Power BI export | Needed for validation and reporting | P0 |
| Disclaimers | Prevents overclaiming | P0 |

## 7. Five-person team split / División de 5 personas

### Person 1 - Data engineer / Ingeniero de datos

**English**

- Clean and version datasets.
- Build Dataset A, B, and C.
- Create manifests and exports.
- Store heavy files in OCI Object Storage.

**Español**

- Limpiar y versionar los datasets.
- Construir Dataset A, B y C.
- Crear manifests y exports.
- Guardar archivos pesados en OCI Object Storage.

### Person 2 - Data scientist / Científico de datos

**English**

- Define the target.
- Build the baseline model.
- Calibrate the output.
- Audit leakage and model quality.

**Español**

- Definir el target.
- Construir el modelo baseline.
- Calibrar la salida.
- Auditar leakage y calidad del modelo.

### Person 3 - Backend engineer / Ingeniero backend

**English**

- Build FastAPI.
- Expose prediction and analysis endpoints.
- Connect the model to storage and Supabase.
- Add request validation and logs.

**Español**

- Construir FastAPI.
- Exponer endpoints de predicción y análisis.
- Conectar el modelo con storage y Supabase.
- Agregar validación y logs.

### Person 4 - Frontend engineer / Ingeniero frontend

**English**

- Build the MVP UI.
- Show score, risk, profit, and comparable products.
- Keep the interface simple and readable.
- Adjust the design if the current HTML changes.

**Español**

- Construir la UI del MVP.
- Mostrar score, risk, profit y comparables.
- Mantener la interfaz simple y legible.
- Ajustar el diseño si el HTML actual cambia.

### Person 5 - QA, BI, and documentation / QA, BI y documentación

**English**

- Build Power BI validation views.
- Write the bilingual docs.
- Test the end-to-end flow.
- Track acceptance criteria.

**Español**

- Construir las vistas de validación en Power BI.
- Escribir la documentación bilingue.
- Probar el flujo end-to-end.
- Seguir los criterios de aceptación.

## 8. Suggested working sequence / Secuencia sugerida de trabajo

### English

Week 1:

- Finish data cleaning and dataset split.
- Define the target and the leakage rules.
- Confirm the storage architecture.

Week 2:

- Train the baseline.
- Calibrate the score.
- Validate comparable products.

Week 3:

- Build FastAPI.
- Build the MVP frontend.
- Connect persistence in Supabase.

Week 4:

- Add Power BI exports.
- Run QA.
- Polish documentation and demo flow.

### Español

Semana 1:

- Terminar cleaning y split de datasets.
- Definir target y reglas de leakage.
- Confirmar la arquitectura de storage.

Semana 2:

- Entrenar el baseline.
- Calibrar el score.
- Validar los comparables.

Semana 3:

- Construir FastAPI.
- Construir el frontend del MVP.
- Conectar la persistencia en Supabase.

Semana 4:

- Agregar exports para Power BI.
- Hacer QA.
- Pulir documentación y demo.

## 9. First milestone / Primer hito

### English

The first milestone is reached when the team can input one product and see:

- Success Score.
- Decision Risk.
- Profit per Sale.
- Comparable products.
- Suggested price range.
- A visible disclaimer.

### Español

El primer hito se alcanza cuando el equipo puede ingresar un producto y ver:

- Success Score.
- Decision Risk.
- Profit per Sale.
- Productos comparables.
- Rango de precio sugerido.
- Un disclaimer visible.
