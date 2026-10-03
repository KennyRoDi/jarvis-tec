# Especificación de modelos de ML — JarvisTEC

> **Estado:** propuesta inicial, pendiente de confirmación del equipo y del profesor.
> Selección balanceada: 5 de regresión y 5 de clasificación, todos tomados de la lista del enunciado.

## Convenciones (aplican a los 10 modelos)

Cada modelo vive en `backend/features/modelo_XX_<slug>/` con estos archivos:

| Archivo          | Responsable | Contenido                                                                   |
|------------------|-------------|-----------------------------------------------------------------------------|
| `dataset.csv`    | Dev A       | Datos crudos (descargados de la fuente indicada abajo)                      |
| `train.py`       | Dev A       | Entrenamiento estructurado en las 6 etapas de evaluación                    |
| `router.py`      | Dev A       | `router` de FastAPI + `MODELO_INFO` + esquema `Entrada`                     |
| `analisis.md`    | Dev A       | Resumen académico de las 6 etapas, listo para pasar a LaTeX                 |
| `modelo.joblib`  | generado    | Artefacto: `{"pipeline": Pipeline de sklearn, "metricas": {...}, ...}`      |
| `metricas.json`  | generado    | Métricas de evaluación (las lee `GET /api/modelos/{slug}/info`)             |
| `figuras/`       | generado    | Gráficos de exploración y evaluación (PNG, para Overleaf)                   |

- Entrenar un modelo: `cd backend && python -m features.modelo_XX_<slug>.train`
- El modelo se publica solo: `main.py` descubre cualquier `features/*/router.py` que exporte `router`.
- **Etapas evaluadas** (rúbrica, 5 pts c/u): 1) Análisis del problema 0.5 · 2) Entendimiento de los datos 0.5 ·
  3) Exploración de los datos 0.5 · 4) Modelo 2 · 5) Evaluación 1 · 6) Conclusión 0.5.
- Las métricas mínimas a reportar son: regresión → R², MAE, RMSE; clasificación → accuracy, precision, recall, F1 (macro) y matriz de confusión.

## Modelos seleccionados

| #  | Carpeta                       | `slug`            | Tipo                     | Variable objetivo          | Fuente del dataset |
|----|-------------------------------|-------------------|--------------------------|----------------------------|--------------------|
| 01 | `modelo_01_bitcoin`           | `bitcoin`         | Regresión (serie temporal) | `Close` del día siguiente | [Kaggle team-ai/bitcoin-price-prediction](https://www.kaggle.com/team-ai/bitcoin-price-prediction/version/1) (`bitcoin_price_Training - bitcoin_price.2013Apr-2017Aug.csv`) |
| 02 | `modelo_02_autos`             | `autos`           | Regresión                | `Selling_Price`            | [GitHub amankharwal/car data.csv](https://raw.githubusercontent.com/amankharwal/Website-data/master/car%20data.csv) ✅ descargado |
| 03 | `modelo_03_vino`              | `vino`            | Clasificación            | `quality` (agrupada)       | [Kaggle rajyellow46/wine-quality](https://www.kaggle.com/rajyellow46/wine-quality) |
| 04 | `modelo_04_churn`             | `churn`           | Clasificación binaria    | `Churn`                    | [GitHub IBM Telco-Customer-Churn.csv](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv) ✅ descargado |
| 05 | `modelo_05_acv`               | `acv`             | Clasificación binaria (desbalanceada) | `stroke`      | [Kaggle fedesoriano/stroke-prediction-dataset](https://www.kaggle.com/fedesoriano/stroke-prediction-dataset) |
| 06 | `modelo_06_hepatitis`         | `hepatitis`       | Clasificación multiclase | `Category`                 | [Kaggle fedesoriano/hepatitis-c-dataset](https://www.kaggle.com/fedesoriano/hepatitis-c-dataset) ⚠️ |
| 07 | `modelo_07_cirrosis`          | `cirrosis`        | Clasificación multiclase | `Stage`                    | [Kaggle fedesoriano/cirrhosis-prediction-dataset](https://www.kaggle.com/fedesoriano/cirrhosis-prediction-dataset) |
| 08 | `modelo_08_grasa_corporal`    | `grasa_corporal`  | Regresión                | `BodyFat`                  | [Kaggle fedesoriano/body-fat-prediction-dataset](https://www.kaggle.com/fedesoriano/body-fat-prediction-dataset) ⚠️ |
| 09 | `modelo_09_aguacate`          | `aguacate`        | Regresión                | `AveragePrice`             | [Kaggle neuromusic/avocado-prices](https://www.kaggle.com/neuromusic/avocado-prices) |
| 10 | `modelo_10_sp500`             | `sp500`           | Regresión (serie temporal) | `close` del día siguiente | [Kaggle camnugent/sandp500](https://www.kaggle.com/camnugent/sandp500) |

⚠️ En el enunciado los enlaces de *hepatitis* y *masa corporal* están intercambiados; arriba se usan los correctos.

Los datasets de Kaggle requieren iniciar sesión. Descarga manual o con la CLI:
`kaggle datasets download -d <usuario>/<dataset> --unzip -p backend/features/modelo_XX_<slug>/`
y renombrar el CSV principal a `dataset.csv`.

## Detalle por modelo

Cada modelo debe completar aquí su **esquema de entrada** (lo que recibe `POST /api/modelos/{slug}/predecir`)
antes de implementar el `router.py`. Los nombres van en `snake_case`.

### 01 · bitcoin
- **Pregunta:** ¿cuál será el precio de cierre del Bitcoin mañana?
- **Algoritmos candidatos:** Regresión lineal con rezagos (lags), Random Forest Regressor, Gradient Boosting.
- **Entrada:** _por definir_ (p. ej. `dias_adelante: int`, usando los últimos precios del dataset).
- **Comandos:** "precio del bitcoin", "bitcoin mañana", "tipo de cambio del bitcoin".

### 02 · autos — ✅ modelo de referencia implementado
- **Pregunta:** ¿cuál es el precio de reventa de un automóvil usado?
- **Algoritmos:** Random Forest Regressor (implementado); comparar con Regresión lineal.
- **Entrada:** `year: int`, `present_price: float`, `kms_driven: int`, `fuel_type: "Petrol"|"Diesel"|"CNG"`,
  `seller_type: "Dealer"|"Individual"`, `transmission: "Manual"|"Automatic"`, `owner: int`.
- **Unidad:** lakhs INR (1 lakh = 100 000 rupias).
- **Comandos:** "precio de un auto", "precio de un carro", "cuánto vale mi carro".

### 03 · vino
- **Pregunta:** ¿el vino es de calidad baja, media o alta?
- **Algoritmos candidatos:** Random Forest Classifier, SVM, Regresión logística.
- **Entrada:** _por definir_ (11 variables fisicoquímicas + `type`).
- **Comandos:** "calidad del vino", "clasificar vino".

### 04 · churn
- **Pregunta:** ¿el cliente abandonará la compañía telefónica?
- **Algoritmos candidatos:** Regresión logística, Random Forest, Gradient Boosting.
- **Entrada:** _por definir_ (subconjunto de columnas de `dataset.csv`; `TotalCharges` viene como texto con vacíos).
- **Comandos:** "cliente se va", "abandono de cliente", "churn".

### 05 · acv
- **Pregunta:** ¿el paciente tiene riesgo de sufrir un accidente cerebrovascular?
- **Algoritmos candidatos:** Regresión logística con `class_weight="balanced"`, Random Forest; considerar SMOTE.
- **Entrada:** _por definir_.
- **Comandos:** "riesgo de derrame", "accidente cerebrovascular".

### 06 · hepatitis
- **Pregunta:** ¿qué categoría de hepatitis C presenta el paciente (donante, hepatitis, fibrosis, cirrosis)?
- **Algoritmos candidatos:** KNN, Random Forest, SVM.
- **Entrada:** _por definir_.
- **Comandos:** "tipo de hepatitis", "diagnóstico de hepatitis".

### 07 · cirrosis
- **Pregunta:** ¿en qué etapa histológica de cirrosis está el paciente?
- **Algoritmos candidatos:** Random Forest, Gradient Boosting, Regresión logística multinomial.
- **Entrada:** _por definir_.
- **Comandos:** "etapa de cirrosis", "tipo de cirrosis".

### 08 · grasa_corporal
- **Pregunta:** ¿qué porcentaje de grasa corporal tiene el paciente según sus medidas?
- **Algoritmos candidatos:** Regresión lineal, Ridge/Lasso, Random Forest Regressor.
- **Entrada:** _por definir_.
- **Comandos:** "grasa corporal", "masa corporal".

### 09 · aguacate
- **Pregunta:** ¿cuál será el precio promedio del aguacate en una región y fecha?
- **Algoritmos candidatos:** Random Forest Regressor, Gradient Boosting.
- **Entrada:** _por definir_.
- **Comandos:** "precio del aguacate".

### 10 · sp500
- **Pregunta:** ¿cuál será el precio de cierre de una acción del S&P 500 mañana?
- **Algoritmos candidatos:** Regresión lineal con rezagos, Random Forest Regressor.
- **Entrada:** _por definir_ (p. ej. `simbolo: str`).
- **Comandos:** "precio de la acción", "bolsa", "sp500".
