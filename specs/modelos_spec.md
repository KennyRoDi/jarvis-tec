# Especificación de modelos de ML — JarvisTEC

> Selección balanceada: 5 de regresión y 5 de clasificación, **todos de la lista del enunciado**, por lo
> que no requieren validación de dataset por parte del profesor (requerimiento 3).
> El detalle de cada modelo (datos, enfoque, entrada, criterios de aceptación) está en su `SPEC.md`.

## Convenciones (aplican a los 10 modelos)

Cada modelo vive en `backend/features/modelo_XX_<slug>/` con estos archivos:

| Archivo          | Responsable | Contenido                                                                   |
|------------------|-------------|-----------------------------------------------------------------------------|
| `dataset.csv`    | Dev A       | Datos crudos (descargados de la fuente indicada abajo)                      |
| `train.py`       | Dev A       | Entrenamiento estructurado en las 6 etapas de evaluación                    |
| `router.py`      | Dev A       | `router` de FastAPI + `MODELO_INFO` + esquema `Entrada`                     |
| `analisis.md`    | Dev A       | Resumen académico de las 6 etapas, listo para pasar a LaTeX                 |
| `SPEC.md`        | Dev A       | Especificación y criterios de aceptación del modelo                         |
| `test_modelo.py` | Dev A       | Pruebas del endpoint con el modelo real                                     |
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
| 01 | `modelo_01_bitcoin`           | `bitcoin`         | Regresión (serie temporal) | retorno logarítmico del día siguiente (el precio es `Close`) | [Kaggle team-ai/bitcoin-price-prediction](https://www.kaggle.com/team-ai/bitcoin-price-prediction/version/1) (`bitcoin_price_Training - Training.csv`) |
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

## Especificación por modelo

| #  | SPEC                                                                            | ¿Se ejecuta solo con la voz? | Estado |
|----|---------------------------------------------------------------------------------|------------------------------|--------|
| 01 | [bitcoin](../backend/features/modelo_01_bitcoin/SPEC.md)                        | Sí                           | ✅ entrenado (falta interfaz) |
| 02 | [autos](../backend/features/modelo_02_autos/SPEC.md)                            | No (formulario)              | ✅ entrenado (falta interfaz y referencias) |
| 03 | [vino](../backend/features/modelo_03_vino/SPEC.md)                              | No                           | ✅ entrenado (falta interfaz) |
| 04 | [churn](../backend/features/modelo_04_churn/SPEC.md)                            | No                           | ✅ entrenado (falta interfaz) |
| 05 | [acv](../backend/features/modelo_05_acv/SPEC.md)                                | No                           | ✅ entrenado (falta interfaz) |
| 06 | [hepatitis](../backend/features/modelo_06_hepatitis/SPEC.md)                    | No                           | ✅ entrenado (falta interfaz) |
| 07 | [cirrosis](../backend/features/modelo_07_cirrosis/SPEC.md)                      | No                           | ✅ entrenado (falta interfaz) |
| 08 | [grasa_corporal](../backend/features/modelo_08_grasa_corporal/SPEC.md)          | No                           | ✅ entrenado (falta interfaz) |
| 09 | [aguacate](../backend/features/modelo_09_aguacate/SPEC.md)                      | Sí (valores por defecto)     | ✅ entrenado (falta interfaz) |
| 10 | [sp500](../backend/features/modelo_10_sp500/SPEC.md)                            | Sí                           | ⏳ |
