# Modelo 09 · Predicción del precio del aguacate

<!-- Notas para LaTeX: cada sección corresponde a un criterio de la rúbrica. Las figuras están en
     figuras/*.png y las métricas exactas en metricas.json (se regeneran con train.py). Las referencias
     [Clave] están en docs_latex/referencias.bib. -->

## 1. Análisis del problema (0.5 pts)

Se plantea estimar el precio promedio de un aguacate Hass en el mercado minorista de Estados Unidos a partir de
tres datos que el usuario conoce: la **región**, el **tipo** (convencional u orgánico) y la **fecha**. Se trata de
un problema de **regresión supervisada** sobre una serie de tiempo en panel (muchas series paralelas, una por
región y tipo), cuya variable objetivo es `AveragePrice`, en dólares por unidad.

La pregunta tiene una particularidad: el usuario puede consultar fechas posteriores a los datos. Por ello, la
evaluación debe medir la capacidad de **predecir el futuro** y no solo de reproducir el pasado.

## 2. Entendimiento de los datos (0.5 pts)

El conjunto reúne datos semanales del Hass Avocado Board, recopilados en Kaggle [Kiggins2018]: 18 249 filas,
169 semanas (del 4 de enero de 2015 al 25 de marzo de 2018), 54 regiones y 2 tipos. No hay valores nulos ni
filas duplicadas. Hay 3 semanas con 107 filas en lugar de 108 (falta una combinación de región y tipo).

| Variable                  | Tipo        | Uso                                                                         |
|---------------------------|-------------|-----------------------------------------------------------------------------|
| `AveragePrice`            | objetivo    | Precio por aguacate (USD): media 1.41, desviación 0.40, rango 0.44 a 3.25     |
| `Date`                    | fecha       | Semana (domingo). Se transforma en mes, semana del año y tendencia          |
| `region`                  | categórica  | 54 valores; incluye agregados (`TotalUS`, `West`, …) que se solapan con ciudades |
| `type`                    | categórica  | `conventional` u `organic`                                                  |
| `Total Volume`, `4046`, `4225`, `4770`, `*Bags` | numéricas | **Excluidas** |
| `Unnamed: 0`, `year`      | índice / redundante | Descartadas (`year` se deriva de `Date`)                            |

**Exclusión de los volúmenes.** Los volúmenes de venta se registran en la misma semana que el precio y el
usuario no los conoce cuando consulta. Usarlos produciría un modelo que no puede ejecutarse en la aplicación.

**Regiones agregadas.** Se conservan las 54 regiones como opciones de consulta, porque `TotalUS` es la consulta
más natural. Esto implica que las regiones no son independientes (el total nacional incluye a las ciudades), lo
que no afecta a la predicción de un punto pero sí a cualquier conclusión estadística sobre las regiones.

## 3. Exploración de los datos (0.5 pts)

- `figuras/distribucion_objetivo.png`: distribución con dos concentraciones suaves (alrededor de 1.1 y 1.4 USD, reflejo de los dos tipos, que se solapan
  bastante: el convencional tiene media 1.16 y desviación 0.26, y el orgánico, 1.65 y 0.36) y cola derecha hasta 3.25 USD.
- `figuras/estacionalidad.png`: el precio medio general sube del mínimo de febrero (1.27) al máximo de octubre (1.58). En ambos
  tipos el mínimo es en febrero y el máximo en septiembre (orgánico) u octubre (convencional): hay estacionalidad anual clara.
- El orgánico cuesta en promedio 1.65 USD frente a 1.16 del convencional.
- `figuras/serie_nacional.png`: el nivel de precios cambia de un año a otro de forma no estacional: la media
  anual fue 1.38 (2015), 1.34 (2016), 1.52 (2017) y 1.35 (2018, solo hasta marzo). En agosto–octubre de 2017
  hay un pico marcado (el precio nacional convencional pasó de 1.33 a 1.65 USD).
- `figuras/regiones_extremas.png`: la región más barata es Houston (1.05) y la más cara, Hartford–Springfield
  (1.82); San Francisco ocupa el puesto 53 de 54 por precio (1.80).

**Anomalía detectada.** El precio orgánico nacional (`TotalUS`) queda exactamente en 1.00 durante 6 semanas
consecutivas (del 5 de julio al 9 de agosto de 2015), mientras que las semanas vecinas valen 1.64 y 1.75. Es
un artefacto evidente de los datos, pero afecta solo a 6 de 18 249 filas (0.03 %). Otras series tienen rachas de hasta 5 semanas con el mismo precio,
pero ninguna tan larga ni con un salto de unos 0.65 USD. Se conservó porque un precio de 1.00 no es imposible y su efecto es despreciable.

## 4. Modelo (2 pts)

**Partición temporal.** Se reservaron como prueba las **últimas 34 semanas** (del 6 de agosto de 2017 al 25 de
marzo de 2018; 3 672 filas) y se entrenó con las 135 anteriores (14 577 filas). A diferencia de una partición
aleatoria, esta mide la capacidad de predecir el futuro respecto del entrenamiento [Hyndman2021].

**Variables.** Un transformador propio (`preprocesamiento.py`), incluido dentro del `Pipeline`, deriva de la
fecha el mes, la semana del año y una tendencia `t` (años desde el inicio) **truncada al último valor visto**:
el modelo no extrapola la tendencia a fechas futuras (para los modelos de árboles esto no cambia nada, porque
ya son constantes más allá del último valor visto; protege a los lineales). La semana se calcula a partir del día
del año y no con el calendario ISO, que contradice al mes cerca de año nuevo. Así, el entrenamiento y la API aplican exactamente el
mismo tratamiento.

**Candidatos.** Se compararon siete variantes, con y sin tendencia:

| Candidato                         | Justificación                                                            |
|-----------------------------------|--------------------------------------------------------------------------|
| Efecto región y tipo (línea base) | Regresión lineal solo con región y tipo: el precio medio de cada serie    |
| Ridge estacional (± tendencia)    | Añade el mes como categoría; la penalización L2 estabiliza los coeficientes [Hoerl1970] |
| Random Forest (± tendencia)       | Captura interacciones región–estación sin supuestos de forma [Breiman2001] |
| Gradient Boosting (± tendencia)   | Ensamble secuencial de árboles, habitual en datos tabulares [Friedman2001]; se usó la implementación de histogramas de scikit-learn [Pedregosa2011] |

**Selección.** Se realizó solo con el conjunto de entrenamiento, mediante validación cruzada con **ventana
creciente por semanas** (3 pliegues; cada pliegue evalúa 20 semanas posteriores a las que entrena, y todas las
filas de una misma semana van juntas) [Bergmeir2012]. El conjunto de prueba se utilizó una única vez.

El modelo elegido es **Gradient Boosting con tendencia**.

## 5. Evaluación (1 pt)

**Validación cruzada en entrenamiento** (media ± desviación del RMSE entre pliegues):

| Modelo                              | RMSE (USD)      | R²    |
|-------------------------------------|-----------------|-------|
| Efecto región y tipo (línea base)   | 0.307 ± 0.016   | 0.447 |
| Ridge estacional                    | 0.303 ± 0.025   | 0.459 |
| Ridge estacional + tendencia        | 0.322 ± 0.026   | 0.388 |
| Random Forest                       | 0.330 ± 0.024   | 0.358 |
| Random Forest + tendencia           | 0.337 ± 0.008   | 0.327 |
| Gradient Boosting                   | 0.298 ± 0.021   | 0.477 |
| **Gradient Boosting + tendencia**   | **0.288 ± 0.019** | 0.509 |

Las diferencias entre los tres mejores (0.288, 0.298 y 0.303) son menores que la variación entre pliegues
(≈ 0.02): **no son estadísticamente distinguibles**. La evidencia sobre la tendencia es mixta: empeora a Ridge
y a Random Forest y mejora a Gradient Boosting. Además, el aporte de la estacionalidad respecto de la línea base
en validación cruzada es modesto (0.307 a 0.298).

**Conjunto de prueba (últimas 34 semanas):**

| Modelo                              | R²    | MAE (USD) | RMSE (USD) |
|-------------------------------------|-------|-----------|------------|
| Efecto región y tipo (línea base)   | 0.113 | 0.275     | 0.371      |
| Gradient Boosting + tendencia       | 0.418 | 0.227     | 0.301      |

El modelo reduce el RMSE de 0.371 a 0.301 USD (19 %) y el R² pasa de 0.113 a 0.418. En `figuras/prueba_nacional.png`
se observa que **no reproduce el pico de agosto a octubre de 2017**: predice una línea casi plana alrededor del
último nivel conocido, y sigue mejor la bajada de diciembre a febrero, aunque en marzo de 2018 sobrestima el convencional (1.28 frente a 1.05 reales). Esto es esperable: un
cambio de nivel de ese tipo no se puede anticipar solo con la región, el tipo y la fecha. El error está
concentrado en esas semanas.

**Despliegue.** Las métricas anteriores corresponden al modelo entrenado con la partición temporal. El modelo que
sirve la API se **reentrena con las 169 semanas** para que conozca los precios más recientes; sus métricas no
se pueden medir con datos independientes.

## 6. Conclusión (0.5 pts)

Con solo la región, el tipo y la fecha es posible estimar el precio del aguacate con un error medio de
aproximadamente 0.23 USD (RMSE 0.30), un 19 % mejor que predecir el promedio histórico de cada serie. La
estructura que sí es predecible es doble: el nivel de cada región (Houston barata, Hartford y San Francisco
caras), la prima del orgánico (+0.50 USD) y una estacionalidad anual moderada.

Lo que **no** es predecible con estas variables es el nivel general de precios, que cambia por shocks de oferta
como el de 2017. Por eso la selección entre modelos no fue concluyente y la mejora de la estacionalidad es
pequeña. Para mejorar de forma sustancial haría falta información externa (producción, clima, aranceles) o el
precio reciente del mercado como entrada.

**Limitaciones.** (1) Los datos terminan en marzo de 2018: para fechas posteriores la estimación se basa en la
estacionalidad y en el último nivel conocido, y la API lo advierte (`poco confiable`). Una consulta con la fecha
de hoy recibe esa advertencia. (2) Solo hay tres ciclos anuales completos para estimar la estacionalidad.
(3) Las regiones agregadas se solapan con las ciudades. (4) El conjunto de prueba es una sola ventana de 34
semanas, que incluye un episodio extremo, por lo que la métrica es sensible a él; además, con semillas distintas
el R² de prueba varió entre 0.40 y 0.43 (comprobación independiente), de modo que las diferencias pequeñas entre
candidatos no son concluyentes.

## Referencias (en `docs_latex/referencias.bib`)

- **[Kiggins2018]** Kiggins, J. (2018). *Avocado Prices* [Conjunto de datos del Hass Avocado Board]. Kaggle. Licencia ODbL 1.0.
- **[Friedman2001]** Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *The Annals of Statistics*, 29(5), 1189–1232. https://doi.org/10.1214/aos/1013203451
- **[Breiman2001]** Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.
- **[Hoerl1970]** Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression: Biased estimation for nonorthogonal problems. *Technometrics*, 12(1), 55–67.
- **[Bergmeir2012]** Bergmeir, C., & Benítez, J. M. (2012). On the use of cross-validation for time series predictor evaluation. *Information Sciences*. https://doi.org/10.1016/j.ins.2011.12.028 *(falta confirmar volumen y páginas)*
- **[Hyndman2021]** Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3.ª ed.). OTexts. https://otexts.com/fpp3/
- **[Pedregosa2011]** Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.

> Verificadas por búsqueda web el 2026-10-09, salvo lo indicado en Bergmeir (2012).
