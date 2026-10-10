# Modelo 05 · Clasificación del riesgo de accidente cerebrovascular (ACV)

<!-- Notas para LaTeX: cada sección corresponde a un criterio de la rúbrica. Las figuras están en
     figuras/*.png y las métricas exactas en metricas.json (se regeneran con train.py). Las referencias
     [Clave] están en docs_latex/referencias.bib. -->

> **Aviso.** Este modelo estima un riesgo estadístico con fines educativos. **No es una herramienta de diagnóstico**
> ni reemplaza la valoración de un profesional de la salud.

## 1. Análisis del problema (0.5 pts)

El accidente cerebrovascular (ACV) es una de las principales causas de muerte y discapacidad, y sus factores de
riesgo más importantes (edad, hipertensión, enfermedad cardíaca, diabetes) son conocidos y medibles [Boehme2017].
Se plantea estimar la **probabilidad de que un paciente sufra un ACV** a partir de datos clínicos básicos. Es un
problema de **clasificación binaria supervisada muy desbalanceada**: solo 4.9 % de los pacientes tuvo un ACV, de modo
que un clasificador que siempre responda "no" tiene 95 % de exactitud y no detecta a nadie. Por eso la exactitud no
sirve como métrica y se evalúa con el AUC ROC, el AUC de precisión-recall, la calibración y el recall de la clase
positiva [Saito2015].

El asistente JarvisTEC responde a consultas como "¿qué riesgo de derrame tengo?" con una probabilidad y un nivel de
riesgo (bajo, moderado o alto).

## 2. Entendimiento de los datos (0.5 pts)

El conjunto de Kaggle [Soriano2021] reúne 5 110 pacientes y 12 columnas: identificador, género, edad, hipertensión,
enfermedad cardíaca, estado civil, tipo de trabajo, tipo de residencia, nivel promedio de glucosa, índice de masa
corporal (`bmi`), tabaquismo y el objetivo `stroke` (249 positivos). No hay filas duplicadas. **El origen del
conjunto está declarado como confidencial**, por lo que no se conoce la población ni cómo se definió el ACV (sección
6).

**Valores faltantes informativos.** Hay 201 valores nulos en `bmi` (3.9 %). No son aleatorios: **19.9 % de los
pacientes sin `bmi` tuvo un ACV, frente a 4.3 % de quienes sí lo tienen** (40 de los 249 casos positivos carecen de
`bmi`). Los datos faltantes que dependen del resultado son un mecanismo problemático [Little2019]. Un indicador de
"`bmi` faltante" subiría el AUC de validación cruzada de 0.842 a 0.855, pero es un artefacto de la recolección y
**no es utilizable**: la aplicación siempre solicita los datos completos.

**Hallazgo de la verificación: el artefacto se filtraba por la imputación.** En una primera versión se imputaba el
`bmi` con la mediana y se usaba un Random Forest. Ese modelo obtenía un AUC de validación cruzada de 0.8455, pero
al excluir el `bmi` bajaba a 0.8347: el árbol aislaba el pico de valores imputados (28.0) y "descubría" que el `bmi`
faltante predice el ACV, sin que nadie lo hubiera incluido. Todo el aporte medible del `bmi` era ese artefacto
(con la regresión logística, agregarlo no cambia nada: 0.8418 contra 0.8421). Por eso **el `bmi` se excluye del modelo**.

**Observaciones sobre otras variables:**
- Una sola fila tiene `gender = Other` y no presenta ACV; el género casi no discrimina (4.7 % en mujeres, 5.1 % en
  hombres).
- `smoking_status = Unknown` ocurre en 80 % de los menores de 18 años (y solo 44 % de los `Unknown` son menores): es
  una categoría de "no registrado" y no equivale a "no fuma".
- `ever_married` y `work_type = children` son sustitutos de la edad (correlación de 0.68 entre edad y estado civil).
- La edad tiene valores fraccionarios para los lactantes (mínimo 0.08 años).

## 3. Exploración de los datos (0.5 pts)

- `figuras/tasa_por_edad.png`: **la edad domina**. La tasa de ACV es 0.2 % en menores de 18 años, 0.5 % entre
  18 y 40, 4.1 % entre 40 y 60, 8.2 % entre 60 y 70, 17.7 % entre 70 y 80 y 19.8 % por encima de 80. La mediana de
  edad es 71 años entre quienes tuvieron un ACV y 43 entre quienes no.
- `figuras/numericas_por_clase.png`: la glucosa (mediana 105.2 contra 91.5 mg/dL) y el IMC (29.7 contra 28.0)
  son algo mayores en quienes tuvieron un ACV, con diferencias pequeñas frente a la edad.
- `figuras/tasa_por_categoria.png`: la hipertensión (13.3 % contra 4.0 %) y la enfermedad cardíaca (17.0 % contra
  4.2 %) triplican o cuadruplican la tasa. Las personas casadas tienen más ACV (6.6 % contra 1.7 %), pero eso
  refleja la edad.
- Valores extremos: 13 pacientes con IMC mayor a 60 y 25 con glucosa mayor a 250 mg/dL. Son valores clínicamente
  posibles y se conservaron.

## 4. Modelo (2 pts)

**Partición.** 80 % para entrenamiento (4 088 pacientes, 199 con ACV) y 20 % para prueba (1 022, 50 con ACV),
estratificada por el ACV, con `random_state = 42`. La exploración usa todos los pacientes (es descriptiva); la
elección de variables y del modelo y los umbrales se hicieron solo con el entrenamiento.

**Qué variables hacen falta.** Se compararon conjuntos de variables con regresión logística y validación cruzada
(5 particiones × 4 repeticiones, solo entrenamiento):

| Variables                                                  | AUC ROC (validación cruzada) |
|------------------------------------------------------------|------------------------------|
| 10 (todas)                                                 | 0.8374 ± 0.0193              |
| 9 (sin género)                                             | 0.8384 ± 0.0187              |
| 5 (las 4 elegidas + `bmi` imputado)                        | 0.8418 ± 0.0200              |
| **4 (edad, hipertensión, enfermedad cardíaca, glucosa)**   | **0.8421 ± 0.0202**          |
| 1 (solo la edad)                                           | 0.8341 ± 0.0217              |
| 4 + indicador de `bmi` faltante (artefacto, no se usa)     | 0.8550 ± 0.0186              |

**La edad sola ya alcanza 0.834.** Las otras tres variables aportan unas 0.008 unidades de AUC, menos que la
desviación entre pliegues (0.02), y cuatro variables rinden igual o mejor que diez. Se adoptan las **4 variables
clínicas**: el formulario es corto, se evitan datos sensibles innecesarios (género), variables que son solo
sustitutos de la edad (estado civil, tipo de trabajo) y el `bmi` por lo explicado en la sección 2.

**Candidatos.** Cada uno es un `Pipeline` (escalado de las numéricas y codificación *one-hot* de las categóricas):

| Candidato              | Justificación                                                                   |
|------------------------|---------------------------------------------------------------------------------|
| Tasa base (línea base) | Predice siempre la proporción de ACV: no discrimina (AUC = 0.5)                  |
| Regresión logística    | Modelo lineal interpretable y habitual en riesgo clínico                         |
| Random Forest          | 200 árboles poco profundos (`min_samples_leaf = 20`, profundidad 6), para no sobreajustar con 199 positivos |
| Gradient Boosting      | Ensamble secuencial de árboles superficiales (profundidad 3)                     |

**Sin pesos de clase.** No se ponderaron las clases porque la aplicación muestra la probabilidad de ACV y esta debe
estar calibrada [VanCalster2019]. El desbalance se trata con los umbrales.

**Selección.** Solo con el conjunto de entrenamiento, mediante validación cruzada estratificada repetida (5 × 4)
con el AUC ROC [Fawcett2006]. El modelo elegido es la **regresión logística**. Los hiperparámetros de los modelos
de árboles se fijaron a priori, sin búsqueda ni ajuste con la prueba. Con la edad estandarizada, el coeficiente de
la edad (+1.54) es entre 6 y 10 veces mayor que el de la glucosa (+0.18), la hipertensión (±0.24) y la enfermedad
cardíaca (±0.14): la edad domina el modelo.

**Dos umbrales**, calculados con predicciones fuera de muestra del entrenamiento:
- **Umbral F1 = 0.11**: maximiza el F1 de la clase positiva. Define la clase "Yes" (riesgo alto).
- **Umbral de sensibilidad = 0.045**: el mayor umbral con el que se detecta al menos el 80 % de los casos. Por
  debajo de él el riesgo es bajo; entre 0.045 y 0.11, moderado. Es el criterio habitual de un tamizaje, en el que
  importa no dejar casos sin detectar.

## 5. Evaluación (1 pt)

**Validación cruzada en entrenamiento** (5 × 4):

| Modelo                 | AUC ROC (media ± desv.) | AUC PR |
|------------------------|--------------------------|--------|
| Tasa base (línea base) | 0.500 ± 0.000            | 0.049  |
| **Regresión logística**| **0.8421 ± 0.0202**      | 0.189  |
| Random Forest          | 0.8347 ± 0.0194          | 0.173  |
| Gradient Boosting      | 0.8295 ± 0.0204          | 0.169  |

Las diferencias entre los tres modelos (0.007 a 0.013) son menores que la desviación entre pliegues (0.02): no son
concluyentes. Se elige la regresión logística por tener el valor más alto y ser la más simple.

**Conjunto de prueba (1 022 pacientes, 50 con ACV).** Con tan pocos positivos las métricas son muy ruidosas, por lo que
se acompañan de **intervalos de confianza del 95 % por bootstrap** (1 000 remuestreos) [Efron1993]:

| Métrica                          | Valor  | IC 95 %          | Referencia            |
|----------------------------------|--------|------------------|-----------------------|
| AUC ROC                          | 0.840  | 0.780 – 0.895    | 0.500                 |
| AUC PR                           | 0.262  | 0.174 – 0.370    | 0.049 (azar)          |
| Brier (menor es mejor)           | 0.041  | —                | 0.047 (tasa base)     |

**Efecto del umbral** (prueba):

| Umbral                     | Exactitud | Precisión | Recall (IC 95 %)          | F1    | Matriz [[TN, FP], [FN, TP]] |
|----------------------------|-----------|-----------|---------------------------|-------|------------------------------|
| 0.50                       | 0.951     | 0.000     | **0.000**                 | 0.000 | [[972, 0], [50, 0]]          |
| **0.11 (F1, riesgo alto)** | 0.861     | 0.205     | 0.640 (0.500 – 0.769)     | 0.311 | [[848, 124], [18, 32]]       |
| 0.045 (sensibilidad)       | 0.725     | 0.131     | **0.820** (0.708 – 0.926) | 0.226 | [[700, 272], [9, 41]]        |

Con el umbral convencional de 0.5 la **exactitud es 95.1 %, igual a la de predecir siempre "no"**, y el modelo no
detecta ningún caso: es el ejemplo de por qué la exactitud no sirve aquí. Con el umbral de 0.11 se detectan 32 de 50
casos y la precisión (20.5 %) es 4.2 veces la tasa base (4.9 %). Con el umbral de sensibilidad se detectan 41 de los
50 casos, a costa de señalar como de riesgo moderado o alto a 313 pacientes, de los cuales 272 no tuvieron un ACV.

**Niveles de riesgo en la prueba:**

| Nivel                        | Pacientes | Con ACV | Tasa real |
|------------------------------|-----------|---------|-----------|
| Bajo (p < 0.045)             | 709       | 9       | 1.3 %     |
| Moderado (0.045 ≤ p < 0.11)  | 157       | 9       | 5.7 %     |
| Alto (p ≥ 0.11)              | 156       | 32      | 20.5 %    |

**Los umbrales y el recall son sensibles al azar.** Con la misma partición, al cambiar solo la semilla de los
pliegues que se usan para calcular los umbrales, el umbral F1 varió entre 0.10 y 0.14 (la curva de F1 es una
meseta: elegir 0.11 es casi arbitrario dentro de ese rango), y el recall de prueba con él, entre 0.58 y 0.72 (el
publicado, 0.64, está en el medio). El umbral de sensibilidad fue más estable (0.045 a 0.05) y su recall de prueba
estuvo entre 0.80 y 0.82. Con otra partición de los datos, las diferencias serían mayores. Es razonable esperar que
el umbral de sensibilidad detecte entre 7 y 8 de cada 10 casos, no más.

**Calibración** (`figuras/evaluacion.png`): las probabilidades predichas coinciden en general con las frecuencias
reales; por ejemplo, donde el modelo predice en promedio 16.9 % la tasa real es 18.5 %. En el tramo de 4.9 % predicho la
tasa real fue 2.0 %, aunque con pocos casos por tramo [VanCalster2019].

## 6. Conclusión (0.5 pts)

Con cuatro datos clínicos es posible ordenar a los pacientes por riesgo de ACV con un AUC de 0.84 (IC 95 %
0.78–0.90), y el riesgo estimado es una probabilidad calibrada y no un simple sí o no. La **edad explica casi todo
el poder predictivo** (AUC 0.834 por sí sola): la hipertensión, la enfermedad cardíaca y la glucosa aportan una
mejora pequeña. Un nivel de riesgo bajo, moderado o alto es más útil en un tamizaje que una única clasificación: el
umbral de sensibilidad detecta alrededor de 7 a 8 de cada 10 casos, y el de F1 prioriza la precisión.

Los hallazgos metodológicos principales fueron (1) que la exactitud engaña (95.1 % sin detectar a nadie), (2) que el
`bmi` faltante es un artefacto informativo que **se filtraba al modelo a través de la imputación**, sin que nadie lo
incluyera, y se descubrió al verificar el modelo de forma independiente, y (3) que con solo 50 positivos en la
prueba, los intervalos de confianza son amplios y los umbrales dependen del azar.

**Limitaciones.** (1) **No es un diagnóstico.** (2) El **origen del conjunto es confidencial**: se desconoce la
población, el tipo de ACV (isquémico o hemorrágico) y la fecha, y no hay validación externa; un modelo de predicción
debería reportarse según TRIPOD [Collins2015]. (3) El conjunto no tiene **horizonte temporal**: no se sabe en qué
plazo ocurre el ACV, por lo que "probabilidad de sufrir un ACV" no indica cuándo. (4) Son 249 casos positivos en
total (50 en la prueba): el AUC de prueba tiene un intervalo de ±0.06 y el recall, de ±0.1 a ±0.15. (5) Las
asociaciones no son causales (la edad confunde a casi todas las demás variables). (6) El modelo extrapola en
combinaciones poco frecuentes (por ejemplo, solo hay un menor de edad con hipertensión en los datos). (7) El umbral
de sensibilidad (80 %) es una elección razonable, no un criterio clínico validado. (8) El `bmi` y otros factores de
riesgo conocidos (tabaquismo, diabetes diagnosticada) no aportan en estos datos, lo que no significa que carezcan
de relevancia clínica.

## Referencias (en `docs_latex/referencias.bib`)

- **[Soriano2021]** fedesoriano. (2021). *Stroke Prediction Dataset*. Kaggle. https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset (origen declarado como confidencial; uso solo educativo).
- **[Boehme2017]** Boehme, A. K., Esenwa, C., & Elkind, M. S. V. (2017). Stroke risk factors, genetics, and prevention. *Circulation Research*, 120(3), 472–495. https://doi.org/10.1161/CIRCRESAHA.116.308398
- **[Saito2015]** Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLoS ONE*, 10(3), e0118432. https://doi.org/10.1371/journal.pone.0118432
- **[Fawcett2006]** Fawcett, T. (2006). An introduction to ROC analysis. *Pattern Recognition Letters*, 27(8), 861–874. https://doi.org/10.1016/j.patrec.2005.10.010
- **[VanCalster2019]** Van Calster, B., McLernon, D. J., van Smeden, M., Wynants, L., & Steyerberg, E. W. (2019). Calibration: the Achilles heel of predictive analytics. *BMC Medicine*, 17, 230. https://doi.org/10.1186/s12916-019-1466-7
- **[Efron1993]** Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.
- **[Little2019]** Little, R. J. A., & Rubin, D. B. (2019). *Statistical Analysis with Missing Data* (3.ª ed.). Wiley.
- **[Collins2015]** Collins, G. S., Reitsma, J. B., Altman, D. G., & Moons, K. G. M. (2015). Transparent reporting of a multivariable prediction model for individual prognosis or diagnosis (TRIPOD): The TRIPOD statement. *Annals of Internal Medicine*, 162(1), 55–63. https://doi.org/10.7326/M14-0697

> Verificadas por búsqueda web el 2026-10-09. Pendiente de confirmar: rango final de páginas de Collins (2015), y el DOI de Little y Rubin (2019).
