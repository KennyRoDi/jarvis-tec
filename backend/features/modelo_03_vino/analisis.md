# Modelo 03 · Clasificación de la calidad del vino

<!-- Notas para LaTeX: cada sección corresponde a un criterio de la rúbrica. Las figuras están en
     figuras/*.png y las métricas exactas en metricas.json (se regeneran con train.py). Las referencias
     [Clave] están en docs_latex/referencias.bib. -->

## 1. Análisis del problema (0.5 pts)

La calidad de un vino se determina por una cata sensorial, un proceso lento, costoso y subjetivo. Se plantea
estimarla a partir de sus propiedades fisicoquímicas, que se miden en laboratorio de forma objetiva
[Cortez2009]. Se trata de un problema de **clasificación multiclase supervisada**: la puntuación sensorial
(`quality`, de 3 a 9) se agrupa en tres clases ordenadas, **baja** (puntuación ≤ 5), **media** (= 6) y **alta**
(≥ 7). La agrupación es necesaria porque los extremos casi no tienen ejemplos (30 vinos con 3 puntos y 5 con 9).

El asistente JarvisTEC utiliza el modelo para responder consultas como "¿qué calidad tiene este vino?" a partir
de las 12 variables que el usuario ingresa en el formulario.

## 2. Entendimiento de los datos (0.5 pts)

El conjunto contiene muestras de vinho verde portugués, tinto y blanco [Cortez2009], y está disponible en Kaggle:
6 497 filas y 13 columnas (1 599 tintos y 4 898 blancos). Las variables son el tipo de vino, 11 propiedades
fisicoquímicas (acidez fija y volátil, ácido cítrico, azúcar residual, cloruros, dióxido de azufre libre y total,
densidad, pH, sulfatos y alcohol) y la puntuación.

**Filas duplicadas.** Hay **1 168 filas idénticas (18 %)**. Si se dividen los datos sin eliminarlas, la misma
muestra puede quedar en entrenamiento y en prueba, y la métrica mide la memoria del modelo y no su capacidad de
generalizar. Se comprobó con un experimento: al dividir **con** duplicados (prueba de 1 300 filas), 356 filas de prueba
(27 %) tienen una copia exacta en el entrenamiento y la exactitud sube a **0.695**, frente a **0.594** al
eliminarlos. Las dos particiones no son idénticas, de modo que no es una comparación pareada, pero el efecto
(unos 10 puntos) es claro. Por eso se descartan antes de dividir. Quedan **5 329 filas** (3 970 blancos y 1 359 tintos).

**Valores nulos.** Hay 38 valores nulos repartidos en 7 columnas y en 34 filas (0.6 % de las filas). Se imputan
con la mediana **dentro del `Pipeline`**, de modo que en cada pliegue de validación la mediana se calcule solo con
sus datos de entrenamiento. La API exige los 12 datos (no acepta valores nulos), por lo que el imputador protege el
entrenamiento y cualquier uso directo del `Pipeline`, no las consultas de la aplicación.

**Clases resultantes:** baja 37.4 % (1 991), media 43.7 % (2 327) y alta 19.0 % (1 011): moderadamente
desbalanceadas, por lo que se usa F1 macro como métrica principal.

## 3. Exploración de los datos (0.5 pts)

- `figuras/clases.png`: los vinos tintos concentran más calidad baja (47 %) que los blancos (34 %); los blancos
  tienen más calidad alta (21 % contra 14 %).
- `figuras/variables_por_clase.png`: el **alcohol** es la variable que más separa las clases (mediana de 9.7 % en
  calidad baja, 10.5 % en media y 11.6 % en alta). La acidez volátil es mayor en la calidad baja (mediana 0.34
  contra 0.27–0.28) y la densidad desciende al subir la calidad (0.996, 0.994 y 0.992).
- `figuras/correlacion.png`: las correlaciones más altas con la puntuación son alcohol (0.47), densidad (−0.33),
  acidez volátil (−0.26) y cloruros (−0.20). El alcohol y la densidad están fuertemente relacionados entre sí
  (−0.67), por lo que hay información redundante.
- **Valores extremos.** El azúcar residual llega a 65.8 g/dm³ (2 vinos sobre 30) y el dióxido de azufre total supera
  300 mg/dm³ en 6 vinos. Son mediciones posibles, por lo que se conservaron.

## 4. Modelo (2 pts)

Se dividieron los datos de forma **estratificada** (conserva la proporción de clases): 80 % para entrenamiento
(4 263 filas) y 20 % para prueba (1 066 filas), con `random_state = 42`. Cada candidato es un `Pipeline` con
imputación de nulos, escalado (cuando corresponde) y codificación del tipo de vino:

| Candidato                    | Justificación                                                                |
|------------------------------|------------------------------------------------------------------------------|
| Clase mayoritaria (base)     | Referencia mínima: siempre predice "media"                                    |
| Regresión logística          | Modelo lineal interpretable [Hastie2009], con pesos de clase balanceados      |
| SVM con núcleo RBF           | Frontera no lineal [Cortes1995]; superó a la regresión múltiple y a las redes neuronales en el estudio original [Cortez2009]. Se usó C = 3 fijo, **sin optimizar**, y se compara pero **no es elegible** (no da probabilidades) |
| Random Forest                | Ensamble de árboles que captura interacciones sin suponer su forma [Breiman2001] |
| Gradient Boosting            | Ensamble secuencial de árboles [Friedman2001], versión de histogramas de scikit-learn |

La selección se realizó **solo con el conjunto de entrenamiento**, mediante validación cruzada estratificada
repetida (5 particiones × 2 repeticiones) con F1 macro como criterio [Sokolova2009]. El conjunto de prueba se
utilizó una única vez. El modelo elegido es **Random Forest** (100 árboles, `min_samples_leaf = 5`, profundidad
máxima 16, pesos de clase balanceados).

**Tamaño del modelo.** Un bosque más grande (150 árboles, `min_samples_leaf = 3`, profundidad máxima 18) alcanzaba un F1 de validación
cruzada de 0.599, pero ocupaba 5.4 MB; la configuración adoptada rinde 0.595 y ocupa 2.9 MB. La diferencia de
0.004 es mucho menor que la variación entre pliegues (0.012), por lo que se prefirió el modelo compacto, que se
versiona en git.

## 5. Evaluación (1 pt)

**Validación cruzada en entrenamiento** (media ± desviación del F1 macro):

| Modelo                  | F1 macro        | Exactitud |
|-------------------------|-----------------|-----------|
| Clase mayoritaria       | 0.203 ± 0.000   | 0.437     |
| Regresión logística     | 0.554 ± 0.012   | 0.559     |
| SVM RBF                 | 0.567 ± 0.019   | 0.572     |
| **Random Forest**       | **0.595 ± 0.012** | 0.602   |
| Gradient Boosting       | 0.586 ± 0.016   | 0.598     |

Random Forest y Gradient Boosting no se distinguen con claridad (diferencia de 0.009 con desviaciones de 0.012 a
0.016), mientras que la regresión logística y la SVM quedan por detrás. La SVM no se optimizó (C = 3 fijo), así que se compara solo como referencia. Además se entrena sin estimar probabilidades (el parámetro
`probability` está deprecado en scikit-learn 1.9), por lo que **no es elegible**: la API devuelve las probabilidades de cada clase y la
selección solo considera candidatos que las calculan (`elegibles()` en `train.py`). Con ella habría que calibrar su salida, por ejemplo
con `CalibratedClassifierCV`, lo que no se hizo.

**Conjunto de prueba (1 066 filas):**

| Métrica         | Random Forest | Clase mayoritaria |
|-----------------|---------------|-------------------|
| Exactitud       | 0.594         | 0.437             |
| F1 macro        | 0.591         | 0.203             |
| Precisión macro | 0.583         | 0.146             |
| Exactitud por clase (recall) baja / media / alta | 0.701 / 0.485 / 0.634 | — |

Resultados por clase (prueba):

| Clase | Precisión | Recall | F1    | Muestras |
|-------|-----------|--------|-------|----------|
| baja  | 0.660     | 0.701  | 0.680 | 398      |
| media | 0.572     | 0.485  | 0.525 | 466      |
| alta  | 0.516     | 0.634  | 0.569 | 202      |

Matriz de confusión (filas: real; columnas: predicho; `figuras/matriz_confusion.png`):

|            | baja | media | alta |
|------------|------|-------|------|
| **baja**   | 279  | 105   | 14   |
| **media**  | 134  | 226   | 106  |
| **alta**   | 10   | 64    | 128  |

El F1 de prueba (0.591) es consistente con el de validación cruzada (0.595): no hay indicio de sobreajuste a la
selección. Los errores son **mayoritariamente entre clases vecinas** (409 de 433 errores, 94 %): el modelo casi
nunca confunde un vino bajo con uno alto (24 casos). La clase "media" es la más difícil (F1 0.525), como es de
esperar al ser la intermedia de una escala ordinal.

## 6. Conclusión (0.5 pts)

Con las propiedades fisicoquímicas es posible clasificar la calidad del vino con una exactitud de 0.59 y un F1
macro de 0.59, frente a 0.44 y 0.20 de la clase mayoritaria. El resultado es útil como orientación pero no
reemplaza a la cata: los errores grandes son raros, pero cerca de 4 de cada 10 vinos reciben una clase vecina
equivocada. El alcohol, la acidez volátil y la densidad son las variables más informativas.

El hallazgo metodológico principal es la **fuga por filas duplicadas**: sin eliminarlas la exactitud parece
unos 10 puntos mejor. La API incluye en la respuesta las probabilidades de cada clase y avisa cuando ninguna supera el
50 %, para no presentar como firme una clasificación dudosa. También advierte cuando alguna medida queda fuera del rango
de los vinos de entrenamiento (en las esquinas de lo permitido el modelo puede dar clasificaciones confiadas sin
sentido) y rechaza combinaciones imposibles, como un dióxido de azufre libre mayor que el total.

**Limitaciones.** (1) La puntuación sensorial es subjetiva: es un techo natural de lo que se puede predecir.
(2) Solo se incluyen vinos verdes portugueses; el modelo no es válido para otras regiones ni variedades.
(3) La SVM no se optimizó ni es elegible (ver la sección 5), y los hiperparámetros del bosque se fijaron sin búsqueda exhaustiva. (4) El conjunto
de prueba es una sola partición de 1 066 filas.

## Referencias (en `docs_latex/referencias.bib`)

- **[Cortez2009]** Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). Modeling wine preferences by data mining from physicochemical properties. *Decision Support Systems*, 47(4), 547–553. https://doi.org/10.1016/j.dss.2009.05.016
- **[Cortes1995]** Cortes, C., & Vapnik, V. (1995). Support-vector networks. *Machine Learning*, 20(3), 273–297. https://doi.org/10.1007/BF00994018
- **[Hastie2009]** Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2.ª ed.). Springer. https://doi.org/10.1007/978-0-387-84858-7
- **[Sokolova2009]** Sokolova, M., & Lapalme, G. (2009). A systematic analysis of performance measures for classification tasks. *Information Processing & Management*, 45(4), 427–437. https://doi.org/10.1016/j.ipm.2009.03.002
- **[Breiman2001]** Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32. https://doi.org/10.1023/A:1010933404324
- **[Friedman2001]** Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *The Annals of Statistics*, 29(5), 1189–1232. https://doi.org/10.1214/aos/1013203451

> Verificadas por búsqueda web el 2026-10-09.
