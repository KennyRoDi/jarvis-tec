# Modelo 07 · Clasificación de la etapa de cirrosis

<!-- Notas para LaTeX: cada sección corresponde a un criterio de la rúbrica. Las figuras están en
     figuras/*.png y las métricas exactas en metricas.json (se regeneran con train.py). Las referencias
     [Clave] están en docs_latex/referencias.bib. -->

> **Aviso.** Es un ejercicio educativo con pocos pacientes. **No es una herramienta de diagnóstico**: la etapa histológica
> se determina por biopsia.

## 1. Análisis del problema (0.5 pts)

La cirrosis biliar primaria (hoy llamada colangitis biliar primaria) es una enfermedad hepática crónica cuya gravedad se
clasifica en cuatro **etapas histológicas**, de la 1 (lesión temprana) a la 4 (cirrosis), mediante biopsia [Ludwig1978;
Lindor2019]. La biopsia es invasiva, por lo que se plantea **estimar la etapa a partir de signos clínicos y análisis de
laboratorio** de la consulta inicial. Es un problema de **clasificación multiclase ordinal supervisada**: equivocarse por
tres etapas es peor que equivocarse por una. Por eso, además de las métricas habituales, se usan el error medio en etapas, la
proporción de aciertos con error de a lo sumo una etapa y el kappa cuadrático ponderado [Cohen1968].

El asistente JarvisTEC responde con la etapa estimada y los puntajes de las cuatro etapas a partir de los 15 datos que el
usuario ingresa en el formulario.

## 2. Entendimiento de los datos (0.5 pts)

El conjunto proviene del ensayo clínico de la Clínica Mayo sobre cirrosis biliar primaria (1974–1984) [Dickson1989;
Therneau2000pbc]: 418 pacientes y 20 columnas. **312 participaron en el ensayo aleatorizado de D-penicilamina** y tienen datos
casi completos; **106 no participaron** y solo tienen las mediciones básicas, por lo que carecen de ascitis, hepatomegalia,
angiomas en araña, cobre, fosfatasa alcalina y SGOT. Seis pacientes no tienen etapa y se descartan: quedan **412**, con las
etapas 1, 2, 3 y 4 en 21, 92, 155 y 144 pacientes.

**Variables excluidas (por qué):**
- `N_Days` y `Status`: son el **seguimiento posterior** (días hasta el fallecimiento o la censura y el estado final); no existen en
  la consulta inicial. La mediana de `N_Days` baja de 2 644 días en la etapa 1 a 1 207 en la 4, y fallecieron 10 % de los
  pacientes de la etapa 1 y 58 % de los de la etapa 4 (calculado sobre los 412 pacientes con etapa). Se midió cuánta información
  aportan: **solas dan un F1 macro de 0.228 con la regresión logística** (línea base 0.143) y de 0.29 a 0.33 con el Random Forest
  (un Random Forest ponderado con una columna de ruido ya da cerca de 0.23), y **agregadas a las 15 variables no mejoran** el
  resultado (regresión logística: 0.395 contra 0.398; el Random Forest, comprobado de forma independiente con 3 semillas,
  tampoco). No inflan la métrica, pero se excluyen por principio: un asistente no puede preguntar por lo que ocurrirá después.
- `Drug`: tratamiento aleatorizado, que no depende de la etapa y está vacío en los 106 pacientes fuera del ensayo.
- `ID` (orden de reclutamiento): correlaciona con la protrombina (deriva del laboratorio en el tiempo) pero no con la etapa.
- La **edad** viene en días: se convierte a años (26.3 a 78.4; media 50.6).

**Valores faltantes: ¿artefacto o señal?** Antes de decidir cómo tratar a los pacientes incompletos se comprobó que los vacíos
**no dependen de la etapa**: la distribución de etapas en el bloque de 100 pacientes fuera del ensayo es casi idéntica a la del
resto (5 %, 25 %, 35 % y 35 %, contra 5 %, 21 %, 38 % y 35 %). A diferencia del `bmi` (modelo 05), aquí no hay un artefacto por
esta vía. Aun así, **los pacientes incompletos no ayudan** (sección 4) y se entrena solo con los **276 pacientes que tienen las
15 variables** (etapas 1, 2, 3 y 4: 12, 59, 111 y 94).

## 3. Exploración de los datos (0.5 pts)

Las cifras de esta sección y las figuras corresponden a los **276 pacientes completos** (los que se usan para entrenar), salvo
que se indique otra cosa.

- `figuras/etapas.png`: la etapa 1 es muy rara (12 pacientes completos).
- `figuras/laboratorio_por_etapa.png` y `figuras/signos_por_etapa.png`: los indicios de gravedad **crecen con la etapa**. Entre las
  etapas 1 y 4, la mediana de bilirrubina sube de 0.75 a 2.95 mg/dL, la de albúmina baja de 3.74 a 3.36 g/dL, la de plaquetas de 268
  a 231 y la de protrombina sube de 10.4 a 11.0 s. La hepatomegalia pasa de 0 % a 82 %, los angiomas en araña de 0 % a 47 %, la
  ascitis de 0 % a 20 % y el edema (con o sin diuréticos) de 0 % a 28 %. Las etapas intermedias (2 y 3) son difíciles de separar:
  sus medianas se parecen.
- **Sesgo de selección.** Entre los completos, la ascitis solo aparece en la etapa 4 (19 de 19); en los 312 pacientes del ensayo hay 3
  casos con ascitis fuera de la etapa 4 (2 en la etapa 2 y 1 en la 3) que quedaron excluidos por faltarles el colesterol o los
  triglicéridos.
- `figuras/seguimiento_posterior.png`: el seguimiento posterior (sobre los 412 pacientes) muestra la misma tendencia, pero no se usa.
- Valores extremos (en los 412): 5 pacientes con cobre mayor a 400, 5 con bilirrubina mayor a 20 y 28 con fosfatasa alcalina mayor a
  5 000. Son posibles en esta enfermedad y se conservaron.

## 4. Modelo (2 pts)

**Partición.** 80 % para entrenamiento (220 pacientes: 10, 47, 88 y 75 por etapa) y 20 % para prueba (56: **solo 2 en la etapa 1**,
12, 23 y 19), estratificada, con `random_state = 42`.

**Qué variables y qué pacientes.** Dos experimentos con regresión logística (F1 macro de validación cruzada 5 × 4, solo
entrenamiento):

| Variables                                                           | F1 macro        |
|---------------------------------------------------------------------|-----------------|
| **15 (todas las de la consulta inicial)**                           | **0.398 ± 0.063** |
| 13 (sin colesterol ni triglicéridos)                                | 0.376 ± 0.054   |
| 7 básicas (las disponibles en todos los pacientes)                  | 0.301 ± 0.070   |

| ¿Se entrena también con los 136 pacientes incompletos (imputados)?   | Solo completos  | Con incompletos |
|---------------------------------------------------------------------|-----------------|-----------------|
| Regresión logística                                                 | 0.398 ± 0.063   | 0.400 ± 0.064   |
| Random Forest                                                       | 0.460 ± 0.083   | 0.445 ± 0.091   |

Las 7 variables básicas rinden mucho peor, y sumar los pacientes incompletos no aporta nada: con 8 particiones distintas la
regresión logística empeora al sumarlos en 7 de 8 (−0.03 en promedio) y el Random Forest no cambia (+0.003). La conclusión "no
mejoran" es firme; el signo exacto del cambio, no. Se usan las **15 variables** y **solo los pacientes completos**, sin que
ninguna imputación entre al ajuste.

**¿Qué variables usa realmente el modelo?** Las medianas de la sección 3 describen los datos, no al modelo. Se midió cuánto
cambia el F1 macro de la regresión logística (validación cruzada 5 × 4, solo entrenamiento) **al quitar cada variable**, respecto
de las 15 (0.398):

| Al quitar…           | Cambio del F1 macro |
|----------------------|---------------------|
| edad                 | −0.026              |
| SGOT                 | −0.025              |
| hepatomegalia        | −0.021              |
| triglicéridos        | −0.016              |
| colesterol, albúmina, ascitis | −0.005 a −0.001 |
| edema, cobre, angiomas, sexo, bilirrubina, plaquetas, fosfatasa alcalina, protrombina | +0.002 a +0.013 |

**Ninguna variable aporta por sí sola más de 0.026**, menos que la desviación entre pliegues (≈ 0.06): el modelo reparte su
información entre variables redundantes. Quitar la bilirrubina, el edema, las plaquetas o la protrombina no lo empeora. Una
comprobación independiente con el Random Forest coincidió en que la hepatomegalia es la variable más consistentemente útil, y mostró
que el efecto de la bilirrubina y del edema sobre la etapa estimada es casi nulo (con la bilirrubina multiplicada por 3, la etapa
esperada sube solo en 61 % de los pacientes).

**Candidatos.** Cada uno es un `Pipeline` (imputación, escalado y codificación *one-hot*), con **clases ponderadas** porque la
etapa 1 tiene solo 10 pacientes en el entrenamiento; por eso los puntajes **no son probabilidades calibradas**:

| Candidato                      | Justificación                                                                |
|--------------------------------|------------------------------------------------------------------------------|
| Clase mayoritaria (línea base) | Siempre predice la etapa 3                                                    |
| Regresión logística            | Modelo lineal multinomial                                                     |
| Random Forest                  | 200 árboles poco profundos (`min_samples_leaf = 3`, profundidad 8) [Breiman2001] |
| Gradient Boosting              | Ensamble secuencial de árboles superficiales [Friedman2001]                   |

**Selección.** Solo con el entrenamiento, mediante validación cruzada estratificada repetida (5 × 4) con el F1 macro como
criterio [Sokolova2009]. El modelo elegido es el **Random Forest**.

## 5. Evaluación (1 pt)

**Validación cruzada en entrenamiento:**

| Modelo                  | F1 macro (media ± desv.) | Exactitud balanceada |
|-------------------------|---------------------------|----------------------|
| Clase mayoritaria       | 0.143 ± 0.003             | 0.250                |
| Regresión logística     | 0.398 ± 0.063             | 0.449                |
| **Random Forest**       | **0.460 ± 0.083**         | 0.467                |
| Gradient Boosting       | 0.424 ± 0.086             | 0.431                |

Las diferencias (0.04 a 0.06) son menores que las desviaciones entre pliegues (0.06 a 0.09): **no son concluyentes**.

**Conjunto de prueba (56 pacientes)** con intervalos de confianza del 95 % por bootstrap [Efron1993]:

| Métrica                              | Modelo  | IC 95 %         | Línea base |
|--------------------------------------|---------|-----------------|------------|
| Exactitud                            | 0.429   | —               | 0.411      |
| F1 macro                             | 0.465   | 0.251 – 0.599   | 0.146      |
| Kappa cuadrático ponderado           | 0.340   | 0.045 – 0.561   | 0.000      |
| Aciertos con error de a lo sumo 1 etapa | 0.875 | 0.786 – 0.946   | 0.964      |
| Error medio (en etapas)              | 0.696   | —               | 0.625      |

| Etapa | Precisión | Recall | Pacientes |
|-------|-----------|--------|-----------|
| 1     | 0.500     | 1.000  | 2         |
| 2     | 0.286     | 0.333  | 12        |
| 3     | 0.429     | 0.522  | 23        |
| 4     | 0.600     | 0.316  | 19        |

Matriz de confusión (filas: etapa real; columnas: etapa estimada; `figuras/matriz_confusion.png`):

|            | 1 | 2 | 3  | 4 |
|------------|---|---|----|---|
| **Etapa 1**| 2 | 0 | 0  | 0 |
| **Etapa 2**| 0 | 4 | 6  | 2 |
| **Etapa 3**| 2 | 7 | 12 | 2 |
| **Etapa 4**| 0 | 3 | 10 | 6 |

**Una advertencia sobre las métricas ordinales.** En la prueba, **la línea base (predecir siempre la etapa 3) tiene mejor error
medio (0.625 contra 0.696) y más aciertos a una etapa (0.964 contra 0.875)** que el modelo, porque la etapa 3 está en el centro:
casi cualquier etapa real queda a una de distancia. **Lo mismo ocurre fuera de muestra** en los aciertos a una etapa (línea base
0.955 contra 0.927 del modelo, y el modelo queda por debajo en las 8 particiones probadas). En el error medio fuera de muestra el
modelo sí gana (0.564 contra 0.645). La ponderación de clases **no es la causa principal**: un Random Forest sin ponderar tiene en la
misma prueba error medio 0.679 y aciertos a una etapa de 0.893 (también pierde contra la línea base); ponderar sube el F1 macro de
0.27 a 0.47 y el kappa de 0.10 a 0.34 a un costo de unos 0.02 en esas dos métricas. Ninguna métrica debe leerse aislada.

**Estimaciones menos ruidosas (predicciones fuera de muestra del entrenamiento, 220 pacientes).** Exactitud 0.509, F1 macro 0.484,
kappa cuadrático 0.531, aciertos a una etapa 0.927 y error medio 0.564. Recall por etapa: **1: 0.40; 2: 0.43; 3: 0.38; 4: 0.73**.

**El recall de la etapa 4: 0.32 en la prueba, 0.73 fuera de muestra.** La partición publicada (semilla 42) es especialmente
pesimista para la etapa 4: en la prueba 10 de sus 19 pacientes se estimaron como etapa 3 (recall 0.32). Con 8 particiones
distintas (semillas 0 a 7), el recall de la etapa 4 en la prueba va de 0.58 a 0.74 (media 0.68) y fuera de muestra de 0.65 a 0.73.
El de la etapa 1 (2 pacientes en la prueba; 10 en el entrenamiento) es puro ruido: fuera de muestra va de 0.2 a 0.7 según la semilla.

**Sensibilidad a la partición (8 particiones, semillas 0 a 7; comprobación independiente).** El Random Forest gana la validación
cruzada en las 8. En la prueba, su F1 macro va de 0.31 a 0.57 (media 0.45; el publicado, 0.465, es típico), el kappa de 0.33 a 0.62
(media 0.45; el publicado, 0.34, es pesimista) y el error medio de 0.48 a 0.66 (media 0.60; el publicado, 0.696, es peor que
los 8). Fuera de muestra, el F1 macro va de 0.41 a 0.53 y el kappa de 0.46 a 0.58.

## 6. Conclusión (0.5 pts)

Estimar la etapa histológica solo con la consulta inicial es **difícil**: el F1 macro es de aproximadamente 0.45–0.48 y la
exactitud, de 0.43–0.51, con un kappa de 0.34 a 0.53. El modelo **distingue mejor el extremo de la etapa 4** (recall de 0.73 fuera
de muestra y de 0.58 a 0.74 en otras particiones, aunque 0.32 en la prueba publicada) y se equivoca por una etapa o menos en cerca de
9 de cada 10 casos fuera de muestra (**la línea base "siempre etapa 3" lo logra en 95 %**, de modo que esa cifra por sí sola no
demuestra nada). **Confunde las etapas intermedias 2 y 3**, que tienen perfiles de laboratorio parecidos.

Los indicios de gravedad de la sección 3 (bilirrubina, albúmina, plaquetas, ascitis, edema) **describen** los datos, pero el
modelo casi no se apoya en ellos: ninguna variable aporta por sí sola más de 0.026 de F1, y las que más pesan son la edad, el SGOT, la
hepatomegalia y los triglicéridos.

Los hallazgos metodológicos principales fueron (1) comprobar antes de afirmar: `N_Days` y `Status` se excluyeron por principio, pero se
midió que **no inflaban** el resultado, de modo que no se afirma una fuga que no se observó; (2) que los pacientes incompletos no aportan
y los vacíos no dependen de la etapa; (3) que describir los datos (medianas por etapa) no es lo mismo que medir lo que el modelo usa
(quitar cada variable); y (4) que con una etapa ordinal conviene mirar varias métricas, porque predecir siempre la categoría central gana
en error medio y en aciertos a una etapa.

**Limitaciones.** (1) **No es un diagnóstico.** (2) Son solo 276 pacientes completos y la etapa 1 tiene 12: en la prueba hay 2, por lo
que su recall perfecto (1.0) no significa nada; los intervalos de confianza son muy amplios y el recall por etapa depende de la
partición. (3) Son datos de un único ensayo de 1974–1984, con cerca de 90 % de mujeres, sin validación externa y con un tratamiento que ya
no es el estándar. (4) Los puntajes no son probabilidades calibradas (se ponderaron las clases). (5) Las unidades del formulario (mg/dL,
g/dL, µg/día, U/L, 10³/µL, segundos) se infirieron de la documentación del conjunto y de los rangos de los datos, y deben confirmarse. (6) La
biopsia tiene variabilidad entre observadores, por lo que la etapa de referencia tiene ruido. (7) Sesgo de selección por entrenar solo con
completos (por ejemplo, 3 casos de ascitis fuera de la etapa 4 quedaron excluidos). (8) La API exige los 15 datos: no hay forma de
consultar con datos parciales.

## Referencias (en `docs_latex/referencias.bib`)

- **[Dickson1989]** Dickson, E. R., Grambsch, P. M., Fleming, T. R., Fisher, L. D., & Langworthy, A. (1989). Prognosis in primary biliary cirrhosis: Model for decision making. *Hepatology*, 10(1), 1–7.
- **[Therneau2000pbc]** Therneau, T. M., & Grambsch, P. M. Conjunto de datos `pbc` (ensayo de la Clínica Mayo, 1974–1984), paquete `survival` de R. https://www.rdocumentation.org/packages/survival/topics/pbc
- **[Ludwig1978]** Ludwig, J., Dickson, E. R., & McDonald, G. S. (1978). Staging of chronic nonsuppurative destructive cholangitis (syndrome of primary biliary cirrhosis). *Virchows Archiv A*, 379(2), 103–112. https://doi.org/10.1007/BF00432479
- **[Lindor2019]** Lindor, K. D., Bowlus, C. L., Boyer, J., Levy, C., & Mayo, M. (2019). Primary biliary cholangitis: 2018 practice guidance from the American Association for the Study of Liver Diseases. *Hepatology*.
- **[Cohen1968]** Cohen, J. (1968). Weighted kappa: Nominal scale agreement with provision for scaled disagreement or partial credit. *Psychological Bulletin*, 70(4), 213–220. https://doi.org/10.1037/h0026256
- **[Breiman2001]** Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32. https://doi.org/10.1023/A:1010933404324
- **[Friedman2001]** Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *The Annals of Statistics*, 29(5), 1189–1232. https://doi.org/10.1214/aos/1013203451
- **[Sokolova2009]** Sokolova, M., & Lapalme, G. (2009). A systematic analysis of performance measures for classification tasks. *Information Processing & Management*, 45(4), 427–437. https://doi.org/10.1016/j.ipm.2009.03.002
- **[Efron1993]** Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.

> Verificadas por búsqueda web el 2026-10-10. Pendiente de confirmar: volumen y páginas de Lindor et al. (2019).
