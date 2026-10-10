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
  pacientes de la etapa 1 y 58 % de los de la etapa 4. Medimos cuánta información aportan: **solas dan un F1 macro de 0.228**
  (la línea base, 0.143) y **agregadas a las 15 variables no mejoran** el resultado (0.395 contra 0.398). No inflan la métrica,
  pero se excluyen por principio: un asistente no puede preguntar por lo que ocurrirá después.
- `Drug`: tratamiento aleatorizado, que no depende de la etapa y está vacío en los 106 pacientes fuera del ensayo.
- `ID`.
- La **edad** viene en días: se convierte a años (26.3 a 78.4; media 50.6).

**Valores faltantes: ¿artefacto o señal?** Antes de decidir cómo tratar a los pacientes incompletos se comprobó que los vacíos
**no dependen de la etapa**: la distribución de etapas en el bloque de 100 pacientes fuera del ensayo es casi idéntica a la del
resto (5 %, 25 %, 35 % y 35 %, contra 5 %, 21 %, 38 % y 35 %). A diferencia del `bmi` (modelo 05), aquí no hay un artefacto por
esta vía. Aun así, **los pacientes incompletos no ayudan** (sección 4) y se entrena solo con los **276 pacientes que tienen las
15 variables** (etapas 1, 2, 3 y 4: 12, 59, 111 y 94).

## 3. Exploración de los datos (0.5 pts)

- `figuras/etapas.png`: la etapa 1 es muy rara (12 pacientes completos).
- `figuras/laboratorio_por_etapa.png` y `figuras/signos_por_etapa.png`: los indicios de gravedad **crecen con la etapa**. Entre
  las etapas 1 y 4, la mediana de bilirrubina sube de 0.8 a 2.6 mg/dL, la de albúmina baja de 3.8 a 3.3 g/dL, la de plaquetas de
  271 a 216 y la de protrombina sube de 10.1 a 11.0 s. La hepatomegalia pasa de 0 % a 81 %, los angiomas en araña de 6 % a 46 %,
  la ascitis de 0 % a 19 % y el edema (con o sin diuréticos) de 5 % a 28 %. Las etapas intermedias (2 y 3) son difíciles de
  separar: sus medianas se parecen.
- `figuras/seguimiento_posterior.png`: el seguimiento posterior muestra la misma tendencia (ver sección 2), pero no se usa.
- Valores extremos: 5 pacientes con cobre mayor a 400, 5 con bilirrubina mayor a 20, 28 con fosfatasa alcalina mayor a 5 000. Son
  posibles en esta enfermedad y se conservaron.

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

Las 7 variables básicas rinden mucho peor, y sumar los pacientes incompletos no aporta nada: se usan las **15 variables** y
**solo los pacientes completos**, sin imputación posible de artefactos.

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
medio (0.625 contra 0.696) y más aciertos a una etapa (0.964 contra 0.875)** que el modelo, porque la etapa 3 está en el
centro: casi cualquier etapa real queda a una de distancia. La ponderación de clases lleva al modelo hacia los extremos, lo que
mejora el F1 macro y el kappa (0.34 contra 0), pero empeora esas dos métricas. Por eso no deben leerse aisladas.

**Estimaciones menos ruidosas (predicciones fuera de muestra del entrenamiento, 220 pacientes).** Exactitud 0.509, F1 macro 0.484,
kappa cuadrático 0.531, aciertos a una etapa 0.927 y error medio 0.564 (predecir siempre la etapa 3 daría 0.645 en esos
mismos pacientes). Recall por etapa: **1: 0.40; 2: 0.43; 3: 0.38; 4: 0.73**. La etapa 4 (cirrosis) es la que mejor se reconoce.

## 6. Conclusión (0.5 pts)

Estimar la etapa histológica solo con la consulta inicial es **difícil**: el F1 macro es de aproximadamente 0.46–0.48 y la
exactitud, de 0.43–0.51, con un kappa de 0.34 a 0.53. El modelo **distingue bien los extremos** (la etapa 4 con recall de
0.73 fuera de muestra) y se equivoca por una etapa o menos en cerca de 9 de cada 10 casos fuera de muestra, pero **confunde las
etapas intermedias 2 y 3**, que tienen perfiles de laboratorio parecidos. Los signos clínicos (hepatomegalia, angiomas en araña,
ascitis, edema) y la bilirrubina, la albúmina, las plaquetas y la protrombina son las señales más claras.

Los hallazgos metodológicos principales fueron (1) comprobar antes de afirmar: las variables de seguimiento posterior
(`N_Days`, `Status`) se excluyeron por principio, pero se midió que **no inflaban** el resultado, de modo que no se afirma
una fuga que no se observó; (2) que los pacientes incompletos no aportan y los vacíos no dependen de la etapa; y (3) que con una
etapa ordinal conviene mirar varias métricas, porque predecir siempre la categoría central gana en error medio.

**Limitaciones.** (1) **No es un diagnóstico.** (2) Son solo 276 pacientes completos y la etapa 1 tiene 12: en la prueba hay 2,
por lo que su recall perfecto (1.0) no significa nada; los intervalos de confianza son muy amplios. (3) Son datos de un único
ensayo de 1974–1984, con 90 % de mujeres, sin validación externa y con un tratamiento que ya no es el estándar. (4) Los puntajes
no son probabilidades calibradas (se ponderaron las clases). (5) Las unidades del formulario (mg/dL, g/dL, µg/día, U/L, U/mL,
miles por mL, segundos) provienen de la documentación del conjunto de datos y deben confirmarse. (6) La biopsia tiene variabilidad
entre observadores, por lo que la etapa de referencia tiene ruido. (7) Con una prueba de 56 pacientes, los resultados cambiarían
con otra partición.

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
