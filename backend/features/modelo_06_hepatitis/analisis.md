# Modelo 06 · Clasificación del estado hepático por hepatitis C

<!-- Notas para LaTeX: cada sección corresponde a un criterio de la rúbrica. Las figuras están en
     figuras/*.png y las métricas exactas en metricas.json (se regeneran con train.py). Las referencias
     [Clave] están en docs_latex/referencias.bib. -->

> **Aviso.** Es un ejercicio educativo con un conjunto de datos pequeño. **No es una herramienta de diagnóstico** ni
> reemplaza la valoración de un profesional de la salud.

## 1. Análisis del problema (0.5 pts)

La infección crónica por el virus de la hepatitis C puede evolucionar de una hepatitis a fibrosis y a cirrosis, y la
etapa se estima con análisis de sangre y biopsias o pruebas no invasivas [Manns2017]. Se plantea clasificar el estado
hepático de una persona en cuatro clases ordenadas, **donante** (sano), **hepatitis**, **fibrosis** y **cirrosis**, a
partir de la edad y diez análisis de laboratorio, cuya interpretación clínica es conocida [Giannini2005]. Es un
problema de **clasificación multiclase supervisada, muy desbalanceada**: 88 % son donantes y las tres clases de
enfermedad tienen solo 21 a 30 casos cada una.

El asistente JarvisTEC responde con la clase más probable y los puntajes de cada clase a partir de los valores que
el usuario ingresa en el formulario.

## 2. Entendimiento de los datos (0.5 pts)

El conjunto reúne los datos de laboratorio de donantes de sangre y de pacientes con hepatitis C [Lichtinghagen2013;
Hoffmann2018]: 615 personas y 13 columnas útiles (clase, edad, sexo y diez análisis: ALB, ALP, ALT, AST, BIL, CHE, CHOL,
CREA, GGT y PROT). El archivo incluye una columna de índice (`Unnamed: 0`) que se descarta. No hay filas totalmente duplicadas, pero 7 pares de donantes tienen los diez análisis idénticos con edad (y a veces sexo) distintos:
son casi duplicados cuyos datos demográficos no son fiables; 3 de esos pares cruzan entrenamiento y prueba (todos son donantes, con efecto práctico nulo).

**Clase ambigua.** Siete personas tienen la categoría "suspect Blood Donor". No se parecen a un donante sano (su
albúmina mediana es de 21.6 g/L y sus proteínas totales de 47.8 g/L) ni corresponden a una etapa de la enfermedad, por
lo que se **excluyen**. Quedan **608 personas**: 533 donantes, 24 con hepatitis, 21 con fibrosis y 30 con cirrosis.

**Valores faltantes y el ALP: ¿artefacto o señal?** Hay 18 valores vacíos de ALP, **todos en pacientes** (24 % de los 75
pacientes; 43 % de los de fibrosis) y **ninguno en donantes**. Eso hace sospechar un artefacto de la recolección, como el del
`bmi` en el modelo de ACV, de modo que se comprobó. Si el aporte del ALP al Random Forest viniera de los vacíos, **rellenar los
vacíos con valores observados al azar** lo haría desaparecer, y un **indicador de "ALP faltante"** lo reproduciría. Ocurre lo
contrario (F1 macro de validación cruzada, Random Forest):

| Variante                                                              | F1 macro        |
|-----------------------------------------------------------------------|-----------------|
| Con ALP (la elegida)                                                  | 0.635 ± 0.111   |
| **Sin ALP**                                                           | 0.580 ± 0.110   |
| Con los vacíos de ALP rellenados al azar con valores observados       | 0.623 ± 0.128   |
| Solo con el indicador de ALP faltante (sin ALP)                       | 0.598 ± 0.122   |
| Solo en las filas con ALP observada, con ALP                          | 0.673 ± 0.106   |
| Solo en las filas con ALP observada, sin ALP                          | 0.575 ± 0.096   |

La mejora se mantiene con el relleno aleatorio (0.623), el indicador solo apenas aporta (0.598) y en las filas donde el ALP
sí se observó la mejora es mayor (+0.098): **el ALP aporta señal real y no es un artefacto de sus vacíos**, a diferencia del
`bmi` del modelo de ACV, donde el relleno aleatorio sí hacía desaparecer el aporte. Se **incluye el ALP** y se imputa con la
mediana dentro del `Pipeline`. (Una primera versión excluía el ALP por considerarlo un artefacto; la revisión independiente
mostró que esa conclusión era incorrecta.) El colesterol, con 10 vacíos mixtos (7 en donantes), no presenta este problema.

**Poblaciones de origen distintas.** Donantes y pacientes provienen de poblaciones diferentes: **ningún donante tiene
menos de 32 años**, mientras que los pacientes llegan a los 19; la edad mediana es de 47 años en los donantes y de 37 en
quienes tienen hepatitis. El modelo puede aprender diferencias de población y no solo de enfermedad (sección 6).

## 3. Exploración de los datos (0.5 pts)

- `figuras/clases.png`: desbalance extremo (533 contra 21–30).
- `figuras/laboratorio_por_clase.png`: las enzimas AST y GGT **suben con la gravedad** (mediana de AST: 24.8 en donantes,
  47.2 en hepatitis, 70.0 en fibrosis y 92.9 en cirrosis; de GGT: 21.4, 45.6, 72.2 y 96.4). En la cirrosis disminuyen la
  albúmina (33.0 contra 42.2 g/L), la colinesterasa (3.4 contra 8.4) y el colesterol (3.9 contra 5.4 mmol/L), y la
  bilirrubina aumenta (34.0 contra 6.9 µmol/L). Todo ello es coherente con la fisiología hepática [Giannini2005]. La
  hepatitis es la clase más difícil de separar del donante: su laboratorio es casi normal salvo por las enzimas.
- `figuras/faltantes_por_clase.png`: el ALP vacío (0 % en donantes; 12 %, 43 % y 20 % en hepatitis, fibrosis y
  cirrosis) y el colesterol vacío (1.3 % en donantes) son informativos.
- `figuras/correlacion.png`: AST, ALT y GGT están correlacionadas entre sí.
- **Valores extremos:** 1 persona con creatinina de 1 079 µmol/L (3 superan 300), 6 con bilirrubina mayor a 100 y 3 con
  GGT mayor a 400. Son valores posibles en enfermedad hepática o renal grave y se conservaron.

## 4. Modelo (2 pts)

**Partición.** 80 % para entrenamiento (486 personas: 426 donantes, 19 con hepatitis, 17 con fibrosis y 24 con
cirrosis) y 20 % para prueba (122: 107, 5, 4 y 6), estratificada, con `random_state = 42`. **En la prueba hay solo 4 a 6
casos por clase de enfermedad**: cada métrica de las clases raras es extremadamente ruidosa.

**Qué variables hacen falta.** Experimento con regresión logística (F1 macro de validación cruzada 5 × 4, solo
entrenamiento):

| Variables                                              | F1 macro (validación cruzada) |
|--------------------------------------------------------|-------------------------------|
| 12 (todas, con sexo)                                   | 0.582 ± 0.073                 |
| **11 (las elegidas: edad + 10 análisis, con ALP)**     | **0.595 ± 0.072**             |
| 10 (edad + 9 análisis, sin ALP)                        | 0.581 ± 0.062                 |
| 10 (solo laboratorio, sin edad)                        | 0.531 ± 0.065                 |
| 2 (solo edad y sexo)                                   | 0.210 ± 0.044                 |
| 11 + indicador de ALP faltante (diagnóstico, no se usa)| 0.604 ± 0.090                 |

El **sexo no aporta nada medible** (con él, 0.582; sin él, 0.595) y es un dato sensible: se excluye. La edad y el sexo
**solos no discriminan** (0.210, por debajo de la línea base de 0.234). La edad sí mejora el laboratorio (0.531 → 0.595),
pero esa mejora puede deberse a las diferencias de población de la sección 2; una comprobación independiente mostró que
rejuvenecer artificialmente a donantes reales por debajo de los 32 años casi no mueve su puntaje.

**Candidatos.** Cada uno es un `Pipeline` con imputación por la mediana y escalado de las 11 variables:

| Candidato                     | Justificación                                                                |
|-------------------------------|------------------------------------------------------------------------------|
| Clase mayoritaria (línea base)| Siempre predice "donante"                                                    |
| Regresión logística           | Modelo lineal con pesos de clase balanceados                                 |
| k vecinos                     | Clasificación por similitud con los casos más cercanos [Cover1967] (k = 5)   |
| Random Forest                 | Ensamble de árboles con pesos de clase balanceados [Breiman2001]             |

No se incluyó la SVM: su estimación de probabilidades (`probability=True`) está deprecada en scikit-learn 1.9.

**Pesos de clase.** Con 17 a 24 casos por clase de enfermedad se **ponderaron las clases** para que cuenten. A
diferencia de los modelos anteriores, los puntajes resultantes **no son probabilidades calibradas**, lo que la respuesta
de la API declara.

**Selección.** Solo con el entrenamiento, mediante validación cruzada estratificada repetida (5 × 4) con el F1 macro como
criterio [Sokolova2009]. El modelo elegido es el **Random Forest**, que además gana la validación cruzada en las 8 particiones alternativas que se probaron.

## 5. Evaluación (1 pt)

**Validación cruzada en entrenamiento:**

| Modelo                  | F1 macro (media ± desv.) | Exactitud balanceada |
|-------------------------|---------------------------|----------------------|
| Clase mayoritaria       | 0.234 ± 0.000             | 0.250                |
| Regresión logística     | 0.595 ± 0.072             | 0.662                |
| k vecinos               | 0.547 ± 0.093             | 0.511                |
| **Random Forest**       | **0.635 ± 0.111**         | 0.603                |

La diferencia entre el Random Forest y la regresión logística (0.040) es menor que la desviación entre pliegues (0.07 a
0.11), de modo que **no es concluyente** dentro de una misma partición; aun así, el Random Forest ganó en las 8 particiones
alternativas.

**Conjunto de prueba (122 personas)** con intervalos de confianza del 95 % por bootstrap [Efron1993]:

| Métrica                   | Valor  | IC 95 %          | Línea base |
|---------------------------|--------|------------------|------------|
| Exactitud                 | 0.926  | 0.877 – 0.967    | 0.877      |
| F1 macro                  | 0.580  | 0.368 – 0.753    | 0.234      |
| Precisión macro           | 0.770  | —                | 0.219      |
| Recall macro              | 0.529  | —                | 0.250      |

| Clase       | Precisión | Recall | Casos en la prueba |
|-------------|-----------|--------|--------------------|
| donante     | 0.947     | 1.000  | 107                |
| hepatitis   | 1.000     | 0.200  | 5                  |
| fibrosis    | 0.333     | 0.250  | 4                  |
| cirrosis    | 0.800     | 0.667  | 6                  |

Matriz de confusión (filas: real; columnas: predicho; `figuras/matriz_confusion.png`):

|               | donante | hepatitis | fibrosis | cirrosis |
|---------------|---------|-----------|----------|----------|
| **donante**   | 107     | 0         | 0        | 0        |
| **hepatitis** | 3       | 1         | 1        | 0        |
| **fibrosis**  | 2       | 0         | 1        | 1        |
| **cirrosis**  | 1       | 0         | 1        | 4        |

**La partición elegida (semilla 42) es la más desfavorable.** Con 8 particiones estratificadas distintas (semillas 0 a 7), el
F1 macro de prueba del Random Forest varió entre 0.62 y 0.73 (media 0.67): el 0.580 de la partición publicada es inferior a
todas ellas. No se cambió la semilla para mejorar el número; se informa tal cual. En esas 8 particiones, el recall de la
cirrosis fue de 0.67 a 1.0, el de la fibrosis de 0.25 a 0.75 y el de la hepatitis de 0.0 a 0.4.

**Estimaciones por clase menos ruidosas.** Las predicciones fuera de muestra del entrenamiento (486 personas) dan F1 macro
0.661 y recall de **donante 0.998, cirrosis 0.750, hepatitis 0.368 y fibrosis 0.353**.

**Vista binaria (enfermedad contra donante).** Agrupando hepatitis, fibrosis y cirrosis: en la prueba, sensibilidad 0.600
(9 de 15) y especificidad 1.000; fuera de muestra, 0.733 y 0.998. El modelo casi nunca confunde a un donante con un paciente,
pero deja sin detectar a entre 3 y 4 de cada 10 pacientes.

**El compromiso sensibilidad/especificidad.** La regresión logística, casi empatada en la validación cruzada, tiene el
compromiso opuesto: fuera de muestra, sensibilidad 0.917, especificidad 0.927 y recall de hepatitis 0.579 (el Random Forest,
0.733, 0.998 y 0.368). En las 8 particiones alternativas, su especificidad en la prueba fue de 0.87 a 0.95 y su sensibilidad
de 0.87 a 1.0. Que "la hepatitis casi no se detecte" es en parte **consecuencia de haber elegido el Random Forest**, que
privilegia la especificidad; para un tamizaje que no deba dejar pacientes sin detectar sería preferible la regresión logística.

## 6. Conclusión (0.5 pts)

Con la edad y diez análisis de sangre es posible distinguir a los donantes de los pacientes con una especificidad altísima
(0.99–1.00) y reconocer la cirrosis con un recall de 0.67 a 1.0, pero **la hepatitis y la fibrosis se detectan mal** (recall
fuera de muestra de 0.37 y 0.35 con el Random Forest), lo cual es esperable: sus análisis de laboratorio son cercanos a los
de un donante. Las enzimas AST y GGT, la bilirrubina, la albúmina, la colinesterasa y el ALP son las señales más claras.

Los hallazgos metodológicos principales fueron (1) que, ante un valor faltante sospechoso (el ALP, vacío solo en pacientes),
**se debe distinguir artefacto de señal**: rellenar al azar con valores observados y probar el indicador solo, lo que aquí
mostró señal real (en el `bmi` del modelo de ACV mostró artefacto); (2) que la elección entre dos modelos casi empatados
es en realidad un compromiso sensibilidad/especificidad que debe declararse; y (3) que con 4 a 6 casos por clase en la
prueba, el resultado depende mucho de la partición (F1 macro de 0.58 a 0.73).

**Limitaciones.** (1) **No es un diagnóstico.** (2) Solo hay 21 a 30 casos por clase de enfermedad; en la prueba, 4 a 6: el
intervalo del F1 macro (0.37–0.75) abarca desde un modelo mediocre hasta uno bueno. (3) **Poblaciones de origen distintas**
(los donantes tienen 32 años o más; los pacientes, desde 19): el modelo puede aprender la población y no la enfermedad; no hay
validación externa. (4) Los puntajes no son probabilidades calibradas (se ponderaron las clases). (5) Las etapas de fibrosis y
cirrosis se definieron con un criterio que la fuente no detalla. (6) Las unidades de los análisis no vienen documentadas en
la fuente; las del formulario (g/L, U/L, µmol/L, mmol/L, kU/L) se infirieron por el rango de los valores y deben confirmarse.
(7) Un valor aislado muy alterado (por ejemplo, una bilirrubina de 100) puede pasar inadvertido: el modelo se apoya sobre
todo en la AST. (8) El bootstrap cuenta como F1 cero la clase que no aparece en una remuestra, lo que podría sesgar a la baja
el intervalo del F1 macro.

## Referencias (en `docs_latex/referencias.bib`)

- **[Lichtinghagen2013]** Lichtinghagen, R., Pietsch, D., Bantel, H., Manns, M. P., Brand, K., & Bahr, M. J. (2013). The Enhanced Liver Fibrosis (ELF) score: Normal values, influence factors and proposed cut-off values. *Journal of Hepatology*, 59(2), 236–242.
- **[Hoffmann2018]** Hoffmann, G., Bietenbeck, A., Lichtinghagen, R., & Klawonn, F. (2018). Using machine learning techniques to generate laboratory diagnostic pathways—a case study. *Journal of Laboratory and Precision Medicine*, 3.
- **[Manns2017]** Manns, M. P., Buti, M., Gane, E., Pawlotsky, J.-M., Razavi, H., Terrault, N., & Younossi, Z. (2017). Hepatitis C virus infection. *Nature Reviews Disease Primers*, 3, 17006. https://doi.org/10.1038/nrdp.2017.6
- **[Giannini2005]** Giannini, E. G., Testa, R., & Savarino, V. (2005). Liver enzyme alteration: a guide for clinicians. *CMAJ*, 172(3), 367–379. https://doi.org/10.1503/cmaj.1040752
- **[Cover1967]** Cover, T. M., & Hart, P. E. (1967). Nearest neighbor pattern classification. *IEEE Transactions on Information Theory*, 13(1), 21–27.
- **[Breiman2001]** Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32. https://doi.org/10.1023/A:1010933404324
- **[Sokolova2009]** Sokolova, M., & Lapalme, G. (2009). A systematic analysis of performance measures for classification tasks. *Information Processing & Management*, 45(4), 427–437. https://doi.org/10.1016/j.ipm.2009.03.002
- **[Efron1993]** Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.

> Verificadas por búsqueda web el 2026-10-10. Pendiente de confirmar: el número de artículo de Hoffmann et al. (2018).
