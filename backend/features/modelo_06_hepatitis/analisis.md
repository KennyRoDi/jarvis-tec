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
partir de la edad, el sexo y diez análisis de laboratorio, cuya interpretación clínica es conocida [Giannini2005]. Es un
problema de **clasificación multiclase supervisada, muy desbalanceada**: 88 % son donantes y las tres clases de
enfermedad tienen solo 21 a 30 casos cada una.

El asistente JarvisTEC responde con la clase más probable y los puntajes de cada clase a partir de los valores que
el usuario ingresa en el formulario.

## 2. Entendimiento de los datos (0.5 pts)

El conjunto reúne los datos de laboratorio de donantes de sangre y de pacientes con hepatitis C [Lichtinghagen2013;
Hoffmann2018]: 615 personas y 13 columnas útiles (clase, edad, sexo y diez análisis: ALB, ALP, ALT, AST, BIL, CHE, CHOL,
CREA, GGT y PROT). El archivo incluye una columna de índice (`Unnamed: 0`) que se descarta. No hay filas duplicadas.

**Clase ambigua.** Siete personas tienen la categoría "suspect Blood Donor". No se parecen a un donante sano (su
albúmina mediana es de 21.6 g/L y sus proteínas totales de 47.8 g/L) ni corresponden a una etapa de la enfermedad, por
lo que se **excluyen**. Quedan **608 personas**: 533 donantes, 24 con hepatitis, 21 con fibrosis y 30 con cirrosis.

**Valores faltantes informativos.** Hay 18 valores vacíos de ALP, **todos en pacientes** (24 % de los 75 pacientes;
43 % de los de fibrosis) y **ninguno en donantes**. Es un artefacto de la recolección y no una característica de la
enfermedad. Los vacíos de colesterol (10) son pocos y mixtos (7 en donantes). Hay un vacío más de ALB, ALT y PROT.

**Hallazgo: el artefacto del ALP se filtra a los árboles.** Al imputar el ALP con la mediana, el Random Forest alcanza
un F1 macro de validación cruzada de 0.642 y sin él baja a 0.588; la regresión logística, en cambio, no cambia (0.582
contra 0.576). Es el mismo patrón que se encontró con el `bmi` en el modelo de ACV: un modelo de árboles aísla el pico
de valores imputados y "descubre" que el valor faltante predice la enfermedad. La aplicación siempre recibe los datos
completos, así que ese aporte es inalcanzable y espurio. Por eso **el ALP se excluye del modelo**. Un indicador de "ALP
faltante" se mide en el experimento pero no se usa.

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

| Variables                                               | F1 macro (validación cruzada) |
|---------------------------------------------------------|-------------------------------|
| 12 (todas, ALP imputada)                                | 0.582 ± 0.073                 |
| **11 (las elegidas, sin ALP)**                          | **0.576 ± 0.066**             |
| 9 (solo laboratorio, sin edad ni sexo)                  | 0.519 ± 0.079                 |
| 2 (solo edad y sexo)                                    | 0.210 ± 0.044                 |
| 12 + indicador de ALP faltante (artefacto, no se usa)   | 0.586 ± 0.066                 |

La edad y el sexo **solos no discriminan** (0.210, por debajo de la línea base de 0.234), pero al sumarlos al laboratorio
mejoran 0.057, algo menor que la desviación entre pliegues. Se conservan, pero esa mejora puede deberse a las
diferencias de población descritas en la sección 2.

**Candidatos.** Cada uno es un `Pipeline` con imputación por la mediana, escalado y codificación del sexo:

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
criterio [Sokolova2009]. El modelo elegido es el **Random Forest**.

## 5. Evaluación (1 pt)

**Validación cruzada en entrenamiento:**

| Modelo                  | F1 macro (media ± desv.) | Exactitud balanceada |
|-------------------------|---------------------------|----------------------|
| Clase mayoritaria       | 0.234 ± 0.000             | 0.250                |
| Regresión logística     | 0.576 ± 0.066             | 0.660                |
| k vecinos               | 0.516 ± 0.083             | 0.488                |
| **Random Forest**       | **0.588 ± 0.117**         | 0.559                |

Las diferencias entre la regresión logística y el Random Forest (0.012) son mucho menores que las desviaciones (0.07 y
0.12): **no son concluyentes**. El Random Forest se eligió por tener el valor más alto del criterio, pero la regresión
logística es más estable y tiene mejor exactitud balanceada (0.660 contra 0.559).

**Conjunto de prueba (122 personas)** con intervalos de confianza del 95 % por bootstrap [Efron1993]:

| Métrica                   | Valor  | IC 95 %          | Línea base |
|---------------------------|--------|------------------|------------|
| Exactitud                 | 0.934  | 0.893 – 0.975    | 0.877      |
| F1 macro                  | 0.640  | 0.411 – 0.815    | 0.234      |
| Precisión macro           | 0.772  | —                | 0.219      |
| Recall macro              | 0.579  | —                | 0.250      |

| Clase       | Precisión | Recall | Casos en la prueba |
|-------------|-----------|--------|--------------------|
| donante     | 0.955     | 1.000  | 107                |
| hepatitis   | 1.000     | 0.400  | 5                  |
| fibrosis    | 0.333     | 0.250  | 4                  |
| cirrosis    | 0.800     | 0.667  | 6                  |

Matriz de confusión (filas: real; columnas: predicho; `figuras/matriz_confusion.png`):

|               | donante | hepatitis | fibrosis | cirrosis |
|---------------|---------|-----------|----------|----------|
| **donante**   | 107     | 0         | 0        | 0        |
| **hepatitis** | 2       | 2         | 1        | 0        |
| **fibrosis**  | 2       | 0         | 1        | 1        |
| **cirrosis**  | 1       | 0         | 1        | 4        |

**Estimaciones por clase menos ruidosas.** Como la prueba tiene tan pocos casos, se calcularon también las predicciones
fuera de muestra del entrenamiento (486 personas): F1 macro 0.587; recall de **donante 0.988, cirrosis 0.750, fibrosis
0.412 y hepatitis 0.105**. La hepatitis es la clase que casi nunca se detecta.

**Vista binaria (enfermedad contra donante).** Agrupando hepatitis, fibrosis y cirrosis: en la prueba, sensibilidad 0.667
(10 de 15) y especificidad 1.000; fuera de muestra, 0.617 y 0.988. El modelo casi no confunde a un donante con un
paciente, pero deja sin detectar a entre 3 y 4 de cada 10 pacientes.

## 6. Conclusión (0.5 pts)

Con edad, sexo y diez análisis de sangre es posible distinguir a los donantes de los pacientes con una especificidad
altísima (0.99–1.00) y reconocer la cirrosis con un recall cercano a 0.7, pero **la hepatitis y la fibrosis se detectan
mal** (recall de 0.1 a 0.4 fuera de muestra), lo cual es esperable: sus análisis de laboratorio son cercanos a los de un
donante. Las enzimas AST y GGT, la bilirrubina, la albúmina y la colinesterasa son las señales más claras.

El hallazgo metodológico principal es que **el ALP faltante, informativo, se filtraba al Random Forest** (F1 de 0.642 con
la ALP imputada contra 0.588 sin ella), igual que el `bmi` en el modelo de ACV. La regresión logística no se ve afectada,
lo que sugiere un criterio práctico: comprobar siempre con y sin la variable.

**Limitaciones.** (1) **No es un diagnóstico.** (2) Solo hay 21 a 30 casos por clase de enfermedad; en la prueba, 4 a 6:
el intervalo del F1 macro (0.41–0.81) abarca desde un modelo mediocre hasta uno bueno. (3) **Poblaciones de origen
distintas** (los donantes tienen 32 años o más; los pacientes, desde 19): el modelo puede aprender la población y no la
enfermedad; no hay validación externa. (4) Los puntajes no son probabilidades calibradas (se ponderaron las clases).
(5) Las etapas de fibrosis y cirrosis se definieron con un criterio que la fuente no detalla. (6) Las unidades de los
análisis no vienen documentadas en la fuente; las del formulario (g/L, U/L, µmol/L, mmol/L, kU/L) se infirieron por
el rango de los valores y deben confirmarse. (7) Con un conjunto tan pequeño, los resultados cambiarían con otra
partición.

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
