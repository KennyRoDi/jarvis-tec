# Modelo 04 · Clasificación de abandono de clientes de telefonía (churn)

<!-- Notas para LaTeX: cada sección corresponde a un criterio de la rúbrica. Las figuras están en
     figuras/*.png y las métricas exactas en metricas.json (se regeneran con train.py). Las referencias
     [Clave] están en docs_latex/referencias.bib. -->

## 1. Análisis del problema (0.5 pts)

Retener a un cliente cuesta menos que conseguir uno nuevo, pero solo si se sabe a quién ofrecerle algo antes de
que se vaya. Se plantea estimar la **probabilidad de que un cliente abandone la compañía telefónica** a partir
de su contrato, sus servicios y su facturación. Es un problema de **clasificación binaria supervisada
desbalanceada**: la variable objetivo `Churn` vale "Yes" en 26.5 % de los clientes. En este contexto los dos
errores no cuestan lo mismo, y los modelos de abandono suelen evaluarse según el beneficio de la campaña de
retención y no solo con métricas estadísticas [Verbeke2012].

El asistente JarvisTEC responde consultas como "¿se va a pasar de compañía este cliente?" con la probabilidad y
una clasificación de riesgo alto o bajo.

## 2. Entendimiento de los datos (0.5 pts)

El conjunto de datos de IBM contiene 7 043 clientes y 21 columnas [IBM2018]: 19 variables predictoras, el
identificador y el objetivo. **Es un conjunto de datos de ejemplo ficticio** (muestra de IBM Cognos Analytics), no
datos reales de una compañía, lo que limita la validez de las conclusiones (sección 6).

| Grupo         | Variables                                                                                      |
|---------------|------------------------------------------------------------------------------------------------|
| Cuenta        | `tenure` (meses), `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges` |
| Servicios     | `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` |
| Demografía    | `gender`, `SeniorCitizen`, `Partner`, `Dependents`                                              |

- **`TotalCharges`** viene como texto y tiene 11 cadenas vacías, todas de clientes con 0 meses de antigüedad
  (ninguno ha abandonado). Además es redundante: su correlación con `tenure × MonthlyCharges` es de 0.9996. Se
  descarta.
- **Categorías dependientes.** Sin servicio de internet, los seis servicios adicionales valen siempre "No internet
  service"; sin teléfono, `MultipleLines` vale "No phone service". La API valida esta coherencia.
- **Filas repetidas:** 40 filas repiten exactamente las 19 variables de otra (48 con las 18 que quedan sin
  `TotalCharges`), pero son clientes distintos, cada uno con su identificador, y pueden tener un abandono distinto;
  se conservan. Con las 9 variables del modelo es normal que muchos perfiles coincidan (son variables de pocos valores).
- No hay valores nulos aparte de los descritos.

## 3. Exploración de los datos (0.5 pts)

- `figuras/abandono_por_categoria.png`: el **tipo de contrato** es la señal más fuerte: abandonan 42.7 % de los
  clientes mes a mes, 11.3 % con contrato de un año y 2.8 % con contrato de dos años. Con fibra óptica abandonan
  41.9 % (DSL 19.0 %, sin internet 7.4 %) y con cheque electrónico 45.3 % (frente a 15–19 % con otros métodos).
- Sin soporte técnico abandonan 41.6 % (con soporte, 15.2 %) y sin seguridad en línea 41.8 % (con ella, 14.6 %).
  Los adultos mayores abandonan más (41.7 % contra 23.6 %) y la factura electrónica se asocia a más abandono
  (33.6 % contra 16.3 %).
- `figuras/antiguedad_y_mensualidad.png`: quienes abandonan tienen **poca antigüedad** (mediana de 10 meses frente a
  38) y **mensualidades más altas** (mediana de 79.7 USD frente a 64.4).

## 4. Modelo (2 pts)

**Partición.** 80 % para entrenamiento (5 634 clientes) y 20 % para prueba (1 409), estratificada por el
abandono, con `random_state = 42`. La exploración de la sección 3 es descriptiva y usa los 7 043 clientes; la
elección de las variables, del modelo y del umbral se hizo solo con el conjunto de entrenamiento. Los
hiperparámetros del bosque (150 árboles, `min_samples_leaf = 10`, profundidad 10) se fijaron a priori, sin búsqueda
ni ajuste con la prueba.

**Cuántas variables hacen falta.** El formulario de la aplicación sería incómodo con 18 campos, por lo que se
midió con la regresión logística y validación cruzada (solo entrenamiento) cuánto se pierde al reducirlas:

| Variables                              | AUC ROC (validación cruzada) |
|----------------------------------------|------------------------------|
| 18 (todas, sin `TotalCharges`)         | 0.8450 ± 0.0116              |
| **9 (las elegidas)**                   | **0.8426 ± 0.0110**          |
| 6 mínimas                              | 0.8389 ± 0.0117              |
| 3 (antigüedad, mensualidad, contrato)  | 0.8276 ± 0.0126              |

Con 9 variables se pierden 0.0024, cinco veces menos que la desviación entre pliegues, por lo que se adoptan:
`tenure`, `monthly_charges`, `contract`, `internet_service`, `payment_method`, `paperless_billing`, `tech_support`,
`online_security` y `senior_citizen`.

**Candidatos.** Cada uno es un `Pipeline` (escala de las numéricas y codificación *one-hot* de las categóricas):

| Candidato                 | Justificación                                                              |
|---------------------------|----------------------------------------------------------------------------|
| Tasa base (línea base)    | Predice siempre la proporción de abandono: no discrimina (AUC = 0.5)        |
| Regresión logística       | Modelo lineal interpretable, habitual en abandono                           |
| Random Forest             | Ensamble de árboles que capta interacciones (150 árboles, `min_samples_leaf = 10`, profundidad 10) |
| Gradient Boosting         | Ensamble secuencial de árboles (versión de histogramas de scikit-learn)     |

**Sin pesos de clase.** No se ponderaron las clases: la ponderación distorsiona las probabilidades, y la
aplicación muestra al usuario la probabilidad de abandono, que debe ser fiable [NiculescuMizil2005]. El
desbalance se trata ajustando el **umbral de decisión**.

**Selección.** Solo con el conjunto de entrenamiento, mediante validación cruzada estratificada repetida (5 × 2)
con el AUC ROC como criterio [Fawcett2006]. El modelo elegido es **Random Forest**.

**Umbral.** Se eligió el umbral que maximiza el F1 de la clase "Yes" usando **predicciones fuera de muestra** del
entrenamiento (validación cruzada de 5 particiones), nunca el conjunto de prueba: **0.35**.

## 5. Evaluación (1 pt)

**Validación cruzada en entrenamiento:**

| Modelo                | AUC ROC (media ± desv.) | AUC PR |
|-----------------------|--------------------------|--------|
| Tasa base (línea base)| 0.500 ± 0.000            | 0.265  |
| Regresión logística   | 0.8426 ± 0.0110          | 0.650  |
| **Random Forest**     | **0.8452 ± 0.0110**      | 0.655  |
| Gradient Boosting     | 0.8387 ± 0.0116          | 0.650  |

Los tres modelos reales no se distinguen entre sí (diferencias de 0.003 a 0.007 con una desviación de 0.011);
Random Forest se eligió por tener el valor más alto, pero la regresión logística, más simple, rinde casi igual.

**Conjunto de prueba (1 409 clientes, 374 de ellos abandonaron):**

| Métrica                 | Modelo | Línea base |
|-------------------------|--------|------------|
| AUC ROC                 | 0.840  | 0.500      |
| AUC PR                  | 0.649  | 0.265 (tasa base) |
| Brier (menor es mejor)  | 0.138  | 0.195 (tasa base) |

El AUC de prueba (0.840) es coherente con el de validación cruzada (0.845). La métrica de Brier mide la calidad de
las probabilidades [NiculescuMizil2005]: 0.138 frente a 0.195 de predecir siempre la tasa base. Con datos
desbalanceados conviene mirar también el AUC PR [Saito2015].

**Efecto del umbral** (prueba):

| Umbral             | Exactitud | Precisión ("Yes") | Recall ("Yes") | F1 ("Yes") | Matriz [[TN, FP], [FN, TP]] |
|--------------------|-----------|-------------------|----------------|------------|------------------------------|
| **0.35 (elegido)** | 0.769     | 0.553             | **0.690**      | **0.614**  | [[826, 209], [116, 258]]     |
| 0.50               | 0.801     | 0.656             | 0.529          | 0.586      | [[931, 104], [176, 198]]     |
| Predecir siempre "No" | 0.735  | —                 | 0.000          | —          | —                            |

Métricas macro al umbral de 0.35 (promedio de las dos clases): precisión 0.715, recall 0.744 y F1 **0.725**
(0.728 al umbral de 0.50).

Con el umbral de 0.35, el modelo detecta 258 de los 374 clientes que se van (69 %), a costa de contactar a 209
que se habrían quedado. Su precisión (55 %) duplica la tasa base (26.5 %). Con el umbral de 0.50 se detectan solo
198 (53 %) pero se molesta a menos clientes. **El umbral óptimo depende del coste de la campaña de retención**;
aquí se usó F1, que da el mismo peso a ambos errores. El umbral exacto es poco crítico: con otras semillas del
cálculo osciló entre 0.32 y 0.35, y el F1 de prueba se mantiene entre 0.60 y 0.63 para cualquier umbral de 0.20 a
0.45.

**Calibración** (`figuras/evaluacion.png`): las probabilidades predichas se parecen a las frecuencias reales. Por
ejemplo, donde el modelo predice en promedio 0.26 la tasa real es 0.28, y donde predice 0.45, 0.40 [más
tramos en `metricas.json`]. Por eso la aplicación puede mostrarlas como probabilidades.

## 6. Conclusión (0.5 pts)

Con solo 9 datos del cliente es posible ordenarlos por riesgo de abandono con un AUC de 0.84 y detectar a siete
de cada diez clientes que se irán, con probabilidades bien calibradas. Los factores asociados al abandono son el
**contrato mes a mes, la poca antigüedad, la fibra óptica, el pago con cheque electrónico y la falta de soporte
técnico y de seguridad en línea**. Los resultados son asociaciones, no causas: no prueban que cambiar el contrato
de un cliente evite su abandono.

Los hallazgos metodológicos principales fueron (1) que 9 variables rinden prácticamente igual que 18, lo que
permite un formulario manejable, y (2) que con un 26.5 % de abandono el umbral de 0.5 desaprovecha el modelo: el
umbral de 0.35 sube el recall de 0.53 a 0.69.

**Limitaciones.** (1) El conjunto de datos es una muestra **ficticia** de IBM: sirve para construir y evaluar el
método, no para tomar decisiones reales. (2) Es una instantánea sin dimensión temporal; no se puede evaluar cómo se
comportaría el modelo con clientes futuros. (3) El umbral se eligió con F1, sin conocer los costos reales de una
campaña de retención [Verbeke2012]. (4) La prueba es una sola partición de 1 409 clientes. (5) Los tres modelos
reales rinden casi igual, de modo que la elección de Random Forest no es concluyente.

## Referencias (en `docs_latex/referencias.bib`)

- **[IBM2018]** IBM. (2018). *Telco customer churn* [Conjunto de datos de ejemplo de IBM Cognos Analytics]. https://github.com/IBM/telco-customer-churn-on-icp4d
- **[Verbeke2012]** Verbeke, W., Dejaeger, K., Martens, D., Hur, J., & Baesens, B. (2012). New insights into churn prediction in the telecommunication sector: A profit driven data mining approach. *European Journal of Operational Research*, 218(1), 211–229. https://doi.org/10.1016/j.ejor.2011.09.031
- **[Fawcett2006]** Fawcett, T. (2006). An introduction to ROC analysis. *Pattern Recognition Letters*, 27(8), 861–874. https://doi.org/10.1016/j.patrec.2005.10.010
- **[Saito2015]** Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLoS ONE*, 10(3), e0118432. https://doi.org/10.1371/journal.pone.0118432
- **[NiculescuMizil2005]** Niculescu-Mizil, A., & Caruana, R. (2005). Predicting good probabilities with supervised learning. En *Proceedings of the 22nd International Conference on Machine Learning* (pp. 625–632). https://doi.org/10.1145/1102351.1102430

> Verificadas por búsqueda web el 2026-10-09. El repositorio de IBM está archivado desde 2024; la muestra proviene de IBM Cognos Analytics.
