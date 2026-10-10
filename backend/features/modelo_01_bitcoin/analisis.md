# Modelo 01 · Predicción del precio del Bitcoin

<!-- Notas para LaTeX: cada sección corresponde a un criterio de la rúbrica. Las figuras están en
     figuras/*.png y las métricas exactas en metricas.json (se regeneran con train.py). Las referencias
     [Clave] están en docs_latex/referencias.bib. -->

## 1. Análisis del problema (0.5 pts)

Se plantea estimar el **precio de cierre del Bitcoin en dólares** (USD) para el día siguiente y hasta siete días
después del último dato disponible. Es un problema de **regresión sobre una serie de tiempo**, cuya variable objetivo es
`Close`. El usuario solo indica cuántos días adelante quiere la estimación (por defecto, 1): el modelo parte de su propio historial.

Dos particularidades condicionan todo el trabajo:

1. **La referencia obligada es la persistencia** ("mañana = hoy"). La hipótesis de los mercados eficientes en su forma débil [Fama1970]
   sostiene que los precios pasados no sirven para predecir los futuros, y la evidencia empírica con modelos complejos suele coincidir:
   en la literatura de tipos de cambio, ningún modelo estimado mejoró de forma consistente al paseo aleatorio fuera de muestra [Meese1983].
   Para el Bitcoin existen estudios que lo encuentran ineficiente en algunos períodos [Urquhart2016] y trabajos de aprendizaje automático
   que intentan predecirlo [McNally2018]; por eso la pregunta no es "qué error tiene el modelo", sino **si mejora de forma demostrable a repetir el último precio**.
2. **Los datos terminan el 31 de julio de 2017.** "Mañana" significa el 1 de agosto de 2017, y la respuesta de la API lo declara: el modelo no conoce
   el precio actual.

## 2. Entendimiento de los datos (0.5 pts)

El conjunto [TeamAI2017bitcoin] tiene 1 556 filas diarias, del 28 de abril de 2013 al 31 de julio de 2017. Las filas vienen de la fecha más reciente
a la más antigua y se ordenan; las fechas (`Jul 31, 2017`) se leen con el formato mes-día-año. No hay fechas repetidas ni huecos (los 1 555 intervalos entre registros son de
exactamente un día, lo que la predicción recursiva necesita), y en todas las filas se cumple mínimo ≤ apertura, cierre ≤ máximo.

| Variable                  | Tipo     | Uso                                                                                       |
|---------------------------|----------|-------------------------------------------------------------------------------------------|
| `Close`                   | objetivo | Cierre diario (USD): media 584, desviación 526, mínimo 68.43 (5-jul-2013), máximo 2 958.11 (11-jun-2017) |
| `Date`                    | fecha    | Ordena la serie                                                                           |
| `Open`, `High`, `Low`     | numéricas | **Excluidas** (se explican abajo)                                                          |
| `Volume`                  | numérica | **Excluida**: 243 valores `-` (del 28-abr al 26-dic-2013), con separadores de miles        |
| `Market Cap`              | numérica | **Excluida**                                                                              |

**Exclusión de variables.** Para predecir varios días hay que encadenar predicciones (el cierre estimado de mañana entra al cálculo de pasado mañana),
y eso solo es posible con variables que el propio modelo pueda producir: se usan únicamente **cierres pasados**. Apertura, máximo, mínimo, volumen y capitalización
del día siguiente no se conocen al consultar ni se pueden predecir por separado. El volumen, además, falta en los primeros 243 días.

**Variables construidas** (`serie.py`, el mismo código en el entrenamiento y la API): retornos logarítmicos de 1 a 5, 7 y 30 días; distancia del cierre a su media de 7 y 30 días (en
logaritmos); y desviación de los retornos de 7 y 30 días. **El objetivo no es el precio sino el retorno logarítmico del día siguiente**, y el precio se obtiene como cierre × exp(retorno): así el modelo no tiene que aprender la escala,
que cambió 20 veces a lo largo de la serie.

## 3. Exploración de los datos (0.5 pts)

- `figuras/serie_precio.png` (escala logarítmica): el precio sube de 134 a 2 875 USD con episodios de burbuja y caída (2013-2014) y un rally sostenido al final. **La prueba (últimos 312 días, desde el
  23-sep-2016) es un mercado alcista: el precio se multiplica por 4.77.** Esto importa para interpretar cualquier métrica de dirección.
- `figuras/distribucion_retornos.png`: el retorno diario tiene media 0.20 %, desviación 4.26 %, mínimo −26.6 % y máximo +35.7 %: **colas pesadas**, una regularidad empírica general de los activos financieros [Cont2001].
- `figuras/autocorrelacion.png`: la autocorrelación de los retornos es casi nula (rezagos 1 a 5: −0.001, −0.043, −0.019, 0.064, 0.039; la cota del 95 % es ±0.050; algunos rezagos aislados superan la cota por poco), mientras que la de
  los retornos al cuadrado es alta (0.32 en el rezago 1): **el signo del movimiento no se repite, pero su tamaño sí** (agrupamiento de la volatilidad [Cont2001]). Esto anticipa lo que se encuentra más adelante.
- `figuras/volatilidad.png`: la volatilidad móvil de 30 días cambia mucho con el tiempo: la desviación diaria por año fue 6.7 % (2013), 3.9 % (2014), 3.7 % (2015), 2.5 % (2016) y 4.4 % (2017). Por eso el σ del intervalo (calculado con todo el entrenamiento, 4.2 %) es una aproximación gruesa.
- El precio sube en el 54.5 % de los días de todo el período.

## 4. Modelo (2 pts)

**Partición temporal.** La prueba son los **últimos 312 días** (del 23-sep-2016 al 31-jul-2017); el entrenamiento, 1 213 días de origen hasta el 22-sep-2016 (cada fila usa los cierres hasta ese día y tiene como objetivo el
retorno del día siguiente). Ningún objetivo de entrenamiento cae en la prueba, y una prueba automática infla diez veces los cierres de la prueba y exige que la selección y el ajuste no cambien [Hyndman2021].

**Candidatos.** Dos líneas base (que **no pueden ser elegidas**) y tres modelos de aprendizaje automático:

| Candidato                          | Qué hace                                                                              |
|------------------------------------|---------------------------------------------------------------------------------------|
| Persistencia (línea base)          | Retorno predicho = 0: mañana = hoy                                                      |
| Deriva (línea base)                | Retorno predicho = promedio del entrenamiento (0.126 % diario)                          |
| Ridge                              | Regresión lineal con penalización L2 [Hoerl1970], variables estandarizadas, α elegido por validación cruzada interna |
| Random Forest                      | 200 árboles de profundidad ≤ 6 y ≥ 20 días por hoja [Breiman2001]                        |
| Gradient Boosting                  | Histogramas, 100 iteraciones, tasa 0.03, profundidad ≤ 3 [Friedman2001; Pedregosa2011]   |

**Selección.** Solo con el entrenamiento y con validación cruzada de **ventana creciente** (5 pliegues) sobre el RMSE del retorno diario [Bergmeir2012]. El conjunto de prueba se usó una vez, para medir.

**Predicción a varios días.** Se predice el retorno de un día, se calcula el cierre, se agrega al historial y se repite hasta el horizonte pedido (máximo 7). El **intervalo del 95 %** que se informa es
cierre × exp(± 1.96 · σ · √días), donde σ (4.2 %) es la desviación del retorno diario en el entrenamiento: un supuesto de paseo aleatorio que la evaluación pone a prueba.

## 5. Evaluación (1 pt)

**Validación cruzada en el entrenamiento** (RMSE del retorno diario, media ± desviación entre pliegues):

| Modelo                      | RMSE            |
|-----------------------------|-----------------|
| Persistencia (línea base)   | 0.03641 ± 0.01025 |
| Deriva (línea base)         | 0.03665 ± 0.01053 |
| **Ridge (elegido)**         | **0.03665 ± 0.01053** |
| Random Forest               | 0.03716 ± 0.01117 |
| Gradient Boosting           | 0.03776 ± 0.01177 |

Ningún modelo mejora a la persistencia, y las diferencias (0.0002–0.0013) son mucho menores que la variación entre pliegues (≈ 0.01). Los dos modelos de árboles empeoran con su flexibilidad: ajustan ruido.
Entre los de aprendizaje automático gana el Ridge, **que coincide exactamente con la deriva**: el α elegido es el máximo de la rejilla (10⁵), todos los coeficientes quedan en ≈ 0 (el mayor, 5·10⁻⁵ en unidades estandarizadas) y
el modelo se reduce a su intercepto. Es decir, **el Ridge "decide" que las variables de rezago no aportan nada** y predice el último cierre multiplicado por el crecimiento medio.

**Prueba, 306 orígenes por horizonte** (cada día de la prueba como punto de partida, con las siete predicciones posibles con dato real; se predice y compara con el cierre real). Habilidad = 1 − RMSE(modelo)/RMSE(persistencia); su IC 95 %
sale de un bootstrap por bloques circulares de 14 días [Kunsch1989], porque los errores de días vecinos están correlacionados:

| Días adelante | RMSE modelo (USD) | RMSE persistencia (USD) | MAPE modelo / persistencia | Habilidad (IC 95 %)         | Cobertura del intervalo 95 % |
|---------------|-------------------|--------------------------|-----------------------------|------------------------------|------------------------------|
| 1             | 68.0              | 68.1                     | 2.41 % / 2.43 %             | +0.17 % (−0.23 %, +0.67 %)   | 96.4 %                       |
| 3             | 116.0             | 116.6                    | 4.51 % / 4.61 %             | +0.55 % (−0.56 %, +1.78 %)   | 95.8 %                       |
| 7             | 174.3             | 177.0                    | 7.04 % / 7.25 %             | +1.53 % (−0.87 %, +3.50 %)   | 97.1 %                       |

- **No hay mejora demostrable sobre repetir el último precio**: los tres intervalos incluyen el 0. La pequeña ventaja nominal viene solo de la deriva positiva en un mercado alcista (la deriva obtiene RMSE casi idéntico: 68.00, 115.99, 174.23).
- El **acierto de dirección** del modelo (63.1 %, 67.7 % y 69.3 % a 1, 3 y 7 días) es **exactamente igual al de predecir siempre "sube"**, porque la deriva es positiva y el modelo siempre predice una subida. En la prueba el precio subió en esa proporción de días. No es capacidad predictiva.
- El error crece con el horizonte (RMSE de 68 a 174 USD; MAPE de 2.4 % a 7.0 %), como en un paseo aleatorio.
- Los intervalos del 95 % cubren entre 95.8 % y 97.1 % de los cierres reales: están **bien calibrados** y es lo más útil que ofrece el servicio.
- `figuras/prueba_7_dias.png` muestra la predicción casi superpuesta con la persistencia y rezagada respecto del precio real: llega tarde a cada giro.

**Despliegue.** Las métricas son las de la partición temporal; el modelo que sirve la API se reentrena con **todos** los días para partir del último cierre real (2 875.34 USD). Con todo el período la deriva sube a 0.20 % diario, y por eso
la API predice 2 881.6 USD para el 1-ago-2017 y 2 917.9 USD para 7 días. Estas cifras del modelo final no se pueden medir con datos independientes.

## 6. Conclusión (0.5 pts)

El Bitcoin del período 2013-2017 se comporta, para fines de predicción con su propio historial, como un **paseo aleatorio con deriva**: ni los rezagos, ni las medias móviles, ni la volatilidad, ni modelos de árboles aportan una ventaja sobre repetir el último precio.
El modelo entregado es un Ridge que lo reconoce (coeficientes ≈ 0), de modo que su pronóstico es "el último cierre más el crecimiento medio". Se publica porque cumple con evaluar con honestidad y porque su producto útil no es el punto sino el **rango del 95 %**,
que sí está calibrado y que crece con la raíz del horizonte.

**Limitaciones.** (1) Los datos terminan el 31 de julio de 2017; la API no conoce el precio actual y lo advierte siempre. (2) Una sola ventana de prueba, de un mercado alcista: la ventaja de dirección es engañosa y la deriva positiva se extrapola hacia adelante; en un mercado bajista el modelo seguiría prediciendo subidas.
(3) El intervalo supone volatilidad constante (la serie muestra agrupamiento de volatilidad), aunque la cobertura en la prueba fue adecuada. (4) Solo se usó el historial de cierres: ninguna información externa (noticias, regulación, volumen) ni exógena. (5) La conclusión depende de los candidatos probados; no descarta que modelos de otra clase
(p. ej., redes recurrentes con más variables) encuentren estructura, aunque la evidencia de eficiencia débil [Fama1970; Meese1983] hace improbable una mejora sostenida. (6) **No es una recomendación de inversión.**

## Referencias (en `docs_latex/referencias.bib`)

- **[TeamAI2017bitcoin]** Conjunto de datos *Bitcoin Price Prediction* (`bitcoin_price_Training - Training.csv`). Kaggle. https://www.kaggle.com/team-ai/bitcoin-price-prediction
- **[Fama1970]** Fama, E. F. (1970). Efficient capital markets: A review of theory and empirical work. *The Journal of Finance*, 25(2), 383–417. https://doi.org/10.1111/j.1540-6261.1970.tb00518.x
- **[Meese1983]** Meese, R. A., & Rogoff, K. (1983). Empirical exchange rate models of the seventies: Do they fit out of sample? *Journal of International Economics*, 14(1–2), 3–24.
- **[Cont2001]** Cont, R. (2001). Empirical properties of asset returns: Stylized facts and statistical issues. *Quantitative Finance*, 1(2), 223–236. https://doi.org/10.1080/713665670
- **[Urquhart2016]** Urquhart, A. (2016). The inefficiency of Bitcoin. *Economics Letters*, 148, 80–82. https://doi.org/10.1016/j.econlet.2016.09.019
- **[McNally2018]** McNally, S., Roche, J., & Caton, S. (2018). Predicting the price of Bitcoin using machine learning. *26th Euromicro International Conference on PDP*, 339–343. IEEE. *(falta confirmar el DOI en IEEE Xplore)*
- **[Kunsch1989]** Künsch, H. R. (1989). The jackknife and the bootstrap for general stationary observations. *The Annals of Statistics*, 17(3), 1217–1241. https://doi.org/10.1214/aos/1176347265
- **[Bergmeir2012]** Bergmeir, C., & Benítez, J. M. (2012). On the use of cross-validation for time series predictor evaluation. *Information Sciences*. https://doi.org/10.1016/j.ins.2011.12.028 *(falta confirmar volumen y páginas)*
- **[Hyndman2021]** Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3.ª ed.). OTexts. https://otexts.com/fpp3/
- **[Hoerl1970]** Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression: Biased estimation for nonorthogonal problems. *Technometrics*, 12(1), 55–67.
- **[Breiman2001]** Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.
- **[Friedman2001]** Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *The Annals of Statistics*, 29(5), 1189–1232. https://doi.org/10.1214/aos/1013203451
- **[Pedregosa2011]** Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
