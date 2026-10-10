# Modelo 10 · Predicción del precio de acciones del S&P 500

<!-- Notas para LaTeX: cada sección corresponde a un criterio de la rúbrica. Las figuras están en
     figuras/*.png y las métricas exactas en metricas.json (se regeneran con train.py). Las referencias
     [Clave] están en docs_latex/referencias.bib. Es el mismo enfoque que el modelo 01 (Bitcoin), aplicado a cuatro series. -->

## 1. Análisis del problema (0.5 pts)

Se plantea estimar el **precio de cierre, en dólares, de una acción del S&P 500** (Apple, Microsoft, Amazon o Google) para la
sesión de bolsa siguiente y hasta siete sesiones después del último dato. Es un problema de **regresión sobre series de tiempo**
(una por símbolo). El usuario indica el símbolo (por defecto, Apple) y cuántas sesiones adelante quiere (por defecto, 1); la voz puede decir "Apple", "Microsoft",
"Amazon" o "Google" (`ALIAS_SIMBOLOS` en el `router.py`).

Como en el modelo 01, la pregunta honesta no es qué error tiene el modelo sino **si mejora a repetir el último precio** (persistencia). La hipótesis de los mercados
eficientes en su forma débil [Fama1970] sostiene que el historial de precios no permite predecir los siguientes, y los retornos de los activos
muestran regularidades conocidas (colas pesadas, casi nula autocorrelación del signo, agrupamiento de la volatilidad [Cont2001]). El dataset termina el 7 de febrero de 2018: "mañana" es la sesión siguiente (8 de febrero) y la API declara que
no conoce el precio actual.

## 2. Entendimiento de los datos (0.5 pts)

Se usa el recorte de cuatro símbolos de *all_stocks_5yr.csv* [Nugent2018sp500] (el archivo completo tiene 505 símbolos y 619 040 filas y no se versiona): 5 036 filas, 1 259 sesiones por símbolo del 8 de febrero de 2013 al 7 de febrero de 2018. No hay nulos
ni filas duplicadas (fecha, símbolo), los cuatro símbolos comparten exactamente las mismas fechas y en todas las filas mínimo ≤ apertura, cierre ≤ máximo. Los intervalos entre sesiones son de 1 día natural (986 veces),
2 (11), 3 (227, fines de semana) y 4 (34, feriados con fin de semana); es decir, **hay un registro por día hábil, no por día natural**.

| Variable               | Tipo     | Uso                                                                                             |
|------------------------|----------|-------------------------------------------------------------------------------------------------|
| `close`                | objetivo | Cierre diario (USD); media: AAPL 109, MSFT 51, AMZN 577, GOOGL 682 (escalas muy distintas)       |
| `date`                 | fecha    | Ordena la serie                                                                                 |
| `Name`                 | símbolo  | Una serie por símbolo (AAPL, MSFT, AMZN, GOOGL)                                                   |
| `open`, `high`, `low`, `volume` | numéricas | **Excluidas**: no se pueden predecir para encadenar la predicción recursiva y no se conocen al consultar |

**Variables construidas** (`serie.py`, el mismo cálculo del modelo 01 porque cada modelo es una carpeta aislada): retornos logarítmicos de 1 a 5, 7 y 30 días, distancia a la media móvil de 7 y 30 días y desviación de los retornos de 7 y 30 días,
todos calculados **solo con cierres pasados de cada símbolo**. Como son retornos y no precios, no dependen de la escala y **un único modelo se comparte entre los cuatro símbolos**; el símbolo solo determina el precio de partida y la volatilidad del intervalo.
El objetivo es el retorno logarítmico de la sesión siguiente.

## 3. Exploración de los datos (0.5 pts)

- `figuras/series_normalizadas.png`: los cuatro precios en base 100 suben de forma sostenida en todo el período (AAPL ×2.35, MSFT ×3.25, AMZN ×5.41, GOOGL ×2.68) y las series se mueven juntas. **La prueba (desde el 8-feb-2017) es un tramo alcista y, salvo el final, tranquilo** (AAPL +21 %, MSFT +41 %, AMZN +73 %, GOOGL +27 %); termina en la corrección de febrero de 2018, y en su segunda mitad la volatilidad sube (desviación diaria por mitades de la prueba: AAPL 1.12 % → 1.25 %, MSFT 0.82 % → 1.21 %, AMZN 1.01 % → 1.64 %, GOOGL 0.97 % → 1.20 %).
- `figuras/retornos_por_simbolo.png`: el retorno diario tiene media 0.07–0.13 % y desviación 1.4–1.8 % (AAPL 1.46 %, MSFT 1.42 %, AMZN 1.81 %, GOOGL 1.37 %). Las colas son pesadas: curtosis en exceso 3.8 (AAPL), 11.1 (MSFT), 11.0 (AMZN) y 18.6 (GOOGL) [Cont2001].
- `figuras/correlacion_retornos.png`: los retornos de los símbolos están **correlacionados** (0.29 a 0.55; media de los pares 0.41). Esto importa para la evaluación: las filas de un mismo día no son independientes, por lo que la validación cruzada agrupa los cuatro símbolos de cada día y el bootstrap remuestrea días completos.
- `figuras/autocorrelacion.png`: la autocorrelación media de los retornos (rezagos 1 a 5: 0.022, −0.010, −0.026, −0.028, −0.014) queda dentro de la cota del 95 % (±0.055), mientras que la de los retornos al cuadrado es positiva (0.054, 0.021, 0.033, 0.031, 0.023): hay algo de agrupamiento de volatilidad, aunque mucho menor que en el Bitcoin.
- El precio sube en el 52.6 % de las sesiones. La volatilidad diaria fue mayor en el entrenamiento que en la prueba (AAPL 1.52 % → 1.19 %, MSFT 1.50 % → 1.03 %, AMZN 1.91 % → 1.36 %, GOOGL 1.43 % → 1.09 %).

## 4. Modelo (2 pts)

**Partición temporal.** La prueba son las **últimas 252 sesiones** (del 8-feb-2017 al 7-feb-2018); el entrenamiento, 3 904 filas (símbolo, día) hasta el 7-feb-2017 (cada fila usa los cierres hasta ese día y su objetivo es el retorno del día siguiente; el último origen de entrenamiento es el 6-feb-2017). Los cuatro símbolos se cortan en la misma fecha.
Ningún objetivo de entrenamiento cae en la prueba, y una prueba automática infla diez veces los cierres de la prueba y exige que la selección y el ajuste no cambien. Se sigue la práctica habitual de evaluar series con datos posteriores al entrenamiento [Hyndman2021].

**Candidatos.** Dos líneas base (que **no pueden ser elegidas**) y tres modelos de aprendizaje automático:

| Candidato                          | Qué hace                                                                              |
|------------------------------------|---------------------------------------------------------------------------------------|
| Persistencia (línea base)          | Retorno predicho = 0: mañana = hoy                                                      |
| Deriva (línea base)                | Retorno predicho = promedio del entrenamiento (0.086 % diario)                          |
| Ridge                              | Regresión lineal con penalización L2 [Hoerl1970], variables estandarizadas; α con la validación cruzada leave-one-out interna de `RidgeCV` (no temporal, pero solo ve el entrenamiento) |
| Random Forest                      | 200 árboles de profundidad ≤ 6 y ≥ 50 filas por hoja [Breiman2001]                       |
| Gradient Boosting                  | Histogramas, 100 iteraciones, tasa 0.03, profundidad ≤ 3 [Friedman2001; Pedregosa2011]   |

**Selección.** Solo con el entrenamiento, con validación cruzada de **ventana creciente por fechas** (5 pliegues, los cuatro símbolos de un día siempre en el mismo bloque) sobre el RMSE del retorno diario [Bergmeir2012]. El conjunto de prueba se usó una vez, para medir.

**Predicción a varias sesiones y fechas.** Se predice el retorno de una sesión, se calcula el cierre, se agrega al historial y se repite hasta el horizonte pedido (máximo 7). La fecha del resultado cuenta **días hábiles** (lunes a viernes; no se descuentan los feriados de la bolsa).
El **intervalo del 95 %** es cierre × exp(± 1.96 · σ · √sesiones), con la desviación σ del retorno diario **de cada símbolo** en el entrenamiento (AAPL 1.52 %, MSFT 1.52 %, AMZN 1.92 %, GOOGL 1.44 %) para la evaluación, y la calculada con todos los días (AAPL 1.46 %, MSFT 1.43 %, AMZN 1.82 %, GOOGL 1.38 %) para la API.

## 5. Evaluación (1 pt)

**Validación cruzada en el entrenamiento** (RMSE del retorno diario, media ± desviación entre pliegues):

| Modelo                      | RMSE              |
|-----------------------------|-------------------|
| Persistencia (línea base)   | 0.01587 ± 0.00236 |
| Deriva (línea base)         | 0.01586 ± 0.00237 |
| **Ridge (elegido)**         | **0.01586 ± 0.00238** |
| Random Forest               | 0.01594 ± 0.00243 |
| Gradient Boosting           | 0.01597 ± 0.00248 |

Ningún modelo mejora a las líneas base de forma apreciable: las diferencias (≈ 0.0001) son decenas de veces menores que la variación entre pliegues (≈ 0.002), y los modelos de árboles tampoco mejoran. El Ridge, el elegido entre los de aprendizaje automático, **coincide con la deriva**: ajustado solo con el entrenamiento elige α ≈ 5.6·10³ y sus coeficientes son ≈ 0 (el mayor, 2.4·10⁻⁴ en unidades estandarizadas); el modelo final, con todos los días, elige α = 10⁴ (coeficiente mayor 1.4·10⁻⁴). En ambos casos se reduce a su intercepto.

**Prueba, 246 orígenes por horizonte y símbolo** (cada sesión desde el 7-feb-2017, último día de entrenamiento, hasta el 29-ene-2018, con dato real a siete sesiones). Como los precios de los símbolos tienen escalas muy distintas, el error se mide como **proporción del precio real** y se junta en los cuatro símbolos. Habilidad = 1 − RMSE(modelo)/RMSE(referencia); su IC 95 %
sale de un bootstrap de bloques de 14 días con envoltura circular (adaptación del de bloques [Kunsch1989]) que remuestrea **días completos** (con los cuatro símbolos) por la correlación entre ellos.

| Sesiones | Error relativo modelo / persistencia / deriva (RMSE %) | Habilidad frente a la persistencia (IC 95 %) | Habilidad frente a la deriva (IC 95 %) |
|----------|---------------------------------------------------------|-----------------------------------------------|-----------------------------------------|
| 1        | 1.081 / 1.088 / 1.080                                   | +0.59 % (−0.10 %, +1.32 %)                    | −0.11 % (−0.45 %, +0.16 %)              |
| 3        | 1.893 / 1.926 / 1.887                                   | +1.71 % (−0.05 %, +3.46 %)                    | −0.30 % (−1.10 %, +0.38 %)              |
| 7        | 3.026 / 3.138 / 3.030                                   | +3.55 % (+0.01 %, +6.99 %)                    | +0.12 % (−0.59 %, +0.92 %)              |

- La mejora sobre la persistencia es pequeña y **no es robustamente distinguible de 0**: a 7 sesiones el límite inferior del IC es +0.01 %, pero cambia de signo según la semilla y el tamaño del bloque del bootstrap (con otras semillas queda por debajo de 0 en ≈ 80 % de los casos), y a 1 y 3 sesiones el IC incluye el 0. Además, **la mejora se explica por la deriva**: la deriva sola obtiene +0.7 %, +2.0 % y +3.4 % frente a la persistencia, y frente a la deriva el modelo tiene habilidad nula en los tres horizontes (los IC incluyen el 0; un IC con 0 es ausencia de evidencia de mejora, no prueba de que no exista). Es decir, el modelo
  no muestra aprender nada de las variables; su ventaja sobre repetir el último precio es sumar el crecimiento medio del entrenamiento (0.086 % diario, ≈ 24 % anual compuesto), que en este tramo alcista ayuda.
- Por símbolo, la habilidad frente a la persistencia a 7 sesiones (entre paréntesis, lo que subió el precio en la prueba) fue AAPL +0.8 % (+21 %), GOOGL +2.9 % (+27 %), AMZN +4.5 % (+73 %) y MSFT +7.0 % (+41 %): orden parecido pero no igual, y con n = 4 solo ilustrativo; con 246 orígenes solapados por símbolo no se interpreta cada uno aislado.
- El **acierto de dirección** (54.7 %, 60.5 % y 64.3 % a 1, 3 y 7 sesiones) no supera al de predecir siempre "sube" (55.8 %, 62.0 % y 64.7 %).
- Los intervalos del 95 % cubren entre 98.3 % y 98.8 % de los cierres reales (98.8, 98.8 y 98.3 % a 1, 3 y 7 sesiones; por símbolo, entre 97.2 % y 100 %): son **conservadores** en este tramo, porque la volatilidad del entrenamiento fue entre 1.3 y 1.5 veces la de la prueba (no se comparó con otros métodos de intervalo).
  La cobertura no cambió entre mitades de la prueba a 1 sesión (98.8 % y 98.8 %) y bajó un poco en la segunda a 7 (99.4 % y 97.2 %).
- Errores en dólares a 1 sesión (RMSE del modelo / de la persistencia): AAPL 1.76 / 1.76, MSFT 0.73 / 0.74, AMZN 13.61 / 13.74, GOOGL 9.38 / 9.45.

**Despliegue.** Las métricas son las de la partición temporal; el modelo que sirve la API se reentrena con **todas** las sesiones y los cuatro símbolos (la deriva sube a 0.096 % diario) para partir del último cierre de cada símbolo.
Ejemplo (7-feb-2018): AAPL cerró en 159.54 y la API estima 159.79 USD para el 8-feb (rango 155.29–164.42) y 160.92 para el 16-feb. Estas cifras del modelo final no se pueden medir con datos independientes.

## 6. Conclusión (0.5 pts)

Para los cuatro valores y el período 2013–2018, el precio de cierre se comporta, para fines de predicción con su propio historial, como un **paseo aleatorio con deriva**: ninguna variable de rezago, media móvil o volatilidad, ni modelos de árboles, aportó nada medible sobre la deriva.
El modelo entregado es un Ridge que lo reconoce (coeficientes ≈ 0); su pronóstico es "el último cierre más el crecimiento medio" y lo que aporta, más que el punto, es el **rango del 95 %** específico de cada símbolo, que crece con la raíz del horizonte y fue conservador en la prueba (su utilidad frente a otras alternativas no se midió). La conclusión coincide con la del modelo 01 (Bitcoin), aunque aquí la serie es mucho menos volátil y el intervalo resulta conservador.

**Limitaciones.** (1) Los datos terminan el 7 de febrero de 2018; la API no conoce el precio actual y lo advierte siempre. (2) Solo cuatro símbolos de empresas tecnológicas muy grandes, que subieron mucho en el período (probable selección favorable; no se midió contra los 505 símbolos): la deriva estimada, ≈ 24 % anual compuesto, no es una expectativa razonable a futuro, y en un mercado bajista el modelo seguiría prediciendo subidas.
(3) Una sola ventana de prueba, alcista y de volatilidad baja salvo el tramo final (la corrección de febrero de 2018, que coincide con las fechas que predice la API); el intervalo supone volatilidad constante y resultó ancho en ella. (4) Los símbolos están correlacionados (0.29–0.55), por lo que los 246 orígenes por horizonte no son independientes ni entre símbolos ni entre días vecinos; los IC por bloques lo reflejan solo en parte.
(5) Las fechas cuentan solo días hábiles de lunes a viernes: no se descuentan feriados de la bolsa (en la ventana de la API, 8 al 16 de febrero de 2018, no hay ninguno). (6) Solo se usó el historial de cierres; el conjunto completo de 505 símbolos podría permitir un modelo con más datos, pero no se versiona y no cambia la advertencia de eficiencia débil [Fama1970].
(7) **No es una recomendación de inversión.**

## Referencias (en `docs_latex/referencias.bib`)

- **[Nugent2018sp500]** Usuario de Kaggle *camnugent* (actualizado a feb-2018). *S&P 500 stock data* (`all_stocks_5yr.csv`). Kaggle *(falta confirmar el nombre completo del autor)*. https://www.kaggle.com/camnugent/sandp500
- **[Fama1970]** Fama, E. F. (1970). Efficient capital markets: A review of theory and empirical work. *The Journal of Finance*, 25(2), 383–417. https://doi.org/10.1111/j.1540-6261.1970.tb00518.x
- **[Cont2001]** Cont, R. (2001). Empirical properties of asset returns: Stylized facts and statistical issues. *Quantitative Finance*, 1(2), 223–236. https://doi.org/10.1080/713665670
- **[Kunsch1989]** Künsch, H. R. (1989). The jackknife and the bootstrap for general stationary observations. *The Annals of Statistics*, 17(3), 1217–1241. https://doi.org/10.1214/aos/1176347265
- **[Bergmeir2012]** Bergmeir, C., & Benítez, J. M. (2012). On the use of cross-validation for time series predictor evaluation. *Information Sciences*. https://doi.org/10.1016/j.ins.2011.12.028 *(falta confirmar volumen y páginas)*
- **[Hyndman2021]** Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3.ª ed.). OTexts. https://otexts.com/fpp3/
- **[Hoerl1970]** Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression: Biased estimation for nonorthogonal problems. *Technometrics*, 12(1), 55–67.
- **[Breiman2001]** Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.
- **[Friedman2001]** Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *The Annals of Statistics*, 29(5), 1189–1232. https://doi.org/10.1214/aos/1013203451
- **[Pedregosa2011]** Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
