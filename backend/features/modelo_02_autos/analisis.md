# Modelo 02 · Predicción del precio de un automóvil usado

<!-- Notas para LaTeX: cada sección corresponde a un criterio de la rúbrica. Las figuras están en
     figuras/*.png y las métricas exactas en metricas.json (se regeneran con train.py). Las referencias
     [Clave] están en docs_latex/referencias.bib. -->

## 1. Análisis del problema (0.5 pts)

Se plantea estimar el precio de reventa de un automóvil usado a partir de siete datos que su dueño conoce: año de fabricación, precio de agencia
actual de ese modelo, kilometraje, combustible, tipo de vendedor, transmisión y número de dueños anteriores. Es un problema de **regresión
supervisada** cuya variable objetivo es `Selling_Price`, en *lakhs* de rupias indias (1 lakh = 100 000 INR). Tiene una dificultad particular: hay
solo 299 autos distintos, y los precios van de 0.1 a 35 lakhs, con unos pocos autos muy caros que dominan los errores en términos absolutos.

El asistente JarvisTEC usa este modelo para responder a "¿cuánto vale mi carro?": el comando abre el formulario con los siete datos.

## 2. Entendimiento de los datos (0.5 pts)

El conjunto [CarDekho2018autos] contiene 301 autos y 9 columnas, sin valores nulos. Es una muestra pequeña de un solo mercado (India).

| Variable        | Tipo        | Descripción                                   |
|-----------------|-------------|-----------------------------------------------|
| `Car_Name`      | texto       | **Excluida**: unos 100 nombres distintos en 301 filas, no generaliza |
| `Year`          | numérica    | Año de fabricación (2003–2018)                |
| `Selling_Price` | objetivo    | Precio de venta (lakhs): mediana 3.5, media 4.6, máximo 35 |
| `Present_Price` | numérica    | Precio de agencia actual (lakhs): mediana 6.1, máximo 92.6 |
| `Kms_Driven`    | numérica    | Kilometraje (500 a 500 000 km)                |
| `Owner`         | numérica    | Dueños anteriores: 288 autos con 0, 10 con 1 y **uno con 3** |
| `Fuel_Type`     | categórica  | Petrol (239), Diesel (58) y **CNG (solo 2)**   |
| `Seller_Type`   | categórica  | Dealer (193), Individual (106)                |
| `Transmission`  | categórica  | Manual (260), Automatic (39)                  |

**Filas duplicadas.** Hay **2 autos repetidos exactamente**. Se descartan antes de dividir, para que una copia no caiga en el entrenamiento y la otra en la
prueba; quedan **299 autos** (el efecto es pequeño con solo 2 filas, pero es la regla del proyecto y no cuesta nada).

**Categorías con casi ningún ejemplo.** El gas natural (2 autos) y los autos con tres dueños (1 auto) no permiten aprender nada fiable; la API acepta esos valores,
pero **avisa** que la estimación es poco confiable.

**Una regularidad clave.** La razón entre el precio de reventa y el de agencia **nunca supera 1** en los datos: va de 0.11 a 0.99 con mediana 0.65, y
depende sobre todo de la antigüedad (la mediana es de 0.22 en los autos de 2003 y de 0.91 en los de 2017). Esto sugiere modelar la razón y no el precio (sección 4).

## 3. Exploración de los datos (0.5 pts)

- `figuras/distribucion_objetivo.png`: el precio de venta tiene asimetría positiva marcada (2.5): la mayoría de los autos cuesta poco y unos pocos llegan a precios muy altos.
- `figuras/correlacion.png`: `present_price` es, con mucha ventaja, la variable más asociada al precio (r = 0.88). El año tiene r = 0.23, el kilometraje 0.03 y los dueños −0.09. El kilometraje
  casi no se asocia al precio por sí solo porque está mezclado con la antigüedad (correlación de −0.53 con el año).
- `figuras/present_vs_selling.png`: la relación es aproximadamente lineal, pero con mucha más dispersión en los autos caros. Los diésel tienen una mediana de reventa de 7.6 lakhs y los de gasolina de 2.65, porque
  también son modelos con precio de agencia más alto (10.4 frente a 4.6 de mediana).
- `figuras/razon_por_anio.png`: **la razón de reventa sube de forma regular con el año** (mediana de 0.22 en 2003, 0.49 en 2012, 0.72 en 2015 y 0.91 en 2017). Casi no depende del combustible, la transmisión ni el tipo de vendedor
  (medianas de 0.62 a 0.67, salvo el gas natural con solo 2 autos).
- **Valores extremos.** Un auto tiene precio de agencia de 92.6 lakhs (un Land Cruiser) y 9 superan los 30 lakhs; 2 tienen más de 200 000 km (uno de ellos, con 500 000 km, es un scooter de 0.52 lakhs de agencia). Son valores posibles y se conservaron, pero
  son los que ponen a prueba a un modelo que no extrapola (sección 5).

## 4. Modelo (2 pts)

**Partición.** 80 % para entrenamiento (239 autos) y 20 % para prueba (60), con `random_state = 42`. Con tan pocos datos la selección se hace **solo con el entrenamiento**, mediante validación cruzada repetida
(5 particiones × 3 repeticiones) y el RMSE en lakhs; la prueba se usa una sola vez [Pedregosa2011].

**Candidatos.** Cada uno es un `Pipeline` que estandariza las numéricas y codifica con *one-hot* las categóricas, de modo que la API aplica exactamente el mismo tratamiento. Hay dos líneas base (que no pueden ser elegidas) y seis modelos:

| Candidato                           | Qué predice                                                                                  |
|-------------------------------------|----------------------------------------------------------------------------------------------|
| Precio medio (línea base)           | El precio medio del entrenamiento                                                            |
| Regla de depreciación (línea base)  | Precio de agencia × la razón mediana del entrenamiento (0.65): la regla "vale dos tercios"   |
| Regresión lineal                    | El precio, con un modelo lineal                                                              |
| Random Forest                       | El precio, con 200 árboles [Breiman2001]                                                     |
| Gradient Boosting                   | El precio, con el ensamble secuencial de histogramas [Friedman2001]                          |
| Random Forest (razón)               | **La razón reventa / agencia**; el precio es la razón por el precio de agencia               |
| Gradient Boosting (razón)           | Ídem con boosting                                                                            |
| Regresión lineal (log-razón)        | El logaritmo de la razón; el precio nunca es negativo                                        |

**Por qué la razón.** Un árbol de decisión es constante por tramos [Hastie2009]: nunca predice por encima del mayor precio que vio. Pero el precio de agencia crece sin límite en el uso real (hay autos de 90 lakhs).
La razón, en cambio, es una cantidad acotada, casi independiente de la escala del auto, y la reventa se obtiene multiplicándola por el precio de agencia. Este cambio no es un ajuste ad hoc de hiperparámetros:
cambia la **pregunta** que aprende el modelo, y se evaluó con la misma validación cruzada que los demás.

**Reentrenamiento final.** Las métricas son las de la partición 80/20, pero con solo 299 autos el modelo que sirve la API se **reentrena con todos** para aprovecharlos.

## 5. Evaluación (1 pt)

**Validación cruzada en el entrenamiento** (RMSE en lakhs, media ± desviación entre pliegues, y R²):

| Modelo                               | RMSE            | R²    |
|--------------------------------------|-----------------|-------|
| Precio medio (línea base)            | 4.89 ± 0.84     | −0.01 |
| Regla de depreciación (línea base)   | 2.36 ± 0.86     | 0.754 |
| Regresión lineal                     | 1.73 ± 0.40     | 0.872 |
| Random Forest                        | 1.51 ± 0.89     | 0.892 |
| Gradient Boosting                    | 2.13 ± 0.86     | 0.807 |
| **Random Forest (razón)**            | **0.79 ± 0.18** | 0.971 |
| Gradient Boosting (razón)            | 0.82 ± 0.17     | 0.970 |
| Regresión lineal (log-razón)         | 1.16 ± 0.75     | 0.937 |

Modelar la razón **reduce a casi la mitad el error** del bosque en niveles (0.79 frente a 1.51) y, sobre todo, **estabiliza** el resultado: la desviación entre pliegues baja de 0.89 a 0.18, porque en niveles el error depende de
que un auto muy caro caiga en el pliegue de validación. Los dos modelos de razón con árboles (0.79 y 0.82) no se distinguen entre sí; se eligió el Random Forest por tener el menor valor.

**Conjunto de prueba (60 autos):**

| Modelo                         | R²     | MAE (lakhs) | RMSE (lakhs) |
|--------------------------------|--------|-------------|--------------|
| Precio medio (línea base)      | −0.000 | 3.33        | 5.08         |
| Regla de depreciación (línea base) | 0.555 | 1.81        | 3.39         |
| Random Forest en niveles       | 0.501  | 1.50        | 3.59         |
| Regresión lineal               | 0.753  | 1.47        | 2.52         |
| **Random Forest (razón), elegido** | **0.954** | **0.70** | **1.09**  |

Con intervalos de confianza del 95 % por bootstrap: RMSE 0.79 a 1.34, MAE 0.50 a 0.91 y R² 0.87 a 0.98. La amplitud del intervalo del R² (0.11) recuerda que son solo 60 autos y que **solo uno** de ellos supera los 15 lakhs de reventa.
En `figuras/real_vs_predicho.png` los autos de 8 a 13 lakhs quedan casi todos por debajo de la diagonal: el modelo los subestima un poco. El Random Forest en niveles (0.501) quedó muy por debajo de su R² de validación cruzada (0.89): no es un fallo
de la prueba sino de ese modelo, que no extrapola (ver abajo).

**Corrección de un resultado anterior.** La primera versión de este modelo publicaba un R² de prueba de 0.962 con el Random Forest en niveles. Ese número dependía de la partición (se calculó con los duplicados y otra partición de las filas): con los duplicados quitados el mismo modelo saca
0.50 en la prueba y su validación cruzada es de 0.89 con una desviación enorme. La cifra honesta del bosque en niveles es la de la validación cruzada, y se **reemplazó por el modelo de razón**, mejor y mucho más estable.

**Experimento de extrapolación.** Se entrenó solo con los 271 autos de precio de agencia menor o igual a 15 lakhs y se probó con los 28 más caros:

| Modelo               | RMSE (lakhs) | Sesgo (predicho − real) |
|----------------------|--------------|--------------------------|
| Random Forest (niveles) | 8.98      | −5.80                    |
| Random Forest (razón)   | 2.05      | +0.30                    |

El bosque en niveles **subestima en promedio casi 6 lakhs** a los autos caros, porque no puede predecir más allá de lo que vio; el de razón casi no tiene sesgo. Es la razón por la que la API puede estimar un auto de 90 lakhs de agencia con sentido.

## 6. Conclusión (0.5 pts)

Con siete datos básicos del auto se puede estimar su precio de reventa con un error medio de unos 0.7 lakhs (alrededor de 70 000 rupias) y un R² de validación cruzada de 0.97, frente a un error de 3.3 lakhs al predecir el precio medio y de 1.8 al aplicar la regla "vale dos tercios".
La información que más pesa es el precio de agencia, y el año explica casi toda la depreciación.

El hallazgo principal es de modelado: **cambiar el objetivo de precio a razón** (reventa / agencia) casi duplica la precisión, estabiliza el resultado y permite estimar autos más caros que cualquiera del entrenamiento. El hallazgo metodológico es que un R² de prueba alto puede ser un
accidente de la partición: se corrigió el 0.962 anterior comparando todos los candidatos con la misma validación cruzada.

**Limitaciones.** (1) Son 299 autos de un solo mercado (India, hasta 2018): no sirve para otros países ni para autos nuevos o eléctricos, y el año de fabricación se compara con autos de 2003 a 2018. (2) Los autos de gas natural (2) y con tres dueños (1) casi no tienen ejemplos: la API avisa que la
estimación es poco confiable. (3) La API avisa también cuando el año, el kilometraje o el precio de agencia están fuera del rango del entrenamiento. (4) Con 60 autos de prueba, y solo uno de más de 15 lakhs, las métricas son ruidosas: los intervalos
son amplios y las diferencias entre Random Forest y Gradient Boosting de razón no son concluyentes. (5) El precio de agencia "actual" lo da el usuario y no se verifica. (6) El modelo subestima algo los autos de 8 a 13 lakhs.

## Referencias (en `docs_latex/referencias.bib`)

- **[CarDekho2018autos]** Conjunto de datos *car data.csv* (autos usados, CarDekho). Copia en GitHub: https://raw.githubusercontent.com/amankharwal/Website-data/master/car%20data.csv *(falta confirmar el autor y el año del conjunto original en Kaggle)*
- **[Breiman2001]** Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32. https://doi.org/10.1023/A:1010933404324
- **[Friedman2001]** Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *The Annals of Statistics*, 29(5), 1189–1232. https://doi.org/10.1214/aos/1013203451
- **[Hastie2009]** Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2.ª ed.). Springer. https://doi.org/10.1007/978-0-387-84858-7
- **[Pedregosa2011]** Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
