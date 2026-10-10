# Modelo 08 · Predicción del porcentaje de grasa corporal

<!-- Notas para LaTeX: cada sección corresponde a un criterio de la rúbrica. Las figuras están en
     figuras/*.png y las métricas exactas en metricas.json (se regeneran con train.py). Las referencias
     [Clave] van a docs_latex/referencias.bib (ver la lista al final). -->

## 1. Análisis del problema (0.5 pts)

El porcentaje de grasa corporal es un indicador de salud más informativo que el peso o el índice de masa
corporal, pero su medición de referencia, el pesaje hidrostático (que estima la densidad corporal), exige
equipo especializado [Siri1956]. Se plantea estimarlo a partir de medidas que cualquier persona puede tomar con
una cinta métrica y una balanza: edad, peso, estatura y diez circunferencias corporales. Se trata de un
problema de **regresión supervisada** cuya variable objetivo es `BodyFat`, expresada en porcentaje.

El asistente JarvisTEC utiliza este modelo para responder consultas como "¿qué porcentaje de grasa corporal
tengo?" a partir de los datos que el usuario proporciona en el formulario.

## 2. Entendimiento de los datos (0.5 pts)

El conjunto de datos reúne 252 hombres adultos con mediciones antropométricas [Penrose1985; Johnson1996]. No
presenta valores nulos. Todas las variables son numéricas.

| Variable (original)      | Descripción                          | Unidad original | Unidad en el modelo |
|--------------------------|--------------------------------------|-----------------|---------------------|
| `BodyFat`                | **Objetivo**: grasa corporal         | %               | %                   |
| `Density`                | Densidad corporal                    | g/cm³           | **excluida**        |
| `Age`                    | Edad                                 | años            | años                |
| `Weight`                 | Peso                                 | libras          | kg (`weight_kg`)    |
| `Height`                 | Estatura                             | pulgadas        | cm (`height_cm`)    |
| `Neck` … `Wrist` (10)    | Circunferencias (cuello, pecho, abdomen, cadera, muslo, rodilla, tobillo, bíceps, antebrazo, muñeca) | cm | cm (`*_cm`) |

Se convirtieron peso y estatura al sistema métrico para que el formulario de la aplicación utilice las
unidades habituales del usuario.

**Exclusión de `Density` (fuga de información).** `BodyFat` no es una medición independiente: se calcula a
partir de la densidad corporal mediante la ecuación de Siri [Siri1956]. La correlación entre ambas es de
−0.99. Un modelo lineal que incluye `Density` alcanza un R² de 0.994 en el conjunto de prueba, frente a 0.557
sin ella. Esa precisión no es utilizable: medir la densidad exige el mismo pesaje hidrostático que se pretende
evitar. Por ello se descartó la variable.

## 3. Exploración de los datos (0.5 pts)

- `figuras/distribucion_objetivo.png`: la grasa corporal tiene una distribución aproximadamente simétrica con
  media de 19.2 % y desviación de 8.3 puntos.
- `figuras/correlacion.png`: la variable más asociada al objetivo es la circunferencia del abdomen (r = 0.81),
  seguida del pecho (0.70), la cadera (0.61) y el peso (0.60). La estatura casi no se relaciona (−0.03). Las
  circunferencias están fuertemente correlacionadas entre sí (por ejemplo, peso y cadera, r = 0.94), por lo que
  existe multicolinealidad, un argumento para considerar regularización.
- `figuras/abdomen_vs_grasa.png`: la relación abdomen–grasa es aproximadamente lineal.

**Datos atípicos.** Se descartaron únicamente dos registros físicamente imposibles: uno con 0 % de grasa
(fila 181) y otro con una estatura de 75 cm y un índice de masa corporal de 165 (fila 41, probablemente un
error de digitación). Se conservaron los demás extremos por ser valores posibles, incluida una persona de
165 kg con abdomen de 148 cm (fila 38), cuyo efecto se analiza en la sección 5. Quedan **250 registros**.

## 4. Modelo (2 pts)

Se dividieron los datos en 80 % para entrenamiento (200 filas) y 20 % para prueba (50 filas), con
`random_state = 42`. Se compararon cinco candidatos, cada uno como un `Pipeline` de scikit-learn que incluye su
preprocesamiento (estandarización cuando corresponde), de modo que la API aplique exactamente el mismo
tratamiento que el entrenamiento:

| Candidato                       | Justificación                                                                   |
|---------------------------------|---------------------------------------------------------------------------------|
| Media (línea base)              | Referencia mínima: cualquier modelo útil debe superarla                          |
| Regresión lineal                | Modelo simple e interpretable; el problema es aproximadamente lineal            |
| Ridge                           | Penalización L2 que estabiliza los coeficientes ante la multicolinealidad [Hoerl1970] |
| Lasso                           | Penalización L1 que además selecciona variables al anular coeficientes [Tibshirani1996] |
| Random Forest                   | Captura relaciones no lineales sin supuestos sobre la forma funcional [Breiman2001] |

La selección se realizó **solo con el conjunto de entrenamiento**, mediante validación cruzada repetida
(5 particiones × 3 repeticiones), tomando el menor RMSE promedio. El conjunto de prueba se utilizó una única
vez, con el modelo elegido. La intensidad de la penalización de Ridge y Lasso se ajustó por validación cruzada
interna.

El modelo elegido es **Lasso** (α = 0.21). Conserva cinco variables con coeficientes distintos de cero y anula
las ocho restantes. Con las variables estandarizadas, el abdomen domina con un coeficiente de +7.5, seguido de
la muñeca (−1.4), la edad (+0.7) y la estatura (−0.7). Que la estatura y la muñeca tengan signo negativo es
coherente con que, a igual circunferencia abdominal, una persona más alta o de mayor estructura ósea acumula
proporcionalmente menos grasa.

## 5. Evaluación (1 pt)

**Validación cruzada en entrenamiento** (media ± desviación del RMSE entre particiones):

| Modelo                | RMSE (puntos de %) | R²     |
|-----------------------|--------------------|--------|
| Media (línea base)    | 8.43 ± 0.81        | −0.051 |
| Regresión lineal      | 4.53 ± 0.46        | 0.689  |
| Ridge                 | 4.54 ± 0.43        | 0.689  |
| **Lasso**             | **4.50 ± 0.39**    | 0.695  |
| Random Forest         | 4.78 ± 0.43        | 0.656  |

Las diferencias entre regresión lineal, Ridge y Lasso (0.03 puntos de RMSE) son mucho menores que la variación
entre particiones (≈ 0.4): **no son estadísticamente distinguibles**. Lasso se eligió por tener el menor RMSE
y por ofrecer un modelo más simple, no porque supere de forma demostrable a los otros dos. Random Forest fue
ligeramente peor, lo que sugiere que con 200 filas y una relación casi lineal la flexibilidad adicional no
compensa.

**Conjunto de prueba (50 filas):**

| Modelo             | R²     | MAE (puntos de %) | RMSE (puntos de %) |
|--------------------|--------|-------------------|--------------------|
| Media (línea base) | −0.006 | 6.50              | 7.68               |
| Lasso              | 0.557  | 3.92              | 5.09               |

El modelo reduce el error medio absoluto de 6.5 a 3.9 puntos porcentuales respecto de la línea base. Sin
embargo, el R² de prueba (0.557) es inferior al de validación cruzada (0.695). Se identificó la causa
principal en `figuras/real_vs_predicho.png`: la persona de 165 kg y abdomen de 148 cm (fila 38) cayó en el
conjunto de prueba y el modelo estimó 55.1 % cuando el valor real era 35.2 % (error de casi 20 puntos). En
entrenamiento el peso máximo fue 119 kg y el abdomen máximo 126 cm: el modelo lineal **extrapoló** fuera del
rango que había visto. Excluyendo ese único punto, el R² de prueba sube a 0.661, el MAE baja a 3.60 y el RMSE
a 4.29, valores coherentes con la validación cruzada. Esta cifra es un análisis de sensibilidad posterior y
**no** sustituye a la métrica oficial reportada arriba.

Dado que el conjunto de prueba tiene solo 50 filas, cada métrica es ruidosa y un solo registro puede moverla
de forma apreciable.

## 6. Conclusión (0.5 pts)

Con medidas corporales simples es posible estimar el porcentaje de grasa corporal con un error medio de
aproximadamente 4 puntos porcentuales, frente a 6.5 de la predicción por la media. La circunferencia
abdominal concentra casi toda la información útil: un modelo lineal regularizado que usa solo cinco variables
rinde igual que uno con las trece y mejor que un Random Forest.

El hallazgo metodológico más importante fue la fuga de información de `Density`: incluirla produce un R² de
0.99 que es engañoso, porque la variable ya contiene la respuesta.

**Limitaciones.** (1) El conjunto tiene solo 252 hombres adultos: el modelo **no es válido para mujeres ni
para menores**, y las estimaciones son orientativas, no diagnósticas. (2) El modelo extrapola mal con medidas
extremas, como mostró la fila 38; por eso la API compara cada entrada con el rango de entrenamiento y advierte
que el resultado es poco confiable cuando lo excede. (3) Con 50 filas de prueba, la incertidumbre de las
métricas es alta. Como trabajo futuro se propone ampliar la muestra, incluir mujeres y evaluar transformaciones
que reduzcan la extrapolación.

## Referencias (agregar a `docs_latex/referencias.bib`)

- **[Siri1956]** Siri, W. E. (1956). The gross composition of the body. *Advances in Biological and Medical Physics*, 4, 239–280.
- **[Penrose1985]** Penrose, K. W., Nelson, A. G., & Fisher, A. G. (1985). Generalized body composition prediction equation for men using simple measurement techniques. *Medicine & Science in Sports & Exercise*, 17(2), 189.
- **[Johnson1996]** Johnson, R. W. (1996). Fitting percentage of body fat to simple body measurements. *Journal of Statistics Education*, 4(1).
- **[Hoerl1970]** Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression: Biased estimation for nonorthogonal problems. *Technometrics*, 12(1), 55–67.
- **[Tibshirani1996]** Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. *Journal of the Royal Statistical Society: Series B*, 58(1), 267–288.
- **[Breiman2001]** Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.

> Las referencias están citadas de memoria y deben verificarse (autores, páginas, DOI) antes de incluirlas en el documento final.
