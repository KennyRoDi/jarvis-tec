# Modelo 02 · Predicción del precio de un automóvil usado

<!-- Notas para LaTeX: cada sección corresponde a un criterio de la rúbrica. Las figuras están en
     figuras/*.png y las métricas exactas en metricas.json (se regeneran con train.py). -->

## 1. Análisis del problema (0.5 pts)

Se plantea estimar el precio de reventa de un automóvil usado a partir de sus características
(año, precio de agencia actual, kilometraje, combustible, tipo de vendedor, transmisión y número de
dueños previos). Se trata de un problema de **regresión supervisada**, cuya variable objetivo es
`Selling_Price`, expresada en *lakhs* de rupias indias (1 lakh = 100 000 INR).

## 2. Entendimiento de los datos (0.5 pts)

El conjunto contiene 301 registros y 9 columnas, sin valores nulos. Se descarta `Car_Name`, ya que
presenta cerca de 100 valores distintos en 301 filas y no aporta capacidad de generalización.

| Variable        | Tipo        | Descripción                                   |
|-----------------|-------------|-----------------------------------------------|
| `Year`          | numérica    | Año de fabricación (2003–2018)                |
| `Present_Price` | numérica    | Precio de agencia actual (lakhs)              |
| `Kms_Driven`    | numérica    | Kilometraje recorrido                         |
| `Owner`         | numérica    | Dueños anteriores (0–3)                       |
| `Fuel_Type`     | categórica  | Petrol, Diesel, CNG                           |
| `Seller_Type`   | categórica  | Dealer, Individual                            |
| `Transmission`  | categórica  | Manual, Automatic                             |
| `Selling_Price` | objetivo    | Precio de venta (lakhs), media 4.66, máx. 35  |

## 3. Exploración de los datos (0.5 pts)

- `figuras/distribucion_objetivo.png`: la variable objetivo presenta asimetría positiva marcada.
- `figuras/correlacion.png`: `Present_Price` es la variable con mayor correlación con el objetivo.
- `figuras/present_vs_selling.png`: relación aproximadamente lineal, con mayor precio en vehículos diésel.

_TODO: ampliar la interpretación de cada figura._

## 4. Modelo (2 pts)

Se construye un `Pipeline` de scikit-learn que estandariza las variables numéricas, codifica las
categóricas mediante *one-hot encoding* y entrena un **Random Forest Regressor** (200 árboles). Como
línea base se entrena una **regresión lineal** con el mismo preprocesamiento. Se reserva el 20 % de los
datos para prueba (`random_state = 42`) y se aplica validación cruzada de 5 particiones sobre el
conjunto de entrenamiento.

_TODO: justificar la elección de Random Forest con referencias bibliográficas._

## 5. Evaluación (1 pt)

| Modelo                     | R²     | MAE    | RMSE   |
|----------------------------|--------|--------|--------|
| Regresión lineal (base)    | 0.849  | 1.216  | 1.865  |
| Random Forest              | 0.962  | 0.612  | 0.935  |

R² medio en validación cruzada (Random Forest): 0.885. Ver `figuras/real_vs_predicho.png`.

## 6. Conclusión (0.5 pts)

_TODO: redactar. Puntos sugeridos: mejora del Random Forest frente a la línea base, diferencia entre
R² de prueba y de validación cruzada (tamaño reducido del conjunto), limitaciones del dataset
(mercado indio, 301 registros)._
