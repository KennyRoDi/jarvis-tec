"""Modelo 02 · Predicción del precio de un automóvil usado — MODELO DE REFERENCIA.

Ejecutar desde backend/:  python -m features.modelo_02_autos.train
Dataset: https://raw.githubusercontent.com/amankharwal/Website-data/master/car%20data.csv

El script sigue las 6 etapas de la rúbrica; la redacción de cada etapa va en analisis.md.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from core.entrenamiento import carpeta_figuras, guardar_figura, guardar_modelo, metricas_regresion

CARPETA = Path(__file__).parent
OBJETIVO = "selling_price"
NUMERICAS = ["year", "present_price", "kms_driven", "owner"]
CATEGORICAS = ["fuel_type", "seller_type", "transmission"]
SEMILLA = 42

# 1. Análisis del problema: estimar el precio de reventa (lakhs INR) a partir de las
#    características del vehículo. Problema de regresión supervisada. Ver analisis.md.


def cargar_datos() -> pd.DataFrame:
    df = pd.read_csv(CARPETA / "dataset.csv")
    df.columns = [c.lower() for c in df.columns]
    return df.drop(columns=["car_name"])  # ~100 nombres distintos en 301 filas: no generaliza


def entender(df: pd.DataFrame) -> None:
    """2. Entendimiento de los datos."""
    print(f"Filas: {len(df)}  Columnas: {df.shape[1]}")
    print("\nTipos:\n", df.dtypes)
    print("\nNulos por columna:\n", df.isna().sum())
    print("\nResumen estadístico:\n", df.describe().round(2))


def explorar(df: pd.DataFrame, figuras: Path) -> None:
    """3. Exploración de los datos."""
    sns.histplot(df[OBJETIVO], kde=True)
    plt.title("Distribución del precio de venta")
    guardar_figura(figuras, "distribucion_objetivo")

    sns.heatmap(df[NUMERICAS + [OBJETIVO]].corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlación entre variables numéricas")
    guardar_figura(figuras, "correlacion")

    sns.scatterplot(data=df, x="present_price", y=OBJETIVO, hue="fuel_type")
    plt.title("Precio actual vs. precio de venta")
    guardar_figura(figuras, "present_vs_selling")


def construir_modelo(estimador) -> Pipeline:
    """4. Modelo: preprocesamiento + estimador en un único Pipeline (lo mismo se usa en la API)."""
    preprocesador = ColumnTransformer([
        ("num", StandardScaler(), NUMERICAS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAS),
    ])
    return Pipeline([("preprocesador", preprocesador), ("modelo", estimador)])


def evaluar(pipeline: Pipeline, X_test, y_test, figuras: Path) -> dict:
    """5. Evaluación sobre el conjunto de prueba."""
    y_pred = pipeline.predict(X_test)
    plt.scatter(y_test, y_pred, alpha=0.7)
    limite = max(y_test.max(), y_pred.max())
    plt.plot([0, limite], [0, limite], "r--")
    plt.xlabel("Precio real")
    plt.ylabel("Precio predicho")
    plt.title("Real vs. predicho (prueba)")
    guardar_figura(figuras, "real_vs_predicho")
    return metricas_regresion(y_test, y_pred)


def main() -> None:
    figuras = carpeta_figuras(CARPETA)
    df = cargar_datos()
    entender(df)
    explorar(df, figuras)

    X, y = df[NUMERICAS + CATEGORICAS], df[OBJETIVO]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEMILLA)

    base = construir_modelo(LinearRegression()).fit(X_train, y_train)
    pipeline = construir_modelo(RandomForestRegressor(n_estimators=200, random_state=SEMILLA))
    cv_r2 = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="r2")
    pipeline.fit(X_train, y_train)

    metricas = evaluar(pipeline, X_test, y_test, figuras)
    metricas["cv_r2_media"] = round(float(cv_r2.mean()), 4)
    metricas["baseline_regresion_lineal"] = metricas_regresion(y_test, base.predict(X_test))

    guardar_modelo(CARPETA, pipeline, metricas, entrada_ejemplo=X_test.iloc[0].to_dict())
    print("\nMétricas:", metricas)
    # 6. Conclusión: ver analisis.md


if __name__ == "__main__":
    main()
