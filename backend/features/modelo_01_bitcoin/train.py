"""Modelo 01 · Predicción del precio del Bitcoin.

Ejecutar desde backend/:  python -m features.modelo_01_bitcoin.train
Dataset: https://www.kaggle.com/team-ai/bitcoin-price-prediction/version/1  →  guardar como features/modelo_01_bitcoin/dataset.csv

Sigue las 6 etapas de la rúbrica; la redacción va en analisis.md.
Referencia completa: features/modelo_02_autos/train.py
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from core.entrenamiento import carpeta_figuras, guardar_modelo, metricas_regresion

CARPETA = Path(__file__).parent
OBJETIVO = "Close"
NUMERICAS: list[str] = []  # TODO(Dev A)
CATEGORICAS: list[str] = []  # TODO(Dev A)
SEMILLA = 42

# 1. Análisis del problema: ver analisis.md


def cargar_datos() -> pd.DataFrame:
    return pd.read_csv(CARPETA / "dataset.csv")


def entender(df: pd.DataFrame) -> None:
    """2. Entendimiento de los datos."""
    print(f"Filas: {len(df)}  Columnas: {df.shape[1]}")
    print("\nTipos:\n", df.dtypes)
    print("\nNulos por columna:\n", df.isna().sum())
    print("\nResumen estadístico:\n", df.describe().round(2))


def explorar(df: pd.DataFrame, figuras: Path) -> None:
    """3. Exploración de los datos: guardar gráficos en figuras/ con guardar_figura()."""
    # TODO(Dev A)


def construir_modelo() -> Pipeline:
    """4. Modelo: Pipeline(preprocesamiento + estimador)."""
    raise NotImplementedError("TODO(Dev A): definir el pipeline (ver modelo_02_autos/train.py)")


def evaluar(pipeline: Pipeline, X_test, y_test, figuras: Path) -> dict:
    """5. Evaluación sobre el conjunto de prueba."""
    return metricas_regresion(y_test, pipeline.predict(X_test))


def main() -> None:
    figuras = carpeta_figuras(CARPETA)
    df = cargar_datos()
    entender(df)
    explorar(df, figuras)

    X, y = df[NUMERICAS + CATEGORICAS], df[OBJETIVO]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEMILLA)
    pipeline = construir_modelo().fit(X_train, y_train)

    metricas = evaluar(pipeline, X_test, y_test, figuras)
    guardar_modelo(CARPETA, pipeline, metricas, entrada_ejemplo=X_test.iloc[0].to_dict())
    print("\nMétricas:", metricas)
    # 6. Conclusión: ver analisis.md


if __name__ == "__main__":
    main()
