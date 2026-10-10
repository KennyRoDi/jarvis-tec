"""Transformador de fechas del modelo 09.

Vive en su propio módulo (y no en train.py) porque joblib guarda las clases por su ruta de importación:
una clase definida en `train.py` se guardaría como `__main__.X` y la API no podría cargar el modelo.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

INICIO = pd.Timestamp("2015-01-04")  # primera semana del dataset


class CaracteristicasFecha(BaseEstimator, TransformerMixin):
    """(region, tipo, fecha) -> (region, tipo, mes, semana, t).

    `t` son los años transcurridos desde INICIO, **truncados al último valor visto en el entrenamiento**: el
    modelo no extrapola la tendencia a fechas futuras; para ellas conserva la estacionalidad (mes y semana)
    sobre el último nivel de precios conocido.
    """

    def fit(self, X, y=None):
        self.t_max_ = float(self._anios(pd.to_datetime(X["fecha"])).max())
        return self

    @staticmethod
    def _anios(fechas: pd.Series) -> pd.Series:
        return (fechas - INICIO).dt.days / 365.25

    def transform(self, X):
        fechas = pd.to_datetime(X["fecha"])
        return pd.DataFrame({
            "region": X["region"].to_numpy(),
            "tipo": X["tipo"].to_numpy(),
            "mes": fechas.dt.month.to_numpy(),
            "semana": fechas.dt.isocalendar().week.astype(int).to_numpy(),
            "t": np.minimum(self._anios(fechas).to_numpy(), self.t_max_),
        }, index=X.index)
