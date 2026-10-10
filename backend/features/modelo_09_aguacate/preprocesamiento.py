"""Transformador de fechas del modelo 09.

Vive en su propio módulo (y no en train.py) porque joblib guarda las clases por su ruta de importación:
una clase definida en `train.py` se guardaría como `__main__.X` y la API no podría cargar el modelo.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

INICIO = pd.Timestamp("2015-01-04")  # primera semana del dataset


class CaracteristicasFecha(BaseEstimator, TransformerMixin):
    """(region, tipo, fecha) -> (region, tipo, mes, semana, t); `semana` va de 1 a 52 según el día del año.

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
            # Semana derivada del día del año (no la ISO): la ISO contradice al mes cerca de año nuevo
            # (p. ej. el 31-dic puede caer en la semana ISO 1 y producir saltos de hasta 0.26 USD en un día).
            "semana": np.minimum((fechas.dt.dayofyear - 1) // 7 + 1, 52).to_numpy(),  # 53 solo existe 1-2 días
            "t": np.minimum(self._anios(fechas).to_numpy(), self.t_max_),
        }, index=X.index)
