"""Regresor de 'razón al precio actual' del modelo 02.

Vive en su propio módulo (y no en train.py) porque joblib guarda las clases por su ruta de importación: una clase definida en
`train.py` se guardaría como `__main__.X` y la API no podría cargar el modelo.
"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone


class RazonAlPrecioActual(BaseEstimator, RegressorMixin):
    """Predice el precio de reventa como `precio de agencia actual × razón`, y aprende la **razón** (reventa / agencia).

    La razón es casi independiente de la escala del auto (en los datos va de 0.11 a 0.99 y nunca supera 1), así que un modelo de
    árboles puede estimar el precio de un auto más caro que cualquiera del entrenamiento: en niveles un árbol no extrapola.
    Con `log=True` el modelo aprende el logaritmo de la razón (y el precio nunca es negativo).
    """

    def __init__(self, modelo, log: bool = False):
        self.modelo = modelo
        self.log = log

    def fit(self, X, y):
        razon = np.asarray(y, dtype=float) / X["present_price"].to_numpy(dtype=float)
        self.modelo_ = clone(self.modelo).fit(X, np.log(razon) if self.log else razon)
        return self

    def predict(self, X):
        razon = self.modelo_.predict(X)
        return (np.exp(razon) if self.log else razon) * X["present_price"].to_numpy(dtype=float)
