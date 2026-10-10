"""Pruebas del modelo 10 · S&P 500 (specs/api_rest_spec.md §3). Usan el modelo entrenado real."""
import re
import shutil
from datetime import date

import joblib
import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from scipy.stats import kurtosis

from core.modelos import REGISTRO, cargar_artefacto, leer_metricas
from features.modelo_10_sp500 import train
from features.modelo_10_sp500.router import (ALIAS_SIMBOLOS, CARPETA, MAX_DIAS, NOMBRES, SIMBOLOS, Entrada, dolares, fecha_destino,
                                             fecha_en_texto)
from features.modelo_10_sp500.serie import CARACTERISTICAS, VENTANA_MINIMA, caracteristicas, recursiva, ultima_fila
from features.modelo_10_sp500.train import (HORIZONTES, MAX_HORIZONTE, OBJETIVO, autocorrelacion, candidatos, cargar_datos, construir, entender,
                                            entrenar, es_linea_base, explorar, habilidad, intervalo_habilidad, origenes, particion_temporal,
                                            pliegues_por_fecha, retornos)
from features.modelo_10_sp500.train import SIMBOLOS as SIMBOLOS_TRAIN

URL = "/api/modelos/sp500/predecir"
entrenado = pytest.mark.skipif(not REGISTRO["sp500"].entrenado, reason="ejecutar features.modelo_10_sp500.train")


class Constante:
    """Pipeline falso: predice siempre el mismo retorno diario."""
    def __init__(self, retorno):
        self.retorno = retorno

    def predict(self, X):
        return np.full(len(X), self.retorno)


class Momento:
    """Pipeline falso: repite el retorno del día anterior (ret_1)."""
    def predict(self, X):
        return X["ret_1"].to_numpy()


@pytest.fixture(scope="module")
def reentrenado():
    """Todo el entrenamiento (selección, evaluación, bootstrap) en memoria, sin escribir nada."""
    return entrenar(cargar_datos(), figuras=None, imprimir=False)


def serie_sintetica(n=120, semilla=0):
    return 100 * np.exp(np.cumsum(np.random.default_rng(semilla).normal(0.001, 0.03, n)))


# --- Preparación de datos (no requiere el modelo entrenado) ---

def test_los_datos_cargados_son_cuatro_simbolos_con_las_mismas_1259_sesiones():
    df = cargar_datos()
    assert len(df) == 5036 and sorted(df.simbolo.unique()) == sorted(SIMBOLOS) and SIMBOLOS_TRAIN == list(SIMBOLOS)
    assert list(df.columns) == ["fecha", "apertura", "maximo", "minimo", OBJETIVO, "volumen", "simbolo"]
    assert df.fecha.min() == pd.Timestamp("2013-02-08") and df.fecha.max() == pd.Timestamp("2018-02-07")
    assert df.groupby("simbolo").size().to_dict() == {s: 1259 for s in SIMBOLOS}
    assert not df.duplicated(["fecha", "simbolo"]).any() and df.isna().sum().sum() == 0
    fechas = [g.fecha.tolist() for _, g in df.groupby("simbolo")]
    assert all(f == fechas[0] for f in fechas), "los cuatro símbolos comparten calendario"
    assert all(g.fecha.is_monotonic_increasing for _, g in df.groupby("simbolo"))


def test_el_dataset_es_el_recorte_de_los_cuatro_simbolos_y_los_precios_son_coherentes():
    df = cargar_datos()
    assert ((df.minimo <= df.cierre) & (df.cierre <= df.maximo) & (df.minimo <= df.apertura) & (df.apertura <= df.maximo)).all()
    ultimos = df.groupby("simbolo").cierre.last().round(2).to_dict()
    assert ultimos == {"AAPL": 159.54, "AMZN": 1416.78, "GOOGL": 1055.41, "MSFT": 89.61}


def test_las_sesiones_son_dias_habiles_sin_fines_de_semana():
    fechas = cargar_datos().fecha.drop_duplicates()
    assert (fechas.dt.dayofweek < 5).all()
    assert fechas.diff().dt.days.value_counts().to_dict() == {1.0: 986, 3.0: 227, 4.0: 34, 2.0: 11}


def test_el_volumen_no_es_variable_del_modelo():
    """No se puede predecir el volumen futuro, así que no podría encadenarse en la predicción recursiva."""
    assert not any("volumen" in c for c in CARACTERISTICAS)


# --- Características y predicción recursiva (mismo cálculo que el modelo 01) ---

def test_caracteristicas_calculadas_a_mano():
    c = np.array([100.0, 110.0, 99.0, 108.9, 120.0, 130.0, 125.0, 140.0] + [150.0] * 30)
    f = caracteristicas(c)
    t = 7
    assert f.loc[t, "ret_1"] == pytest.approx(np.log(140 / 125))
    assert f.loc[t, "ret_2"] == pytest.approx(np.log(125 / 130)) and f.loc[t, "ret_3"] == pytest.approx(np.log(130 / 120))
    assert f.loc[t, "ret_4"] == pytest.approx(np.log(120 / 108.9)) and f.loc[t, "ret_5"] == pytest.approx(np.log(108.9 / 99))
    assert f.loc[t, "ret_7"] == pytest.approx(np.log(140 / 100))
    assert f.loc[t, "dist_media_7"] == pytest.approx(np.log(140 / np.mean(c[1:8])))
    assert f.loc[t, "vol_7"] == pytest.approx(np.std(np.diff(np.log(c[:8])), ddof=1), abs=1e-12)
    assert f.loc[30, "ret_30"] == pytest.approx(np.log(c[30] / c[0])) and f.loc[30, "dist_media_30"] == pytest.approx(np.log(c[30] / np.mean(c[1:31])))


def test_ultima_fila_coincide_con_la_tabla_completa_en_todos_los_dias():
    c = serie_sintetica(200)
    f = caracteristicas(c)
    for t in range(VENTANA_MINIMA - 1, len(c)):
        assert ultima_fila(c[:t + 1]) == pytest.approx(f.loc[t].to_dict(), abs=1e-12)


def test_las_caracteristicas_no_miran_el_futuro_y_la_ventana_minima_es_exacta():
    c = serie_sintetica(150)
    c2 = c.copy()
    c2[100:] *= 7
    assert caracteristicas(c).iloc[:100].equals(caracteristicas(c2).iloc[:100])
    f = caracteristicas(c)
    assert f.iloc[:VENTANA_MINIMA - 1].isna().any(axis=1).all() and f.iloc[VENTANA_MINIMA - 1:].notna().all().all()
    with pytest.raises(ValueError, match=str(VENTANA_MINIMA)):
        ultima_fila(c[:VENTANA_MINIMA - 1])
    assert ultima_fila(c) == ultima_fila(c[-VENTANA_MINIMA:])


def test_recursiva_compone_realimenta_y_no_modifica_los_historiales():
    c = serie_sintetica(60)
    copia = c.copy()
    assert recursiva(Constante(0.01), [c], 5)[0] == pytest.approx(c[-1] * np.exp(0.01 * np.arange(1, 6)))
    r1 = np.log(c[-1] / c[-2])
    assert recursiva(Momento(), [c], 4)[0] == pytest.approx(c[-1] * np.exp(r1 * np.arange(1, 5))), "el segundo paso usa el retorno predicho"
    assert np.array_equal(c, copia)
    historiales = [serie_sintetica(70, s) for s in range(4)]
    lote = recursiva(Momento(), historiales, 6)
    for i, h in enumerate(historiales):
        assert lote[i] == pytest.approx(recursiva(Momento(), [h], 6)[0])
    assert recursiva(Momento(), historiales, 3) == pytest.approx(lote[:, :3])


# --- Partición, validación y selección ---

def test_el_objetivo_es_el_retorno_del_dia_siguiente_dentro_de_cada_simbolo():
    df = cargar_datos()
    X, y = construir(df)
    assert X.index.names == ["simbolo", "posicion"] and list(X.index) == list(y.index)
    assert sorted(X.index.get_level_values("simbolo").unique()) == sorted(SIMBOLOS)
    for s in SIMBOLOS:
        c = df[df.simbolo == s].cierre.to_numpy()
        posiciones = X.xs(s, level="simbolo").index
        assert posiciones[0] == VENTANA_MINIMA - 1 and posiciones[-1] == len(c) - 2
        for t in (30, 700, len(c) - 2):
            assert y[(s, t)] == pytest.approx(np.log(c[t + 1] / c[t]))
    assert X.notna().all().all() and y.notna().all()


def test_cada_simbolo_se_construye_con_sus_propios_cierres_sin_mezclarse():
    df = cargar_datos()
    X, _ = construir(df)
    s = "MSFT"
    esperado = caracteristicas(df[df.simbolo == s].cierre.to_numpy()).loc[500]
    assert X.loc[(s, 500)].to_dict() == pytest.approx(esperado.to_dict())


def test_la_particion_es_temporal_y_no_mira_el_futuro():
    df = cargar_datos()
    corte = particion_temporal(df)
    assert corte == int(1259 * 0.8) == 1007
    X, _ = construir(df)
    entrena, prueba = origenes(X, corte)
    posicion = X.index.get_level_values("posicion").to_numpy()
    assert posicion[entrena].max() == corte - 2 and posicion[prueba].min() == corte - 1 and not set(entrena) & set(prueba)
    assert len(entrena) == 3904 and len(prueba) == 4 * (1259 - 1 - (corte - 1))
    assert set(df[df.simbolo == "AAPL"].fecha.iloc[[corte]].astype(str)) == {"2017-02-08"}


def test_la_particion_exige_el_mismo_calendario_en_todos_los_simbolos():
    df = cargar_datos()
    roto = df.drop(df[(df.simbolo == "MSFT") & (df.fecha == df.fecha.iloc[100])].index)
    with pytest.raises(AssertionError, match="mismas fechas"):
        particion_temporal(roto)


def test_los_pliegues_agrupan_los_simbolos_de_cada_dia_y_crecen_sin_solaparse():
    X, _ = construir(cargar_datos())
    entrena, _ = origenes(X, particion_temporal(cargar_datos()))
    Xt = X.iloc[entrena]
    posicion = Xt.index.get_level_values("posicion").to_numpy()
    pliegues = pliegues_por_fecha(Xt, np.arange(len(Xt)))
    assert len(pliegues) == 5
    previo_fin = -1
    for entr, prueb in pliegues:
        assert posicion[entr].max() < posicion[prueb].min(), "se entrena con el pasado y se valida con el futuro"
        assert not set(posicion[entr]) & set(posicion[prueb]), "los cuatro símbolos de un día van juntos"
        assert posicion[prueb].min() > previo_fin and len(entr) % 4 == 0 and len(prueb) % 4 == 0
        previo_fin = posicion[prueb].max()


def test_candidatos_dos_lineas_base_y_tres_modelos():
    c = candidatos()
    assert [n for n in c if es_linea_base(n)] == ["persistencia (línea base)", "deriva (línea base)"]
    assert set(c) - {n for n in c if es_linea_base(n)} == {"ridge", "random forest", "gradient boosting"}
    assert c["gradient boosting"].named_steps["modelo"].early_stopping is False
    X = pd.DataFrame(np.random.default_rng(0).normal(size=(50, len(CARACTERISTICAS))), columns=CARACTERISTICAS)
    y = pd.Series(np.random.default_rng(1).normal(0.01, 0.02, 50))
    assert (c["persistencia (línea base)"].fit(X, y).predict(X) == 0).all()
    assert c["deriva (línea base)"].fit(X, y).predict(X) == pytest.approx(np.full(50, y.mean()))


def test_habilidad_y_su_bootstrap_en_casos_conocidos():
    e = np.random.default_rng(0).normal(size=(200, 4))
    assert habilidad(e, e) == 0 and habilidad(np.zeros_like(e), e) == 1 and habilidad(e / 2, e) == 0.5 and habilidad(e * 1.5, e) == -0.5
    assert intervalo_habilidad(e, e, 50) == [0.0, 0.0] and intervalo_habilidad(e / 2, e, 50) == [0.5, 0.5]
    ruido = np.random.default_rng(1).normal(size=(200, 4))
    ic = intervalo_habilidad(ruido, e, 200)
    assert ic[0] < 0 < ic[1] and intervalo_habilidad(ruido, e, 50) == intervalo_habilidad(ruido, e, 50)


def test_el_bootstrap_con_errores_constantes_por_simbolo_da_un_solo_valor():
    """Un símbolo con error 1 y otro con error 0 todos los días: cualquier remuestreo de días da la misma habilidad (1 - raíz de 0.5)."""
    e = np.zeros((100, 2))
    e[:, 0] = 1.0
    base = np.ones((100, 2))
    ic = intervalo_habilidad(e, base, 100)
    assert ic[0] == pytest.approx(1 - np.sqrt(0.5), abs=1e-4) and ic[1] == pytest.approx(1 - np.sqrt(0.5), abs=1e-4)


def test_autocorrelacion_en_casos_conocidos():
    alterna = np.tile([1.0, -1.0], 200)
    assert autocorrelacion(alterna, 2) == pytest.approx([-1.0, 1.0], abs=0.01)
    ruido = np.random.default_rng(0).normal(size=5000)
    assert np.abs(autocorrelacion(ruido, 10)).max() < 0.06
    assert autocorrelacion(ruido + 50, 10) == pytest.approx(autocorrelacion(ruido, 10)), "se resta la media antes de correlacionar"


def test_los_retornos_se_calculan_dentro_de_cada_simbolo():
    df = cargar_datos()
    r = retornos(df)
    primeros = df.groupby("simbolo").head(1).index
    assert r.loc[primeros].isna().all() and r.drop(primeros).notna().all(), "el primer día de cada símbolo no hereda el cierre del anterior"
    i = df.index[(df.simbolo == "MSFT")][10]
    assert r[i] == pytest.approx(np.log(df.cierre[i] / df.cierre[i - 1]))


def test_entender_y_explorar(capsys, tmp_path):
    df = cargar_datos()
    entender(df)
    salida = capsys.readouterr().out
    assert "Filas: 5036  Columnas: 7  Símbolos: 4  Fechas: 1259" in salida and "2013-02-08 a 2018-02-07" in salida
    assert "{1.0: 986, 2.0: 11, 3.0: 227, 4.0: 34}" in salida and "duplicados (fecha, símbolo): 0" in salida
    explorar(df, tmp_path, df.fecha[1007])
    assert sorted(p.name for p in tmp_path.glob("*.png")) == ["autocorrelacion.png", "correlacion_retornos.png", "retornos_por_simbolo.png", "series_normalizadas.png"]
    salida = capsys.readouterr().out
    assert "[0.022, -0.01, -0.026, -0.028, -0.014] | cota 95 % por símbolo: ±0.055" in salida
    assert "(rezagos 1-5): [0.054, 0.021, 0.033, 0.031, 0.023]" in salida
    assert "Correlación media entre pares de símbolos: 0.409" in salida and "Sesiones en que el precio sube: 52.6%" in salida


def test_las_cifras_de_exploracion_citadas_en_el_analisis():
    """analisis.md las cita: si los datos o el cálculo cambian, el texto debe actualizarse."""
    df = cargar_datos()
    r = retornos(df).dropna()
    por_simbolo = r.groupby(df.simbolo[r.index]).agg(["mean", "std"]).round(4)
    assert por_simbolo["mean"].to_dict() == {"AAPL": 0.0007, "AMZN": 0.0013, "GOOGL": 0.0008, "MSFT": 0.0009}
    assert por_simbolo["std"].to_dict() == {"AAPL": 0.0146, "AMZN": 0.0181, "GOOGL": 0.0137, "MSFT": 0.0142}
    assert round((r > 0).mean(), 3) == 0.526
    assert {s: round(g.cierre.iloc[-1] / g.cierre.iloc[0], 2) for s, g in df.groupby("simbolo")} == {"AAPL": 2.35, "AMZN": 5.41, "GOOGL": 2.68, "MSFT": 3.25}
    assert {s: round(g.cierre.iloc[-1] / g.cierre.iloc[1007] - 1, 2) for s, g in df.groupby("simbolo")} == {"AAPL": 0.21, "AMZN": 0.73, "GOOGL": 0.27, "MSFT": 0.41}
    por_simbolo_ret = retornos(df).to_frame("r").assign(fecha=df.fecha, simbolo=df.simbolo).pivot(index="fecha", columns="simbolo", values="r").dropna()
    assert {s: round(float(kurtosis(por_simbolo_ret[s])), 1) for s in SIMBOLOS} == {"AAPL": 3.8, "MSFT": 11.1, "AMZN": 11.0, "GOOGL": 18.6}
    corr = por_simbolo_ret.corr().to_numpy()[np.triu_indices(4, 1)]
    assert (corr.min().round(2), corr.max().round(2), corr.mean().round(3)) == (0.29, 0.55, 0.409)
    prueba = por_simbolo_ret.iloc[1006:]
    assert len(prueba) == 252
    por_mitad = {s: (round(prueba[s].iloc[:126].std(), 4), round(prueba[s].iloc[126:].std(), 4)) for s in SIMBOLOS}
    assert por_mitad == {"AAPL": (0.0112, 0.0125), "MSFT": (0.0082, 0.0121), "AMZN": (0.0101, 0.0164), "GOOGL": (0.0097, 0.012)}
    assert df[df.simbolo == "AAPL"].fecha.iloc[[1005, 1006, 1007, 1259 - 8]].dt.strftime("%Y-%m-%d").tolist() == ["2017-02-06", "2017-02-07", "2017-02-08", "2018-01-29"]


# --- Entrada y metadatos ---

def test_los_metadatos_del_modelo_registrado():
    info = REGISTRO["sp500"].info
    assert info["slug"] == "sp500" and info["tipo"] == "regresion" and info["unidad"] == "USD"
    assert info["comandos"] == ["precio de la accion", "bolsa", "sp500"] and REGISTRO["sp500"].entrada is Entrada
    assert info["nombre"] == "Predicción del precio de acciones del S&P 500"


def test_entrada_por_defecto_es_apple_un_dia_y_todos_los_campos_tienen_valor():
    assert Entrada().simbolo == "AAPL" and Entrada().dias_adelante == 1 and Entrada(dias_adelante=MAX_DIAS, simbolo="GOOGL").dias_adelante == 7
    esquema = Entrada.model_json_schema()
    assert "required" not in esquema and esquema.get("additionalProperties") is False
    assert esquema["properties"]["simbolo"]["enum"] == list(SIMBOLOS) and esquema["properties"]["simbolo"]["default"] == "AAPL"
    campo = esquema["properties"]["dias_adelante"]
    assert (campo["minimum"], campo["maximum"], campo["default"]) == (1, 7, 1) and campo["description"] and esquema["properties"]["simbolo"]["description"]
    assert campo["examples"] == [1]


def test_los_simbolos_del_router_son_los_del_entrenamiento_y_cada_uno_tiene_nombre_y_alias():
    assert list(SIMBOLOS) == SIMBOLOS_TRAIN and set(NOMBRES) == set(SIMBOLOS)
    assert set(ALIAS_SIMBOLOS.values()) == set(SIMBOLOS)
    assert NOMBRES == {"AAPL": "Apple", "MSFT": "Microsoft", "AMZN": "Amazon", "GOOGL": "Google"}
    assert ALIAS_SIMBOLOS == {"apple": "AAPL", "microsoft": "MSFT", "amazon": "AMZN", "google": "GOOGL", "alphabet": "GOOGL"}
    assert all(a == a.lower() and a.isascii() for a in ALIAS_SIMBOLOS)


def test_el_maximo_de_la_api_cubre_los_horizontes_evaluados():
    assert MAX_DIAS == MAX_HORIZONTE == max(HORIZONTES) == 7 and set(HORIZONTES) <= set(range(1, MAX_DIAS + 1))


@pytest.mark.parametrize("cuerpo", [{"dias_adelante": 0}, {"dias_adelante": 8}, {"dias_adelante": -1}, {"dias_adelante": "3"},
                                    {"dias_adelante": 2.5}, {"dias_adelante": 3.0}, {"dias_adelante": True}, {"dias_adelante": None},
                                    {"simbolo": "TSLA"}, {"simbolo": "aapl"}, {"simbolo": "Apple"}, {"simbolo": ""}, {"simbolo": 1}, {"simbolo": None},
                                    {"symbol": "AAPL"}, {"dias_adelante": 2, "extra": 1}])
def test_entradas_invalidas_dan_422(cliente, assert_error, cuerpo):
    assert_error(cliente.post(URL, json=cuerpo), 422, "VALIDACION")


@pytest.mark.parametrize("cuerpo", [{"dias_adelante": 0}, {"dias_adelante": 8}, {"dias_adelante": "3"}, {"simbolo": "TSLA"}, {"simbolo": "aapl"}])
def test_la_entrada_rechaza_lo_mismo_sin_pasar_por_la_api(cuerpo):
    with pytest.raises(ValidationError):
        Entrada(**cuerpo)


def test_formato_de_fechas_y_dolares():
    assert [fecha_en_texto(date(2018, m, 1)) for m in range(1, 13)] == [f"1 de {mes} de 2018" for mes in (
        "enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre")]
    assert dolares(1416.78) == "1 416.78" and dolares(93.8) == "93.80" and dolares(1000000.5) == "1 000 000.50"


def test_la_fecha_destino_cuenta_dias_habiles():
    jueves = date(2018, 2, 8)
    assert fecha_destino(date(2018, 2, 7), 1) == jueves
    assert fecha_destino(date(2018, 2, 7), 7) == date(2018, 2, 16)
    assert fecha_destino(date(2018, 2, 9), 1) == date(2018, 2, 12), "un viernes salta al lunes"
    assert fecha_destino(date(2018, 2, 9), 3) == date(2018, 2, 14)
    assert all(fecha_destino(date(2018, 2, 7), d).weekday() < 5 for d in range(1, 8))


# --- Modelo entrenado y API ---

@entrenado
def test_el_artefacto_guarda_lo_necesario_para_servir():
    a = cargar_artefacto(CARPETA, "sp500")
    df = cargar_datos()
    assert a["fecha_max"] == "2018-02-07" and a["variables"] == CARACTERISTICAS and a["reentrenado_con_todo"] is True
    assert a["simbolos"] == list(SIMBOLOS) and set(a["sigma"]) == set(a["ultimos_cierres"]) == set(SIMBOLOS)
    for s in SIMBOLOS:
        assert a["ultimos_cierres"][s] == df[df.simbolo == s].cierre.tail(VENTANA_MINIMA).tolist()
        assert 0.01 < a["sigma"][s] < 0.03
    assert a["sigma"]["AMZN"] > a["sigma"]["GOOGL"], "cada símbolo tiene su propia volatilidad"
    assert date.fromisoformat(a["fecha_max"]).weekday() < 5


@entrenado
def test_el_modelo_elegido_es_el_ridge_y_nunca_una_linea_base():
    m = leer_metricas(CARPETA)["metricas"]
    assert m["modelo"] == "ridge" and not es_linea_base(m["modelo"])
    cv = m["comparacion_cv"]
    assert min(v["cv_rmse_retorno"] for n, v in cv.items() if not es_linea_base(n)) == cv[m["modelo"]]["cv_rmse_retorno"]


@entrenado
def test_las_metricas_publicadas_son_coherentes():
    m = leer_metricas(CARPETA)["metricas"]
    assert set(m["horizontes"]) == {"1", "3", "7"} and m["n_prueba"] == 252 and m["n_entrenamiento"] == 3904
    err = [m["horizontes"][str(h)]["modelo"]["rmse_relativo_pct"] for h in HORIZONTES]
    assert err == sorted(err), "el error crece con los días adelante"
    for h in HORIZONTES:
        e = m["horizontes"][str(h)]
        assert e["n_origenes"] == 246 and set(e["habilidad_por_simbolo"]) == set(SIMBOLOS) == set(e["por_simbolo_usd"])
        assert e["habilidad_ic95"][0] <= e["habilidad_frente_a_persistencia"] <= e["habilidad_ic95"][1]
        assert e["habilidad_frente_a_deriva_ic95"][0] <= e["habilidad_frente_a_deriva"] <= e["habilidad_frente_a_deriva_ic95"][1]
        assert 0.95 <= e["cobertura_intervalo_95"]["global"] <= 1.0 and set(e["cobertura_intervalo_95"]) == {"global", *SIMBOLOS}


@entrenado
def test_la_persistencia_publicada_coincide_con_un_calculo_independiente():
    df = cargar_datos()
    corte = particion_temporal(df)
    m = leer_metricas(CARPETA)["metricas"]["horizontes"]
    t = np.arange(corte - 1, 1259 - MAX_HORIZONTE)
    for h in HORIZONTES:
        errores = []
        for s in SIMBOLOS:
            c = df[df.simbolo == s].cierre.to_numpy()
            errores.append((c[t + h] - c[t]) / c[t + h])
        assert m[str(h)]["persistencia"]["rmse_relativo_pct"] == pytest.approx(100 * np.sqrt(np.mean(np.square(errores))), abs=1e-3)
        assert m[str(h)]["persistencia"]["mape_pct"] == pytest.approx(100 * np.mean(np.abs(errores)), abs=1e-3)
    for s in SIMBOLOS:
        c = df[df.simbolo == s].cierre.to_numpy()
        assert m["1"]["por_simbolo_usd"][s]["rmse_persistencia"] == pytest.approx(np.sqrt(np.mean((c[t + 1] - c[t]) ** 2)), abs=1e-3)


@entrenado
def test_la_ventaja_sobre_la_persistencia_es_solo_la_deriva_y_el_texto_de_la_api_lo_dice(cliente):
    """El texto afirma que la ventaja vino solo del crecimiento medio: frente a la deriva la habilidad es nula (IC con 0) en todos los horizontes."""
    m = leer_metricas(CARPETA)["metricas"]["horizontes"]
    for h in HORIZONTES:
        e = m[str(h)]
        assert e["habilidad_frente_a_deriva_ic95"][0] <= 0 <= e["habilidad_frente_a_deriva_ic95"][1], h
        assert abs(e["habilidad_frente_a_deriva"]) < 0.01 and e["habilidad_frente_a_persistencia"] > 0
        assert e["habilidad_deriva_frente_a_persistencia"] == pytest.approx(e["habilidad_frente_a_persistencia"], abs=0.01)
    assert "vino solo del crecimiento medio histórico y no de las variables que usa" in cliente.post(URL, json={}).json()["texto"]


@entrenado
def test_el_acierto_de_direccion_no_supera_al_de_predecir_siempre_que_sube():
    m = leer_metricas(CARPETA)["metricas"]["horizontes"]
    for h in HORIZONTES:
        assert m[str(h)]["modelo"]["acierto_direccion"] <= m[str(h)]["siempre_sube"]["acierto_direccion"] + 0.02


@entrenado
def test_predecir_por_defecto_es_apple_la_sesion_siguiente_al_ultimo_dato(cliente):
    cuerpo = cliente.post(URL, json={}).json()
    a = cargar_artefacto(CARPETA, "sp500")
    assert cuerpo["modelo"] == "sp500" and cuerpo["unidad"] == "USD" and cuerpo["probabilidades"] is None
    assert cuerpo["prediccion"] == pytest.approx(recursiva(a["pipeline"], [a["ultimos_cierres"]["AAPL"]], 1)[0, 0], abs=0.01)
    assert "Apple (AAPL)" in cuerpo["texto"] and "8 de febrero de 2018" in cuerpo["texto"] and "7 de febrero de 2018" in cuerpo["texto"]
    assert cliente.post(URL, json={"simbolo": "AAPL", "dias_adelante": 1}).json() == cuerpo


@entrenado
@pytest.mark.parametrize("simbolo", SIMBOLOS)
@pytest.mark.parametrize("dias", range(1, MAX_DIAS + 1))
def test_cada_simbolo_y_horizonte_coincide_con_la_recursion_directa_y_su_fecha(cliente, simbolo, dias):
    cuerpo = cliente.post(URL, json={"simbolo": simbolo, "dias_adelante": dias}).json()
    a = cargar_artefacto(CARPETA, "sp500")
    esperado = recursiva(a["pipeline"], [a["ultimos_cierres"][simbolo]], dias)[0, -1]
    assert cuerpo["prediccion"] == pytest.approx(esperado, abs=0.01)
    assert fecha_en_texto(fecha_destino(date(2018, 2, 7), dias)) in cuerpo["texto"] and f"{NOMBRES[simbolo]} ({simbolo})" in cuerpo["texto"]
    assert abs(cuerpo["prediccion"] / a["ultimos_cierres"][simbolo][-1] - 1) < 0.1


@entrenado
def test_cada_simbolo_parte_de_su_propio_precio_y_no_se_mezclan(cliente):
    df = cargar_datos()
    for s in SIMBOLOS:
        ultimo = df[df.simbolo == s].cierre.iloc[-1]
        assert cliente.post(URL, json={"simbolo": s}).json()["prediccion"] == pytest.approx(ultimo, rel=0.01)


@entrenado
def test_el_intervalo_usa_la_volatilidad_de_cada_simbolo_y_crece_con_la_raiz_de_los_dias(cliente):
    a = cargar_artefacto(CARPETA, "sp500")
    anchos = {}
    for simbolo in SIMBOLOS:
        for dias in (1, 4):
            cuerpo = cliente.post(URL, json={"simbolo": simbolo, "dias_adelante": dias}).json()
            bajo, alto = (float(s.replace(" ", "")) for s in re.search(r"entre ([\d .]+) y ([\d .]+) dólares", cuerpo["texto"]).groups())
            factor = np.exp(1.96 * a["sigma"][simbolo] * np.sqrt(dias))
            assert bajo == pytest.approx(cuerpo["prediccion"] / factor, abs=0.01) and alto == pytest.approx(cuerpo["prediccion"] * factor, abs=0.01)
            assert bajo < cuerpo["prediccion"] < alto
            anchos[(simbolo, dias)] = np.log(alto / bajo)
        assert anchos[(simbolo, 4)] / anchos[(simbolo, 1)] == pytest.approx(2, rel=0.01)
    assert anchos[("AMZN", 1)] > anchos[("GOOGL", 1)], "Amazon es más volátil que Google"


@entrenado
def test_el_texto_aclara_que_no_conoce_el_precio_actual_ni_es_una_recomendacion(cliente):
    texto = cliente.post(URL, json={"simbolo": "MSFT", "dias_adelante": 3}).json()["texto"]
    assert "no conoce el precio actual" in texto and "no una recomendación de inversión" in texto
    assert "dólares el 12 de febrero de 2018" in texto and "Microsoft (MSFT)" in texto


@entrenado
def test_info_incluye_metricas_esquema_y_ejemplo_valido(cliente):
    cuerpo = cliente.get("/api/modelos/sp500/info").json()
    assert {"horizontes", "modelo", "comparacion_cv", "sigma_retorno_diario"} <= set(cuerpo["metricas"])
    assert set(cuerpo["esquema_entrada"]["properties"]) == {"simbolo", "dias_adelante"}
    assert cuerpo["entrada_ejemplo"] == {"simbolo": "AAPL", "dias_adelante": 1} and cliente.post(URL, json=cuerpo["entrada_ejemplo"]).status_code == 200


@entrenado
def test_el_artefacto_no_es_excesivo():
    assert (CARPETA / "modelo.joblib").stat().st_size < 5_000_000
    assert joblib.load(CARPETA / "modelo.joblib")["pipeline"] is not None


# --- Entrenamiento (sin tocar el disco) ---

def test_el_entrenamiento_no_usa_la_prueba_ni_para_elegir_ni_para_ajustar(reentrenado):
    """Se inflan 10 veces los cierres de la prueba: todo lo que sale del entrenamiento debe quedar idéntico."""
    df = cargar_datos()
    corte = particion_temporal(df)
    alterado = df.copy()
    alterado["posicion"] = alterado.groupby("simbolo").cumcount()
    alterado.loc[alterado.posicion >= corte, OBJETIVO] *= 10
    otro = entrenar(alterado.drop(columns="posicion"), figuras=None, imprimir=False)
    for clave in ("comparacion_cv", "modelo", "sigma_retorno_diario", "n_entrenamiento", "retorno_medio_dia_entrenamiento"):
        assert otro["metricas"][clave] == reentrenado["metricas"][clave], clave
    assert otro["metricas"]["horizontes"] != reentrenado["metricas"]["horizontes"]


def test_entrenar_devuelve_todo_lo_que_despues_se_publica(reentrenado):
    assert set(reentrenado) == {"pipeline", "metricas", "sigma", "ultimos_cierres", "fecha_max", "entrada_ejemplo"}
    assert set(reentrenado["sigma"]) == set(reentrenado["ultimos_cierres"]) == set(SIMBOLOS)


def test_el_modelo_final_se_reentrena_con_todos_los_dias_y_todos_los_simbolos(reentrenado):
    X, y = construir(cargar_datos())
    assert len(X) == 4 * (1259 - VENTANA_MINIMA)
    assert reentrenado["pipeline"].predict(X) == pytest.approx(candidatos()["ridge"].fit(X, y).predict(X))
    assert reentrenado["fecha_max"] == "2018-02-07"


@pytest.mark.reproduce
@entrenado
def test_reentrenar_reproduce_exactamente_lo_publicado(reentrenado):
    """Cubre todo `entrenar()`: selección, evaluación por horizonte y símbolo, bootstrap por bloques y el ajuste final."""
    publicado = leer_metricas(CARPETA)
    artefacto = cargar_artefacto(CARPETA, "sp500")
    assert reentrenado["metricas"] == publicado["metricas"]
    assert reentrenado["entrada_ejemplo"] == publicado["entrada_ejemplo"]
    assert reentrenado["sigma"] == artefacto["sigma"] and reentrenado["ultimos_cierres"] == artefacto["ultimos_cierres"]
    X, _ = construir(cargar_datos())
    assert reentrenado["pipeline"].predict(X) == pytest.approx(artefacto["pipeline"].predict(X), abs=1e-12)


def test_el_texto_da_el_punto_estimado_y_el_rango_del_95(cliente):
    for simbolo, dias in (("AAPL", 1), ("AMZN", 5)):
        cuerpo = cliente.post(URL, json={"simbolo": simbolo, "dias_adelante": dias}).json()
        assert f"cerraría alrededor de {dolares(cuerpo['prediccion'])} dólares" in cuerpo["texto"] and "con un rango del 95 % entre" in cuerpo["texto"]


def test_la_cobertura_por_mitades_se_publica_y_el_intervalo_es_conservador(reentrenado):
    """El σ viene de un entrenamiento más volátil que la prueba (≈ 1.5 % contra 1.1 % diario): los intervalos cubren más del 95 %."""
    for h, e in reentrenado["metricas"]["horizontes"].items():
        primera, segunda = e["cobertura_intervalo_95_por_mitad"]
        assert 0.95 <= primera <= 1 and 0.95 <= segunda <= 1
        assert e["cobertura_intervalo_95"]["global"] == pytest.approx((primera + segunda) / 2, abs=0.005)


def test_el_entrenamiento_en_memoria_coincide_con_el_calculo_independiente_de_las_lineas_base(reentrenado):
    """Sin leer metricas.json: la persistencia y la deriva se recalculan desde los datos, con el número de orígenes."""
    df = cargar_datos()
    corte = particion_temporal(df)
    t = np.arange(corte - 1, 1259 - MAX_HORIZONTE)
    deriva = reentrenado["metricas"]["retorno_medio_dia_entrenamiento"]
    for h in HORIZONTES:
        e = reentrenado["metricas"]["horizontes"][str(h)]
        persistencia, con_deriva = [], []
        for s in SIMBOLOS:
            c = df[df.simbolo == s].cierre.to_numpy()
            persistencia.append((c[t + h] - c[t]) / c[t + h])
            con_deriva.append((c[t + h] - c[t] * np.exp(deriva * h)) / c[t + h])
        assert e["n_origenes"] == len(t) == 246
        assert e["persistencia"]["rmse_relativo_pct"] == pytest.approx(100 * np.sqrt(np.mean(np.square(persistencia))), abs=1e-3)
        assert e["deriva"]["rmse_relativo_pct"] == pytest.approx(100 * np.sqrt(np.mean(np.square(con_deriva))), rel=0.02), "deriva fijada con el entrenamiento"
        assert e["habilidad_deriva_frente_a_persistencia"] == pytest.approx(1 - e["deriva"]["rmse_relativo_pct"] / e["persistencia"]["rmse_relativo_pct"], abs=1e-3)
    assert reentrenado["metricas"]["n_prueba"] == 252 and reentrenado["metricas"]["fecha_inicio_prueba"] == "2017-02-08"


def test_main_escribe_el_artefacto_con_todo_lo_necesario(tmp_path, monkeypatch):
    shutil.copy(train.CARPETA / "dataset.csv", tmp_path / "dataset.csv")
    monkeypatch.setattr(train, "CARPETA", tmp_path)
    train.main()
    a = joblib.load(tmp_path / "modelo.joblib")
    publicado = cargar_artefacto(CARPETA, "sp500")
    assert a["reentrenado_con_todo"] is True and a["variables"] == CARACTERISTICAS and a["fecha_max"] == "2018-02-07" and a["simbolos"] == list(SIMBOLOS)
    assert a["metricas"] == leer_metricas(CARPETA)["metricas"] and a["ultimos_cierres"] == publicado["ultimos_cierres"] and a["sigma"] == publicado["sigma"]
    assert sorted(p.name for p in (tmp_path / "figuras").glob("*.png")) == sorted(
        ["autocorrelacion.png", "correlacion_retornos.png", "habilidad_por_horizonte.png", "prueba_1_dia.png", "retornos_por_simbolo.png", "series_normalizadas.png"])
    assert (tmp_path / "metricas.json").exists()


@entrenado
def test_la_prediccion_se_redondea_a_dos_decimales(cliente):
    for simbolo in SIMBOLOS:
        for dias in (1, 7):
            valor = cliente.post(URL, json={"simbolo": simbolo, "dias_adelante": dias}).json()["prediccion"]
            assert valor == round(valor, 2)
    a = cargar_artefacto(CARPETA, "sp500")
    exacto = recursiva(a["pipeline"], [a["ultimos_cierres"]["AMZN"]], 1)[0, 0]
    assert abs(exacto - round(exacto, 2)) > 1e-9, "el valor sin redondear tiene más decimales: la prueba sí ejercita el redondeo"
    assert cliente.post(URL, json={"simbolo": "AMZN"}).json()["prediccion"] == round(exacto, 2)


@entrenado
def test_cada_empresa_se_nombra_en_el_texto_con_su_nombre_literal(cliente):
    esperado = {"AAPL": "Apple (AAPL)", "MSFT": "Microsoft (MSFT)", "AMZN": "Amazon (AMZN)", "GOOGL": "Google (GOOGL)"}
    for simbolo, nombre in esperado.items():
        assert nombre in cliente.post(URL, json={"simbolo": simbolo}).json()["texto"]


def test_main_imprime_el_entendimiento_de_los_datos(tmp_path, monkeypatch, capsys):
    shutil.copy(train.CARPETA / "dataset.csv", tmp_path / "dataset.csv")
    monkeypatch.setattr(train, "CARPETA", tmp_path)
    train.main()
    assert "Filas: 5036" in capsys.readouterr().out
