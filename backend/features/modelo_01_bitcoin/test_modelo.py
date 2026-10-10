"""Pruebas del modelo 01 · Bitcoin (specs/api_rest_spec.md §3). Usan el modelo entrenado real."""
import re
import shutil
from datetime import date, timedelta

import joblib
import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from core.modelos import REGISTRO, cargar_artefacto, leer_metricas
from features.modelo_01_bitcoin import train
from features.modelo_01_bitcoin.router import CARPETA, MAX_DIAS, Entrada, dolares, fecha_en_texto
from features.modelo_01_bitcoin.serie import CARACTERISTICAS, VENTANA_MINIMA, caracteristicas, recursiva, ultima_fila
from features.modelo_01_bitcoin.train import (HORIZONTES, MAX_HORIZONTE, OBJETIVO, autocorrelacion, candidatos, cargar_datos, construir,
                                              entender, entrenar, es_linea_base, explorar, intervalo_habilidad, origenes, particion_temporal)

URL = "/api/modelos/bitcoin/predecir"
entrenado = pytest.mark.skipif(not REGISTRO["bitcoin"].entrenado, reason="ejecutar features.modelo_01_bitcoin.train")


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

def test_los_datos_cargados_van_de_la_fecha_mas_antigua_a_la_mas_reciente():
    df = cargar_datos()
    assert len(df) == 1556 and list(df.columns) == ["fecha", "apertura", "maximo", "minimo", OBJETIVO, "volumen"]
    assert df.fecha.iloc[0] == pd.Timestamp("2013-04-28") and df.fecha.iloc[-1] == pd.Timestamp("2017-07-31")
    assert df.cierre.iloc[0] == 134.21 and df.cierre.iloc[-1] == 2875.34
    assert df.fecha.is_monotonic_increasing and not df.fecha.duplicated().any()


def test_no_faltan_dias_en_la_serie():
    """La predicción recursiva y los rezagos suponen un registro por día: sin huecos."""
    assert (cargar_datos().fecha.diff().dropna() == pd.Timedelta(days=1)).all()


def test_el_formato_de_fecha_mes_dia_ano_no_se_confunde():
    df = cargar_datos()
    assert df.fecha[df.cierre.idxmax()] == pd.Timestamp("2017-06-11") and df.cierre.max() == 2958.11
    assert df.fecha[df.cierre.idxmin()] == pd.Timestamp("2013-07-05") and df.cierre.min() == 68.43
    assert df.fecha.dt.month.nunique() == 12 and df.fecha.dt.day.max() == 31


def test_los_precios_son_coherentes_minimo_apertura_cierre_maximo():
    df = cargar_datos()
    assert ((df.minimo <= df.cierre) & (df.cierre <= df.maximo) & (df.minimo <= df.apertura) & (df.apertura <= df.maximo)).all()
    assert (df[["apertura", "maximo", "minimo", OBJETIVO]] > 0).all().all()


def test_el_volumen_se_lee_con_separadores_y_los_guiones_son_los_primeros_243_dias():
    df = cargar_datos()
    assert df.volumen.isna().iloc[:243].all() and df.volumen.notna().iloc[243:].all()
    assert df.volumen.iloc[-1] == 860_575_000


def test_el_volumen_no_es_variable_del_modelo():
    """No se puede predecir el volumen futuro, así que no podría encadenarse en la predicción recursiva."""
    assert not any("volumen" in c for c in CARACTERISTICAS)
    assert set(CARACTERISTICAS) == {"ret_1", "ret_2", "ret_3", "ret_4", "ret_5", "ret_7", "ret_30", "dist_media_7", "dist_media_30", "vol_7", "vol_30"}


# --- Características y predicción recursiva ---

def test_caracteristicas_calculadas_a_mano():
    c = np.array([100.0, 110.0, 99.0, 108.9, 120.0, 130.0, 125.0, 140.0] + [150.0] * 30)
    f = caracteristicas(c)
    t = 7
    assert f.loc[t, "ret_1"] == pytest.approx(np.log(140 / 125))
    assert f.loc[t, "ret_2"] == pytest.approx(np.log(125 / 130)) and f.loc[t, "ret_3"] == pytest.approx(np.log(130 / 120))
    assert f.loc[t, "ret_4"] == pytest.approx(np.log(120 / 108.9)) and f.loc[t, "ret_5"] == pytest.approx(np.log(108.9 / 99))
    assert f.loc[t, "ret_7"] == pytest.approx(np.log(140 / 100))
    assert f.loc[7, "dist_media_7"] == pytest.approx(np.log(140 / np.mean(c[1:8])))
    assert f.loc[7, "vol_7"] == pytest.approx(np.std(np.diff(np.log(c[:8])), ddof=1), abs=1e-12)


def test_las_primeras_filas_son_nulas_hasta_tener_historial_suficiente():
    f = caracteristicas(serie_sintetica())
    assert f.iloc[:VENTANA_MINIMA - 1].isna().any(axis=1).all() and f.iloc[VENTANA_MINIMA - 1:].notna().all().all()
    assert list(f.columns) == CARACTERISTICAS


def test_ultima_fila_coincide_con_la_tabla_completa_en_todos_los_dias():
    c = serie_sintetica(200)
    f = caracteristicas(c)
    for t in range(VENTANA_MINIMA - 1, len(c)):
        assert ultima_fila(c[:t + 1]) == pytest.approx(f.loc[t].to_dict(), abs=1e-12)


def test_las_caracteristicas_no_miran_el_futuro():
    """Cambiar los cierres posteriores a un día no modifica sus características."""
    c = serie_sintetica(150)
    c2 = c.copy()
    c2[100:] *= 7
    assert caracteristicas(c).iloc[:100].equals(caracteristicas(c2).iloc[:100])


def test_ultima_fila_exige_el_historial_minimo_y_solo_usa_el_final():
    c = serie_sintetica(80)
    with pytest.raises(ValueError, match=str(VENTANA_MINIMA)):
        ultima_fila(c[:VENTANA_MINIMA - 1])
    assert ultima_fila(c) == ultima_fila(c[-VENTANA_MINIMA:])


def test_el_objetivo_es_el_retorno_del_dia_siguiente_y_se_descartan_los_extremos():
    c = serie_sintetica(100)
    X, y = construir(c)
    assert X.index[0] == VENTANA_MINIMA - 1 and X.index[-1] == len(c) - 2 and list(X.index) == list(y.index)
    for t in (30, 55, 98):
        assert y[t] == pytest.approx(np.log(c[t + 1] / c[t]))
    assert X.notna().all().all() and y.notna().all()


def test_recursiva_con_retorno_constante_compone_el_precio():
    c = serie_sintetica(60)
    r = recursiva(Constante(0.01), [c], 5)
    assert r.shape == (1, 5)
    assert r[0] == pytest.approx(c[-1] * np.exp(0.01 * np.arange(1, 6)))


def test_recursiva_realimenta_sus_propias_predicciones():
    """Con un modelo que repite ret_1, el primer paso usa el retorno real y los siguientes el predicho: crecimiento geométrico."""
    c = serie_sintetica(60)
    r1 = np.log(c[-1] / c[-2])
    salida = recursiva(Momento(), [c], 4)[0]
    assert salida == pytest.approx(c[-1] * np.exp(r1 * np.arange(1, 5)))


def test_recursiva_en_lote_es_igual_a_una_por_una_y_ignora_historial_antiguo():
    historiales = [serie_sintetica(70, s) for s in range(4)]
    lote = recursiva(Momento(), historiales, 6)
    for i, h in enumerate(historiales):
        assert lote[i] == pytest.approx(recursiva(Momento(), [h], 6)[0])
        assert lote[i] == pytest.approx(recursiva(Momento(), [h[-VENTANA_MINIMA:]], 6)[0])
    assert recursiva(Momento(), historiales, 3) == pytest.approx(lote[:, :3])


def test_recursiva_no_modifica_los_historiales_recibidos():
    c = serie_sintetica(60)
    copia = c.copy()
    recursiva(Constante(0.02), [c], 5)
    assert np.array_equal(c, copia)


# --- Partición y selección ---

def test_la_particion_es_temporal_y_no_mira_el_futuro():
    df = cargar_datos()
    corte, n_prueba = particion_temporal(len(df))
    assert corte == int(len(df) * 0.8) == 1244 and n_prueba == len(df) - corte == 312
    X, _ = construir(df[OBJETIVO].to_numpy())
    entrena, prueba = origenes(X, corte)
    assert entrena.max() == corte - 2 and prueba.min() == corte - 1 and entrena.min() == VENTANA_MINIMA - 1
    assert not set(entrena) & set(prueba)
    assert entrena.max() + 1 < corte, "el objetivo de entrenamiento (día siguiente) cae antes de la prueba"


def test_candidatos_dos_lineas_base_y_tres_modelos():
    c = candidatos()
    assert [n for n in c if es_linea_base(n)] == ["persistencia (línea base)", "deriva (línea base)"]
    assert set(c) - {n for n in c if es_linea_base(n)} == {"ridge", "random forest", "gradient boosting"}
    assert c["gradient boosting"].named_steps["modelo"].early_stopping is False
    X = pd.DataFrame(np.random.default_rng(0).normal(size=(50, len(CARACTERISTICAS))), columns=CARACTERISTICAS)
    y = pd.Series(np.random.default_rng(1).normal(0.01, 0.02, 50))
    assert (c["persistencia (línea base)"].fit(X, y).predict(X) == 0).all()
    assert c["deriva (línea base)"].fit(X, y).predict(X) == pytest.approx(np.full(50, y.mean()))


def test_la_habilidad_del_bootstrap_en_casos_conocidos():
    e = np.random.default_rng(0).normal(size=200)
    assert intervalo_habilidad(e, e, 50) == [0.0, 0.0]
    ceros = np.zeros(200)
    assert intervalo_habilidad(ceros, e, 50) == [1.0, 1.0]
    mitad = intervalo_habilidad(e / 2, e, 100)
    assert mitad[0] == pytest.approx(0.5, abs=1e-9) and mitad[1] == pytest.approx(0.5, abs=1e-9)
    peor = intervalo_habilidad(e * 1.5, e, 100)
    assert peor[0] == pytest.approx(-0.5, abs=1e-9) and peor[1] == pytest.approx(-0.5, abs=1e-9)
    ruido = np.random.default_rng(1).normal(size=200)
    ic = intervalo_habilidad(ruido, e, 200)
    assert ic[0] < 0 < ic[1], "errores independientes de igual tamaño: sin diferencia demostrable"
    assert intervalo_habilidad(ruido, e, 50) == intervalo_habilidad(ruido, e, 50)


def test_autocorrelacion_en_casos_conocidos():
    alterna = np.tile([1.0, -1.0], 200)
    assert autocorrelacion(alterna, 2) == pytest.approx([-1.0, 1.0], abs=0.01)
    ruido = np.random.default_rng(0).normal(size=5000)
    assert np.abs(autocorrelacion(ruido, 10)).max() < 0.06
    assert autocorrelacion(ruido + 50, 10) == pytest.approx(autocorrelacion(ruido, 10)), "se resta la media antes de correlacionar"


def test_entender_describe_los_datos(capsys):
    entender(cargar_datos())
    salida = capsys.readouterr().out
    assert "Filas: 1556" in salida and "2013-04-28 a 2017-07-31" in salida
    assert "{1.0: 1555}" in salida and "fechas repetidas: 0" in salida
    assert re.search(r"volumen\s+243", salida) and "Retorno diario: media 0.0020, desviación 0.0426" in salida


def test_explorar_genera_las_figuras_del_analisis(tmp_path, capsys):
    df = cargar_datos()
    explorar(df, tmp_path, df.fecha[particion_temporal(len(df))[0]])
    assert sorted(p.name for p in tmp_path.glob("*.png")) == ["autocorrelacion.png", "distribucion_retornos.png", "serie_precio.png", "volatilidad.png"]
    salida = capsys.readouterr().out
    assert "Autocorrelación de los retornos (rezagos 1-5)" in salida and "antes de 2013-12-27" in salida


# --- Entrada y metadatos ---

def test_los_metadatos_del_modelo_registrado():
    info = REGISTRO["bitcoin"].info
    assert info["slug"] == "bitcoin" and info["tipo"] == "regresion" and info["unidad"] == "USD"
    assert REGISTRO["bitcoin"].entrada is Entrada
    assert info["comandos"] == ["precio del bitcoin", "bitcoin manana", "tipo de cambio del bitcoin"]


def test_entrada_por_defecto_es_un_dia_y_todos_los_campos_tienen_valor():
    assert Entrada().dias_adelante == 1 and Entrada(dias_adelante=MAX_DIAS).dias_adelante == 7
    esquema = Entrada.model_json_schema()
    assert "required" not in esquema and esquema.get("additionalProperties") is False
    campo = esquema["properties"]["dias_adelante"]
    assert (campo["minimum"], campo["maximum"], campo["default"]) == (1, 7, 1) and campo["description"]


def test_el_maximo_de_la_api_cubre_los_horizontes_evaluados():
    assert MAX_DIAS == MAX_HORIZONTE == max(HORIZONTES) == 7 and set(HORIZONTES) <= set(range(1, MAX_DIAS + 1))


@pytest.mark.parametrize("cuerpo", [{"dias_adelante": 0}, {"dias_adelante": 8}, {"dias_adelante": -1}, {"dias_adelante": "3"},
                                    {"dias_adelante": 2.5}, {"dias_adelante": 3.0}, {"dias_adelante": True}, {"dias_adelante": None},
                                    {"dias": 3}, {"dias_adelante": 2, "extra": 1}])
def test_entradas_invalidas_dan_422(cliente, assert_error, cuerpo):
    assert_error(cliente.post(URL, json=cuerpo), 422, "VALIDACION")


@pytest.mark.parametrize("cuerpo", [{"dias_adelante": 0}, {"dias_adelante": 8}, {"dias_adelante": "3"}, {"dias_adelante": True}])
def test_la_entrada_rechaza_lo_mismo_sin_pasar_por_la_api(cuerpo):
    with pytest.raises(ValidationError):
        Entrada(**cuerpo)


def test_formato_de_fechas_y_dolares():
    assert fecha_en_texto(date(2017, 8, 1)) == "1 de agosto de 2017" and fecha_en_texto(date(2018, 1, 7)) == "7 de enero de 2018"
    assert dolares(2875.34) == "2 875" and dolares(999.5) == "1 000" and dolares(68.4) == "68"


# --- Modelo entrenado y API ---

@entrenado
def test_el_artefacto_guarda_lo_necesario_para_servir():
    a = cargar_artefacto(CARPETA, "bitcoin")
    df = cargar_datos()
    assert a["fecha_max"] == "2017-07-31" and a["variables"] == CARACTERISTICAS and a["reentrenado_con_todo"] is True
    assert a["ultimos_cierres"] == df.cierre.tail(VENTANA_MINIMA).tolist() and 0.03 < a["sigma"] < 0.06


@entrenado
def test_el_modelo_elegido_es_el_ridge_y_nunca_una_linea_base():
    m = leer_metricas(CARPETA)["metricas"]
    assert m["modelo"] == "ridge" and not es_linea_base(m["modelo"])
    cv = m["comparacion_cv"]
    assert min(v["cv_rmse_retorno"] for n, v in cv.items() if not es_linea_base(n)) == cv[m["modelo"]]["cv_rmse_retorno"]


@entrenado
def test_las_metricas_publicadas_son_coherentes():
    m = leer_metricas(CARPETA)["metricas"]
    assert set(m["horizontes"]) == {"1", "3", "7"} and m["n_prueba"] == 312 and m["n_entrenamiento"] == 1213
    rmse = [m["horizontes"][str(h)]["modelo"]["rmse"] for h in HORIZONTES]
    assert rmse == sorted(rmse), "el error crece con los días adelante"
    for h in HORIZONTES:
        e = m["horizontes"][str(h)]
        assert e["n_origenes"] == 306 and 0.9 <= e["cobertura_intervalo_95"] <= 1.0
        assert e["habilidad_ic95"][0] <= e["habilidad_frente_a_persistencia"] <= e["habilidad_ic95"][1]
        assert e["modelo"]["acierto_direccion"] == e["siempre_sube"]["acierto_direccion"], "el modelo solo repite la tendencia: siempre predice que sube"


@entrenado
def test_la_persistencia_publicada_coincide_con_un_calculo_independiente():
    df = cargar_datos()
    c = df.cierre.to_numpy()
    corte, _ = particion_temporal(len(c))
    m = leer_metricas(CARPETA)["metricas"]["horizontes"]
    for h in HORIZONTES:
        t = np.arange(corte - 1, len(c) - MAX_HORIZONTE)
        assert m[str(h)]["persistencia"]["rmse"] == pytest.approx(np.sqrt(np.mean((c[t + h] - c[t]) ** 2)), abs=1e-3)


@entrenado
def test_no_hay_habilidad_demostrable_y_el_texto_de_la_api_lo_dice(cliente):
    """La frase 'no mejoró de forma demostrable' de la respuesta debe ser verdad: el IC 95 % de la habilidad incluye el 0."""
    m = leer_metricas(CARPETA)["metricas"]["horizontes"]
    assert all(m[str(h)]["habilidad_ic95"][0] <= 0 for h in HORIZONTES), "si el modelo mejorara, hay que cambiar el texto del router"
    assert all(abs(m[str(h)]["habilidad_frente_a_persistencia"]) < 0.05 for h in HORIZONTES)
    assert "no mejoró de forma demostrable a repetir el último precio" in cliente.post(URL, json={}).json()["texto"]


@entrenado
def test_predecir_por_defecto_es_el_dia_siguiente_al_ultimo_dato(cliente):
    cuerpo = cliente.post(URL, json={}).json()
    a = cargar_artefacto(CARPETA, "bitcoin")
    assert cuerpo["modelo"] == "bitcoin" and cuerpo["unidad"] == "USD" and cuerpo["probabilidades"] is None
    assert cuerpo["prediccion"] == pytest.approx(recursiva(a["pipeline"], [a["ultimos_cierres"]], 1)[0, 0], abs=0.01)
    assert "1 de agosto de 2017" in cuerpo["texto"] and "31 de julio de 2017" in cuerpo["texto"]
    assert cliente.post(URL, json={"dias_adelante": 1}).json() == cuerpo


@entrenado
@pytest.mark.parametrize("dias", range(1, MAX_DIAS + 1))
def test_cada_horizonte_coincide_con_la_recursion_directa_y_con_su_fecha(cliente, dias):
    cuerpo = cliente.post(URL, json={"dias_adelante": dias}).json()
    a = cargar_artefacto(CARPETA, "bitcoin")
    esperado = recursiva(a["pipeline"], [a["ultimos_cierres"]], dias)[0, -1]
    assert cuerpo["prediccion"] == pytest.approx(esperado, abs=0.01)
    destino = date(2017, 7, 31) + timedelta(days=dias)
    assert fecha_en_texto(destino) in cuerpo["texto"]
    assert abs(cuerpo["prediccion"] / a["ultimos_cierres"][-1] - 1) < 0.15, "una semana de predicción no se aleja del último precio"


@entrenado
def test_el_intervalo_del_texto_es_simetrico_en_log_y_crece_con_la_raiz_de_los_dias(cliente):
    a = cargar_artefacto(CARPETA, "bitcoin")
    amplitudes = {}
    for dias in (1, 4, 7):
        cuerpo = cliente.post(URL, json={"dias_adelante": dias}).json()
        bajo, alto = (int(s.replace(" ", "")) for s in re.search(r"entre ([\d ]+) y ([\d ]+) dólares", cuerpo["texto"]).groups())
        factor = np.exp(1.96 * a["sigma"] * np.sqrt(dias))
        assert bajo == pytest.approx(cuerpo["prediccion"] / factor, abs=1) and alto == pytest.approx(cuerpo["prediccion"] * factor, abs=1)
        assert bajo < cuerpo["prediccion"] < alto
        amplitudes[dias] = np.log(alto / bajo)
    assert amplitudes[4] / amplitudes[1] == pytest.approx(2, rel=0.01) and amplitudes[7] / amplitudes[1] == pytest.approx(np.sqrt(7), rel=0.01)


@entrenado
def test_el_texto_aclara_que_no_conoce_el_precio_actual_ni_es_una_recomendacion(cliente):
    texto = cliente.post(URL, json={"dias_adelante": 3}).json()["texto"]
    assert "no conoce el precio actual" in texto and "no una recomendación de inversión" in texto
    assert "dólares el 3 de agosto de 2017" in texto


@entrenado
def test_la_prediccion_a_un_dia_se_parece_al_ultimo_cierre_mas_la_deriva(cliente):
    """El ridge elegido casi no se aparta de la deriva (coeficientes ≈ 0): la predicción es el último precio por exp(retorno medio)."""
    a = cargar_artefacto(CARPETA, "bitcoin")
    cuerpo = cliente.post(URL, json={}).json()
    assert cuerpo["prediccion"] == pytest.approx(a["ultimos_cierres"][-1], rel=0.02) and cuerpo["prediccion"] > a["ultimos_cierres"][-1]


@entrenado
def test_info_incluye_metricas_esquema_y_ejemplo_valido(cliente):
    cuerpo = cliente.get("/api/modelos/bitcoin/info").json()
    assert {"horizontes", "modelo", "comparacion_cv", "sigma_retorno_diario"} <= set(cuerpo["metricas"])
    assert set(cuerpo["esquema_entrada"]["properties"]) == {"dias_adelante"} and cuerpo["entrada_ejemplo"] == {"dias_adelante": 1}
    assert cliente.post(URL, json=cuerpo["entrada_ejemplo"]).status_code == 200


@entrenado
def test_el_artefacto_no_es_excesivo():
    assert (CARPETA / "modelo.joblib").stat().st_size < 5_000_000
    assert joblib.load(CARPETA / "modelo.joblib")["pipeline"] is not None


# --- Entrenamiento (sin tocar el disco) ---

def test_el_entrenamiento_no_usa_la_prueba_ni_para_elegir_ni_para_ajustar(reentrenado):
    """Se inflan 10 veces los cierres de la prueba: todo lo que sale del entrenamiento debe quedar idéntico."""
    df = cargar_datos()
    corte, _ = particion_temporal(len(df))
    alterado = df.copy()
    alterado.loc[corte:, OBJETIVO] *= 10
    otro = entrenar(alterado, figuras=None, imprimir=False)
    for clave in ("comparacion_cv", "modelo", "sigma_retorno_diario", "n_entrenamiento", "retorno_medio_dia_entrenamiento"):
        assert otro["metricas"][clave] == reentrenado["metricas"][clave], clave
    assert otro["metricas"]["horizontes"] != reentrenado["metricas"]["horizontes"]


def test_entrenar_devuelve_todo_lo_que_despues_se_publica(reentrenado):
    assert set(reentrenado) == {"pipeline", "metricas", "sigma", "ultimos_cierres", "fecha_max", "entrada_ejemplo"}


def test_el_modelo_final_se_reentrena_con_todos_los_dias(reentrenado):
    df = cargar_datos()
    X, y = construir(df[OBJETIVO].to_numpy())
    assert reentrenado["pipeline"].predict(X) == pytest.approx(candidatos()["ridge"].fit(X, y).predict(X))
    assert reentrenado["fecha_max"] == "2017-07-31" and reentrenado["ultimos_cierres"][-1] == 2875.34


@pytest.mark.reproduce
@entrenado
def test_reentrenar_reproduce_exactamente_lo_publicado(reentrenado):
    """Cubre todo `entrenar()`: selección, evaluación por horizonte, bootstrap por bloques y el ajuste final."""
    publicado = leer_metricas(CARPETA)
    artefacto = cargar_artefacto(CARPETA, "bitcoin")
    assert reentrenado["metricas"] == publicado["metricas"]
    assert reentrenado["entrada_ejemplo"] == publicado["entrada_ejemplo"]
    assert reentrenado["sigma"] == artefacto["sigma"] and reentrenado["ultimos_cierres"] == artefacto["ultimos_cierres"]
    X, _ = construir(cargar_datos()[OBJETIVO].to_numpy())
    assert reentrenado["pipeline"].predict(X) == pytest.approx(artefacto["pipeline"].predict(X), abs=1e-12)


def test_las_cifras_de_exploracion_citadas_en_el_analisis():
    """analisis.md las cita: si los datos o el cálculo cambian, el texto debe actualizarse."""
    df = cargar_datos()
    r = np.log(df.cierre).diff()
    por_ano = r.groupby(df.fecha.dt.year).std().round(4).to_dict()
    assert por_ano == {2013: 0.0672, 2014: 0.0394, 2015: 0.0368, 2016: 0.0252, 2017: 0.0435}
    assert autocorrelacion(r.dropna().to_numpy(), 5).round(3).tolist() == [-0.001, -0.043, -0.019, 0.064, 0.039]
    assert autocorrelacion(r.dropna().to_numpy() ** 2, 1).round(2).tolist() == [0.32]
    assert round(r.mean(), 4) == 0.002 and round(r.std(), 4) == 0.0426 and round((r.dropna() > 0).mean(), 3) == 0.545
    assert round(df.cierre.iloc[-1] / df.cierre.iloc[1244], 2) == 4.77


def test_el_texto_da_el_punto_estimado_y_el_rango_del_95(cliente):
    for dias in (1, 5):
        cuerpo = cliente.post(URL, json={"dias_adelante": dias}).json()
        assert f"cerraría alrededor de {dolares(cuerpo['prediccion'])} dólares" in cuerpo["texto"] and "con un rango del 95 % entre" in cuerpo["texto"]


def test_la_cobertura_por_mitades_se_publica_y_la_segunda_mitad_es_menor(reentrenado):
    for h, e in reentrenado["metricas"]["horizontes"].items():
        primera, segunda = e["cobertura_intervalo_95_por_mitad"]
        assert 0.9 <= primera <= 1 and 0.9 <= segunda <= 1 and segunda < primera, "σ viene de años más volátiles: la prueba más reciente cubre menos"
        assert e["cobertura_intervalo_95"] == pytest.approx((primera + segunda) / 2, abs=0.005)


def test_el_entrenamiento_en_memoria_coincide_con_el_calculo_independiente_de_las_lineas_base(reentrenado):
    """Sin leer metricas.json: persistencia y deriva recalculadas desde los datos, con el número de orígenes."""
    c = cargar_datos().cierre.to_numpy()
    corte, _ = particion_temporal(len(c))
    t = np.arange(corte - 1, len(c) - MAX_HORIZONTE)
    deriva = reentrenado["metricas"]["retorno_medio_dia_entrenamiento"]
    for h in HORIZONTES:
        e = reentrenado["metricas"]["horizontes"][str(h)]
        assert e["n_origenes"] == len(t) == 306
        assert e["persistencia"]["rmse"] == pytest.approx(np.sqrt(np.mean((c[t + h] - c[t]) ** 2)), abs=1e-3)
        assert e["deriva"]["rmse"] == pytest.approx(np.sqrt(np.mean((c[t + h] - c[t] * np.exp(deriva * h)) ** 2)), rel=0.01), "deriva fijada con el entrenamiento"
    assert reentrenado["metricas"]["n_prueba"] == 312 and reentrenado["metricas"]["fecha_inicio_prueba"] == "2016-09-23"
    assert reentrenado["metricas"]["prueba_sube_x"] == 4.77


def test_main_escribe_el_artefacto_con_todo_lo_necesario(tmp_path, monkeypatch):
    shutil.copy(train.CARPETA / "dataset.csv", tmp_path / "dataset.csv")
    monkeypatch.setattr(train, "CARPETA", tmp_path)
    train.main()
    a = joblib.load(tmp_path / "modelo.joblib")
    assert a["reentrenado_con_todo"] is True and a["variables"] == CARACTERISTICAS and a["fecha_max"] == "2017-07-31"
    assert a["metricas"] == leer_metricas(CARPETA)["metricas"] and a["ultimos_cierres"] == cargar_artefacto(CARPETA, "bitcoin")["ultimos_cierres"]
    assert sorted(p.name for p in (tmp_path / "figuras").glob("*.png")) == sorted(
        ["autocorrelacion.png", "distribucion_retornos.png", "habilidad_por_horizonte.png", "prueba_1_dia.png", "prueba_7_dias.png", "serie_precio.png", "volatilidad.png"])
    assert (tmp_path / "metricas.json").exists()
