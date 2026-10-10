"""Pruebas del modelo 09 · aguacate (specs/api_rest_spec.md §3). Usan el modelo entrenado real."""
import shutil
from datetime import date, timedelta

import joblib
import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.preprocessing import StandardScaler

from core.modelos import REGISTRO, cargar_artefacto, leer_metricas, predecir_con_pipeline
from features.modelo_09_aguacate import train
from features.modelo_09_aguacate.preprocesamiento import CaracteristicasFecha
from features.modelo_09_aguacate.router import CARPETA, DIAS_DE_GRACIA, MESES, REGIONES, Entrada, fecha_en_texto, nombre_region
from features.modelo_09_aguacate.train import (OBJETIVO, SEMILLA, VARIABLES, candidatos, cargar_datos, cv_por_semanas, entender, entrenar,
                                               explorar, particion_temporal)

URL = "/api/modelos/aguacate/predecir"
ENTRADA = {"region": "TotalUS", "tipo": "conventional", "fecha": "2017-09-15"}  # dentro del rango del dataset
entrenado = pytest.mark.skipif(not REGISTRO["aguacate"].entrenado, reason="ejecutar features.modelo_09_aguacate.train")


@pytest.fixture(scope="module")
def reentrenado():
    """Todo el entrenamiento (selección, evaluación, reentrenamiento final) en memoria, sin escribir nada."""
    return entrenar(cargar_datos(), figuras=None, imprimir=False)


def precio(cliente, **cambios) -> float:
    return cliente.post(URL, json={**ENTRADA, **cambios}).json()["prediccion"]


def test_las_regiones_del_router_coinciden_con_el_dataset():
    del_dataset = set(pd.read_csv(CARPETA / "dataset.csv", usecols=["region"]).region)
    assert set(REGIONES) == del_dataset and len(REGIONES) == 54


def test_nombre_legible_de_la_region():
    assert nombre_region("LosAngeles") == "Los Angeles"
    assert nombre_region("TotalUS") == "todo Estados Unidos"
    assert nombre_region("DallasFtWorth") == "Dallas Ft. Worth"
    assert nombre_region("StLouis") == "St. Louis"


# --- Lógica anti-fuga temporal (no requiere el modelo entrenado) ---

def test_la_prueba_es_posterior_a_todo_el_entrenamiento():
    train, test, corte = particion_temporal(cargar_datos())
    assert train.fecha.max() < corte <= test.fecha.min()
    assert not set(train.fecha) & set(test.fecha)
    assert (train.fecha.nunique(), test.fecha.nunique()) == (135, 34)
    assert (test.groupby("fecha").size() == 108).all()  # semanas completas, ninguna partida


def test_la_validacion_cruzada_tiene_ventana_creciente_sin_solapamiento():
    train, _, _ = particion_temporal(cargar_datos())
    pliegues = cv_por_semanas(train)
    assert len(pliegues) == 3
    previo = 0
    for entrena, valida in pliegues:
        fe, fv = train.fecha.iloc[entrena], train.fecha.iloc[valida]
        assert fe.max() < fv.min(), "la validación debe ser posterior al entrenamiento de su pliegue"
        assert not set(fe) & set(fv)
        assert fe.nunique() > previo, "la ventana de entrenamiento debe crecer"
        assert fv.nunique() == 20
        previo = fe.nunique()


def test_transformador_de_fechas():
    entrenamiento = pd.DataFrame({"region": ["TotalUS"] * 2, "tipo": ["organic"] * 2,
                                  "fecha": pd.to_datetime(["2015-01-04", "2017-07-30"])})
    t = CaracteristicasFecha().fit(entrenamiento)
    nuevas = pd.DataFrame({"region": ["TotalUS"] * 3, "tipo": ["organic"] * 3,
                           "fecha": pd.to_datetime(["2015-01-04", "2017-07-30", "2030-12-31"])})
    out = t.transform(nuevas)
    assert list(out.columns) == ["region", "tipo", "mes", "semana", "t"]
    assert out.t.iloc[0] == 0 and out.t.iloc[2] == out.t.iloc[1], "t no debe extrapolarse más allá de lo visto"
    assert out.mes.tolist() == [1, 7, 12] and out.semana.max() <= 52


def test_el_cambio_de_anio_no_produce_saltos_dentro_de_la_misma_semana(cliente):
    """Con la semana ISO, el 31-dic caía en la semana 1 con mes 12 y la predicción saltaba hasta 0.26 USD."""
    if not REGISTRO["aguacate"].entrenado:
        pytest.skip("ejecutar features.modelo_09_aguacate.train")
    for region in REGIONES:
        a = predecir_con_pipeline(CARPETA, "aguacate", {"region": region, "tipo": "organic", "fecha": pd.Timestamp("2018-12-30")})[0]
        b = predecir_con_pipeline(CARPETA, "aguacate", {"region": region, "tipo": "organic", "fecha": pd.Timestamp("2018-12-31")})[0]
        assert a == b, region


@entrenado
def test_predecir_dentro_del_rango_no_avisa(cliente):
    cuerpo = cliente.post(URL, json=ENTRADA).json()
    assert cuerpo["modelo"] == "aguacate" and cuerpo["unidad"] == "USD por aguacate"
    assert 0.5 < cuerpo["prediccion"] < 3.5
    assert "15 de septiembre de 2017" in cuerpo["texto"] and "poco confiable" not in cuerpo["texto"]


@entrenado
def test_solicitud_vacia_es_valida_y_usa_los_valores_por_defecto(cliente):
    """Permite ejecutarlo solo con la voz ("precio del aguacate"). Hoy está lejos de los datos: debe avisar."""
    respuesta = cliente.post(URL, json={})
    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert "todo Estados Unidos" in cuerpo["texto"] and "convencional" in cuerpo["texto"]
    assert "poco confiable" in cuerpo["texto"]


@entrenado
def test_el_organico_cuesta_mas_que_el_convencional(cliente):
    assert precio(cliente, tipo="organic") > precio(cliente, tipo="conventional")


@entrenado
def test_las_regiones_se_distinguen(cliente):
    """San Francisco es una de las regiones más caras; Houston, de las más baratas (ver figuras/regiones_extremas.png)."""
    assert precio(cliente, region="SanFrancisco") > precio(cliente, region="Houston")


@entrenado
def test_septiembre_es_mas_caro_que_febrero(cliente):
    """Estacionalidad observada en la exploración: el precio medio sube hacia septiembre-octubre."""
    assert precio(cliente, fecha="2017-09-15") > precio(cliente, fecha="2017-02-15")


@entrenado
def test_fecha_futura_avisa_y_mantiene_la_estacionalidad_sin_extrapolar(cliente):
    futura = cliente.post(URL, json={**ENTRADA, "fecha": "2030-09-15"}).json()
    assert "poco confiable" in futura["texto"]
    assert futura["prediccion"] == precio(cliente, fecha="2025-09-15"), "no debe derivar con el paso de los años"
    assert futura["prediccion"] > precio(cliente, fecha="2030-02-15"), "conserva la estacionalidad"


@entrenado
def test_el_aviso_aparece_pasados_60_dias_del_ultimo_dato(cliente):
    """El último dato es el 25-mar-2018: el 24-may son 60 días (sin aviso) y el 25-may, 61 (con aviso)."""
    assert "poco confiable" not in cliente.post(URL, json={**ENTRADA, "fecha": "2018-05-24"}).json()["texto"]
    assert "poco confiable" in cliente.post(URL, json={**ENTRADA, "fecha": "2018-05-25"}).json()["texto"]


@entrenado
def test_sentido_del_modelo_en_todas_las_regiones():
    """Muestra amplia (no un solo punto): el orgánico cuesta más y septiembre es más caro que febrero."""
    def p(region, tipo, fecha):
        return predecir_con_pipeline(CARPETA, "aguacate", {"region": region, "tipo": tipo, "fecha": pd.Timestamp(fecha)})[0]
    organico_mayor = np.mean([p(r, "organic", "2017-09-15") > p(r, "conventional", "2017-09-15") for r in REGIONES])
    sep_mayor_feb = np.mean([p(r, t, "2017-09-15") > p(r, t, "2017-02-15") for r in REGIONES for t in ("conventional", "organic")])
    assert organico_mayor >= 0.95 and sep_mayor_feb >= 0.85


@pytest.mark.parametrize("cambio", [{"region": "Narnia"}, {"tipo": "frito"}, {"fecha": "ayer"}, {"fecha": "1999-01-01"}])
def test_entradas_invalidas_dan_422(cliente, assert_error, cambio):
    assert_error(cliente.post(URL, json={**ENTRADA, **cambio}), 422, "VALIDACION")


@entrenado
def test_info_incluye_metricas_y_esquema_con_las_54_regiones(cliente):
    cuerpo = cliente.get("/api/modelos/aguacate/info").json()
    assert {"r2", "mae", "rmse"} <= set(cuerpo["metricas"])
    assert len(cuerpo["esquema_entrada"]["properties"]["region"]["enum"]) == 54
    assert not cuerpo["esquema_entrada"].get("required")  # todos con valor por defecto


@entrenado
def test_el_artefacto_no_es_excesivo():
    """El joblib se versiona en git: vigilar que no crezca sin control."""
    assert (CARPETA / "modelo.joblib").stat().st_size < 5_000_000


# --- Datos y candidatos (no requieren el modelo entrenado) ---

def test_los_datos_cargados_son_semanales_ordenados_y_sin_volumenes():
    df = cargar_datos()
    assert len(df) == 18249 and list(df.columns) == ["fecha", "region", "tipo", OBJETIVO]
    assert df.fecha.nunique() == 169 and df.region.nunique() == 54 and set(df.tipo) == {"conventional", "organic"}
    assert df.fecha.min() == pd.Timestamp("2015-01-04") and df.fecha.max() == pd.Timestamp("2018-03-25")
    assert df.isna().sum().sum() == 0 and not df.duplicated(["fecha", "region", "tipo"]).any()
    assert df.equals(df.sort_values(["fecha", "region", "tipo"]).reset_index(drop=True)), "ordenado por fecha, región y tipo"
    assert VARIABLES == ["region", "tipo", "fecha"], "los volúmenes de venta no se conocen al consultar"


def test_los_candidatos_son_una_linea_base_y_seis_variantes_con_y_sin_tendencia():
    c = candidatos()
    assert list(c) == ["efecto región y tipo (línea base)", "ridge estacional", "ridge estacional + tendencia", "random forest",
                       "random forest + tendencia", "gradient boosting", "gradient boosting + tendencia"]
    assert isinstance(c["efecto región y tipo (línea base)"].named_steps["modelo"], LinearRegression)
    assert isinstance(c["ridge estacional"].named_steps["modelo"], RidgeCV)
    for nombre, pipe in c.items():
        assert isinstance(pipe.named_steps["fecha"], CaracteristicasFecha), "las fechas se transforman dentro del Pipeline"
        columnas = [t for t in pipe.named_steps["pre"].transformers if t[0] == "num"][0]
        numericas = columnas[2] if columnas[1] != "drop" else []
        assert ("t" in numericas) == ("tendencia" in nombre), nombre
    escala_ridge = [t for t in c["ridge estacional + tendencia"].named_steps["pre"].transformers if t[0] == "num"][0][1]
    assert isinstance(escala_ridge, StandardScaler), "la tendencia se estandariza para el ridge (los árboles no la necesitan)"
    boosting = c["gradient boosting"].named_steps["modelo"]
    assert isinstance(boosting, HistGradientBoostingRegressor) and boosting.early_stopping is False and boosting.random_state == SEMILLA
    bosque = c["random forest"].named_steps["modelo"]
    assert isinstance(bosque, RandomForestRegressor) and (bosque.n_estimators, bosque.min_samples_leaf, bosque.max_depth) == (100, 10, 14)


def test_la_tendencia_no_entra_a_las_variantes_estacionales_y_la_semana_solo_a_los_arboles():
    c = candidatos()
    num = lambda n: [t for t in c[n].named_steps["pre"].transformers if t[0] == "num"][0][2]  # noqa: E731
    assert num("random forest") == ["mes", "semana"] and num("gradient boosting + tendencia") == ["t", "mes", "semana"]
    assert num("ridge estacional + tendencia") == ["t"]


def test_la_particion_temporal_usa_el_20_por_ciento_final_de_las_fechas():
    df = cargar_datos()
    train_, test_, corte = particion_temporal(df)
    assert corte == pd.Timestamp("2017-08-06") and len(train_) + len(test_) == len(df) and (len(train_), len(test_)) == (14577, 3672)


def test_entender_y_explorar(capsys, tmp_path):
    df = cargar_datos()
    entender(df)
    salida = capsys.readouterr().out
    assert "Semanas: 169  Regiones: 54" in salida and "Rango: 2015-01-04 a 2018-03-25" in salida
    explorar(df, tmp_path, pd.Timestamp("2017-08-06"))
    assert sorted(p.name for p in tmp_path.glob("*.png")) == ["distribucion_objetivo.png", "estacionalidad.png", "regiones_extremas.png", "serie_nacional.png"]
    assert "Precio medio por tipo: {'conventional': 1.16, 'organic': 1.65}" in capsys.readouterr().out


# --- Entrada estricta, valores por defecto y textos ---

def test_la_entrada_es_estricta_y_todos_los_campos_tienen_valor_por_defecto():
    esquema = Entrada.model_json_schema()
    assert esquema.get("additionalProperties") is False and "required" not in esquema
    e = Entrada()
    assert (e.region, e.tipo, e.fecha) == ("TotalUS", "conventional", date.today())
    assert all(i.description for i in Entrada.model_fields.values())


@pytest.mark.parametrize("cambio", [{"region": "totalus"}, {"region": 1}, {"tipo": "Organic"}, {"tipo": None}, {"fecha": 20170915}, {"fecha": 1499990400}, {"fecha": 1499990400.0}, {"fecha": "2017-09-15T00:00:00"},
                                    {"fecha": True}, {"fecha": None}, {"fecha": "2017-09-15T10:00:00"}, {"fecha": "15/09/2017"},
                                    {"fecha": "2017-02-30"}, {"fecha": "2014-12-31"}, {"fecha": "2101-01-01"}, {"anio": 2017}])
def test_entradas_laxas_o_desconocidas_dan_422(cliente, assert_error, cambio):
    assert_error(cliente.post(URL, json={**ENTRADA, **cambio}), 422, "VALIDACION")


def test_los_limites_de_fecha_son_inclusivos():
    assert Entrada(fecha="2015-01-01").fecha == date(2015, 1, 1) and Entrada(fecha="2100-12-31").fecha == date(2100, 12, 31)
    for fuera in ("2014-12-31", "2101-01-01"):
        with pytest.raises(ValidationError):
            Entrada(fecha=fuera)


def test_la_entrada_acepta_un_objeto_date_para_uso_interno_pero_no_otros_tipos():
    assert Entrada(fecha=date(2017, 9, 15)).fecha == date(2017, 9, 15)
    for malo in (20170915, 1499990400, 2017.9, [2017, 9, 15], True):
        with pytest.raises(ValidationError):
            Entrada(fecha=malo)


def test_los_doce_meses_y_el_formato_de_fecha():
    assert [fecha_en_texto(date(2018, m, 1)) for m in range(1, 13)] == [f"1 de {mes} de 2018" for mes in MESES]
    assert MESES == ("enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre")
    assert fecha_en_texto(date(2017, 9, 15)) == "15 de septiembre de 2017" and DIAS_DE_GRACIA == 60


# --- Modelo entrenado y API ---

@entrenado
def test_el_artefacto_guarda_la_ultima_fecha_las_variables_y_la_marca_de_reentrenado():
    a = cargar_artefacto(CARPETA, "aguacate")
    assert a["fecha_max"] == "2018-03-25" and a["variables"] == VARIABLES and a["reentrenado_con_todo"] is True


@entrenado
def test_el_modelo_elegido_las_metricas_y_la_linea_base_publicadas():
    m = leer_metricas(CARPETA)["metricas"]
    assert m["modelo"] == "gradient boosting + tendencia" and (m["n_entrenamiento"], m["n_prueba"]) == (14577, 3672)
    cv = m["comparacion_cv"]
    assert cv[m["modelo"]]["cv_rmse"] == min(v["cv_rmse"] for n, v in cv.items() if "línea base" not in n)
    assert m["rmse"] < m["baseline"]["rmse"] and m["r2"] > m["baseline"]["r2"]


@entrenado
def test_la_respuesta_redondea_a_dos_decimales_y_nunca_es_negativa(cliente):
    for region in ("TotalUS", "Houston", "SanFrancisco"):
        valor = cliente.post(URL, json={**ENTRADA, "region": region}).json()["prediccion"]
        assert valor == round(valor, 2) and valor >= 0
    cuerpo = cliente.post(URL, json=ENTRADA).json()
    crudo, _ = predecir_con_pipeline(CARPETA, "aguacate", {"region": "TotalUS", "tipo": "conventional", "fecha": pd.Timestamp("2017-09-15")})
    assert abs(crudo - round(crudo, 2)) > 1e-9 and cuerpo["prediccion"] == round(max(crudo, 0.0), 2)
    assert f"es de {cuerpo['prediccion']} dólares por unidad" in cuerpo["texto"]


@entrenado
def test_ninguna_prediccion_cruda_es_negativa_asi_que_el_recorte_a_cero_es_solo_defensivo():
    """Imposibilidad explícita: con las 54 regiones, los dos tipos y fechas de todo el año el modelo no predice precios negativos."""
    fechas = pd.date_range("2018-01-01", "2018-12-31", freq="21D")
    crudos = [predecir_con_pipeline(CARPETA, "aguacate", {"region": r, "tipo": t, "fecha": f})[0] for r in REGIONES for t in ("conventional", "organic") for f in fechas]
    assert min(crudos) > 0.3


@entrenado
def test_el_texto_nombra_el_tipo_y_la_region(cliente):
    organico = cliente.post(URL, json={**ENTRADA, "tipo": "organic", "region": "LosAngeles"}).json()["texto"]
    assert "aguacate orgánico en Los Angeles" in organico
    convencional = cliente.post(URL, json={**ENTRADA, "tipo": "conventional"}).json()["texto"]
    assert "aguacate convencional en todo Estados Unidos" in convencional


@entrenado
def test_la_fecha_por_defecto_es_hoy_y_avisa_porque_los_datos_son_viejos(cliente):
    hoy = cliente.post(URL, json={}).json()["texto"]
    assert fecha_en_texto(date.today()) in hoy and "25 de marzo de 2018" in hoy


@entrenado
def test_el_aviso_menciona_la_ultima_fecha_de_los_datos(cliente):
    texto = cliente.post(URL, json={**ENTRADA, "fecha": "2030-01-01"}).json()["texto"]
    assert "hasta el 25 de marzo de 2018" in texto and "estacionalidad" in texto


@entrenado
def test_los_metadatos_del_modelo_registrado():
    info = REGISTRO["aguacate"].info
    assert info["slug"] == "aguacate" and info["unidad"] == "USD por aguacate" and info["tipo"] == "regresion"
    assert info["comandos"] == ["precio del aguacate", "precio de aguacate", "cuanto cuesta el aguacate"] and REGISTRO["aguacate"].entrada is Entrada


# --- Entrenamiento (sin tocar el disco) ---

def test_entrenar_devuelve_todo_lo_que_despues_se_publica(reentrenado):
    assert set(reentrenado) == {"pipeline", "metricas", "fecha_max", "entrada_ejemplo"}
    assert reentrenado["fecha_max"] == "2018-03-25" and reentrenado["entrada_ejemplo"] == {"region": "TotalUS", "tipo": "conventional", "fecha": "2018-03-25"}


def test_el_modelo_final_se_reentrena_con_todas_las_semanas(reentrenado):
    df = cargar_datos()
    esperado = candidatos()[reentrenado["metricas"]["modelo"]].fit(df[VARIABLES], df[OBJETIVO])
    assert reentrenado["pipeline"].predict(df[VARIABLES].head(500)) == pytest.approx(esperado.predict(df[VARIABLES].head(500)))
    assert reentrenado["pipeline"].named_steps["fecha"].t_max_ == pytest.approx(esperado.named_steps["fecha"].t_max_)
    assert reentrenado["pipeline"].named_steps["fecha"].t_max_ > 3.2, "conoce hasta la última semana del dataset (marzo de 2018)"


def test_el_entrenamiento_no_usa_la_prueba_ni_para_elegir_ni_para_ajustar(reentrenado):
    """Se inflan los precios de las semanas de prueba: la selección por validación cruzada y la línea base deben quedar idénticas."""
    df = cargar_datos()
    _, _, corte = particion_temporal(df)
    alterado = df.copy()
    alterado.loc[alterado.fecha >= corte, OBJETIVO] *= 5
    otro = entrenar(alterado, figuras=None, imprimir=False)
    assert otro["metricas"]["comparacion_cv"] == reentrenado["metricas"]["comparacion_cv"] and otro["metricas"]["modelo"] == reentrenado["metricas"]["modelo"]
    assert otro["metricas"]["rmse"] != reentrenado["metricas"]["rmse"]


def test_main_escribe_el_artefacto_las_metricas_y_las_figuras(tmp_path, monkeypatch, capsys):
    shutil.copy(train.CARPETA / "dataset.csv", tmp_path / "dataset.csv")
    monkeypatch.setattr(train, "CARPETA", tmp_path)
    train.main()
    a = joblib.load(tmp_path / "modelo.joblib")
    assert a["reentrenado_con_todo"] is True and a["fecha_max"] == "2018-03-25" and a["variables"] == VARIABLES
    assert sorted(p.name for p in (tmp_path / "figuras").glob("*.png")) == [
        "distribucion_objetivo.png", "estacionalidad.png", "prueba_nacional.png", "real_vs_predicho.png", "regiones_extremas.png", "serie_nacional.png"]
    salida = capsys.readouterr().out
    assert "Modelo elegido: gradient boosting + tendencia" in salida and "Semanas: 169" in salida


@pytest.mark.reproduce
@entrenado
def test_reentrenar_reproduce_exactamente_lo_publicado(reentrenado):
    """Cubre todo `entrenar()`: selección por validación cruzada de ventana creciente, evaluación temporal y reentrenamiento."""
    publicado = leer_metricas(CARPETA)
    artefacto = cargar_artefacto(CARPETA, "aguacate")
    assert reentrenado["metricas"] == publicado["metricas"] and reentrenado["entrada_ejemplo"] == publicado["entrada_ejemplo"]
    assert reentrenado["fecha_max"] == artefacto["fecha_max"]
    muestra = cargar_datos()[VARIABLES].iloc[::97]
    assert reentrenado["pipeline"].predict(muestra) == pytest.approx(artefacto["pipeline"].predict(muestra), abs=1e-9)
