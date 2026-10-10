"""Pruebas del modelo 09 · aguacate (specs/api_rest_spec.md §3). Usan el modelo entrenado real."""
import numpy as np
import pandas as pd
import pytest

from core.modelos import REGISTRO, predecir_con_pipeline
from features.modelo_09_aguacate.preprocesamiento import CaracteristicasFecha
from features.modelo_09_aguacate.router import CARPETA, REGIONES, nombre_region
from features.modelo_09_aguacate.train import cargar_datos, cv_por_semanas, particion_temporal

URL = "/api/modelos/aguacate/predecir"
ENTRADA = {"region": "TotalUS", "tipo": "conventional", "fecha": "2017-09-15"}  # dentro del rango del dataset
entrenado = pytest.mark.skipif(not REGISTRO["aguacate"].entrenado, reason="ejecutar features.modelo_09_aguacate.train")


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
