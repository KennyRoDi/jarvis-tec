"""Pruebas del modelo 08 · grasa corporal (specs/api_rest_spec.md §3). Usan el modelo entrenado real."""
import shutil

import joblib
import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from core.modelos import REGISTRO, cargar_artefacto, leer_metricas, predecir_con_pipeline
from features.modelo_08_grasa_corporal.router import CARPETA, Entrada
from features.modelo_08_grasa_corporal import train
from features.modelo_08_grasa_corporal.train import (OBJETIVO, SEMILLA, VARIABLES, candidatos, cargar_datos, entender, entrenar, explorar,
                                                     limpiar)

URL = "/api/modelos/grasa_corporal/predecir"

ENTRADA = {
    "age": 45, "weight_kg": 80.0, "height_cm": 178.0, "neck_cm": 38.0, "chest_cm": 100.0, "abdomen_cm": 92.0,
    "hip_cm": 99.0, "thigh_cm": 59.0, "knee_cm": 38.5, "ankle_cm": 23.0, "biceps_cm": 32.0,
    "forearm_cm": 28.7, "wrist_cm": 18.3,
}
ENTRADA_EXTREMA = {**ENTRADA, "weight_kg": 165.0, "chest_cm": 136.0, "abdomen_cm": 148.0, "hip_cm": 147.0, "thigh_cm": 87.0}
entrenado = pytest.mark.skipif(not REGISTRO["grasa_corporal"].entrenado,
                               reason="ejecutar features.modelo_08_grasa_corporal.train")


@pytest.fixture(scope="module")
def reentrenado():
    """Todo el entrenamiento (selección, evaluación, experimento de la fuga) en memoria, sin escribir nada."""
    return entrenar(limpiar(cargar_datos(), imprimir=False), figuras=None, imprimir=False)


@entrenado
def test_predecir(cliente):
    cuerpo = cliente.post("/api/modelos/grasa_corporal/predecir", json=ENTRADA).json()
    assert cuerpo["modelo"] == "grasa_corporal"
    assert 5 < cuerpo["prediccion"] < 35  # persona de medidas típicas: rango plausible
    assert cuerpo["unidad"] == "%"
    assert "poco confiable" not in cuerpo["texto"]


@entrenado
def test_prediccion_negativa_del_modelo_se_recorta_a_cero(cliente):
    """La esquina inferior del rango válido hace que Lasso prediga un valor negativo: la API debe devolver 0."""
    minimo = {**ENTRADA, "age": 18, "weight_kg": 45, "abdomen_cm": 65, "chest_cm": 75, "hip_cm": 80}
    crudo, _ = predecir_con_pipeline(CARPETA, "grasa_corporal", minimo)
    assert crudo < 0, "el caso ya no ejercita el recorte: elegir otra entrada"
    assert cliente.post("/api/modelos/grasa_corporal/predecir", json=minimo).json()["prediccion"] == 0.0


@entrenado
def test_medidas_extremas_avisan_que_el_resultado_es_poco_confiable(cliente):
    cuerpo = cliente.post("/api/modelos/grasa_corporal/predecir", json=ENTRADA_EXTREMA).json()
    assert "poco confiable" in cuerpo["texto"]


@entrenado
def test_abdomen_mayor_implica_mas_grasa(cliente):
    """Coherencia con lo explorado: el abdomen es la variable más correlacionada con la grasa."""
    menor = cliente.post("/api/modelos/grasa_corporal/predecir", json={**ENTRADA, "abdomen_cm": 80}).json()["prediccion"]
    mayor = cliente.post("/api/modelos/grasa_corporal/predecir", json={**ENTRADA, "abdomen_cm": 105}).json()["prediccion"]
    assert mayor > menor


def test_predecir_valida_entrada(cliente, assert_error):
    assert_error(cliente.post("/api/modelos/grasa_corporal/predecir", json={**ENTRADA, "age": 5}), 422, "VALIDACION")


def test_predecir_exige_todos_los_campos(cliente, assert_error):
    incompleta = {k: v for k, v in ENTRADA.items() if k != "wrist_cm"}
    assert_error(cliente.post("/api/modelos/grasa_corporal/predecir", json=incompleta), 422, "VALIDACION")


@entrenado
def test_info_incluye_metricas_y_esquema(cliente):
    cuerpo = cliente.get("/api/modelos/grasa_corporal/info").json()
    assert {"r2", "mae", "rmse"} <= set(cuerpo["metricas"])
    assert set(ENTRADA) == set(cuerpo["esquema_entrada"]["properties"])
    assert set(ENTRADA) == set(cuerpo["entrada_ejemplo"])
    assert isinstance(cuerpo["entrada_ejemplo"]["age"], int)  # el formulario recibe un entero, no 23.0


# --- Datos (no requieren el modelo entrenado) ---

def test_los_datos_cargados_estan_en_unidades_metricas_y_sin_nulos():
    df = cargar_datos()
    assert len(df) == 252 and df.isna().sum().sum() == 0
    assert set(VARIABLES + ["density", OBJETIVO]) == set(df.columns)
    crudo = pd.read_csv(CARPETA / "dataset.csv")
    assert df.weight_kg.iloc[0] == pytest.approx(crudo.Weight.iloc[0] * 0.45359237, abs=0.01)
    assert df.height_cm.iloc[0] == pytest.approx(crudo.Height.iloc[0] * 2.54, abs=0.05)
    assert df.abdomen_cm.iloc[0] == crudo.Abdomen.iloc[0], "las circunferencias ya venían en cm"


def test_se_descartan_solo_los_dos_registros_imposibles():
    df = cargar_datos()
    limpio = limpiar(df, imprimir=False)
    assert len(limpio) == 250
    assert df.loc[[41, 181], OBJETIVO].tolist()[1] == 0 and df.loc[41, "height_cm"] < 120, "grasa 0 % y 75 cm de estatura"
    assert (limpio[OBJETIVO] > 0).all() and (limpio.height_cm >= 120).all()
    assert limpio.weight_kg.max() > 160, "la persona de 165 kg (extremo posible) se conserva"


def test_density_no_es_variable_del_modelo_porque_el_objetivo_se_calcula_de_ella():
    assert "density" not in VARIABLES and len(VARIABLES) == 13
    df = limpiar(cargar_datos(), imprimir=False)
    assert df.density.corr(df[OBJETIVO]) < -0.98


def test_los_candidatos_son_una_linea_base_y_cuatro_modelos_con_el_preprocesamiento_en_el_pipeline():
    c = candidatos()
    assert list(c) == ["media (línea base)", "regresión lineal", "ridge", "lasso", "random forest"]
    assert isinstance(c["media (línea base)"].named_steps["modelo"], DummyRegressor)
    for nombre, clase in (("regresión lineal", LinearRegression), ("ridge", RidgeCV), ("lasso", LassoCV)):
        assert isinstance(c[nombre].named_steps["escala"], StandardScaler) and isinstance(c[nombre].named_steps["modelo"], clase)
    bosque = c["random forest"].named_steps["modelo"]
    assert isinstance(bosque, RandomForestRegressor) and "escala" not in c["random forest"].named_steps
    assert (bosque.n_estimators, bosque.min_samples_leaf, bosque.random_state) == (300, 2, SEMILLA)


def test_la_particion_es_80_20_con_la_semilla_fija():
    df = limpiar(cargar_datos(), imprimir=False)
    X_train, X_test, _, _ = train_test_split(df[VARIABLES], df[OBJETIVO], test_size=0.2, random_state=SEMILLA)
    assert (len(X_train), len(X_test)) == (200, 50) and SEMILLA == 42
    assert 38 in X_test.index, "la persona de 165 kg cae en la prueba (explica la brecha entre validación cruzada y prueba)"


def test_entender_y_explorar(capsys, tmp_path):
    df = limpiar(cargar_datos(), imprimir=False)
    entender(df)
    assert "Filas: 250  Columnas: 15" in capsys.readouterr().out
    explorar(df, tmp_path)
    assert sorted(p.name for p in tmp_path.glob("*.png")) == ["abdomen_vs_grasa.png", "correlacion.png", "distribucion_objetivo.png"]
    assert "abdomen_cm    0.81" in capsys.readouterr().out


def test_limpiar_informa_los_registros_descartados(capsys):
    limpiar(cargar_datos())
    assert "Registros imposibles descartados: 2 -> filas [41, 181]" in capsys.readouterr().out


# --- Entrada estricta, límites y descripciones ---

# Límites del contrato (literales, no leídos de `Entrada`: si cambian, hay que cambiarlos aquí a propósito).
LIMITES = {"age": (18, 90), "weight_kg": (45, 180), "height_cm": (145, 205), "neck_cm": (28, 55), "chest_cm": (75, 145),
           "abdomen_cm": (65, 155), "hip_cm": (80, 150), "thigh_cm": (43, 90), "knee_cm": (30, 52), "ankle_cm": (17, 36),
           "biceps_cm": (22, 48), "forearm_cm": (19, 37), "wrist_cm": (14, 24)}


def test_la_entrada_es_estricta_y_los_13_campos_son_obligatorios():
    esquema = Entrada.model_json_schema()
    assert esquema.get("additionalProperties") is False and set(esquema["required"]) == set(ENTRADA) == set(VARIABLES)


@pytest.mark.parametrize("cambio", [{"age": "45"}, {"age": 45.5}, {"age": True}, {"weight_kg": "80"}, {"weight_kg": True}, {"wrist_cm": None},
                                    {"densidad": 1.05}, {"density": 1.05}, {"bodyfat": 20}])
def test_tipos_laxos_y_campos_desconocidos_dan_422(cliente, assert_error, cambio):
    assert_error(cliente.post(URL, json={**ENTRADA, **cambio}), 422, "VALIDACION")


@pytest.mark.parametrize("campo", list(LIMITES))
def test_el_limite_de_cada_campo_es_inclusivo_y_rechaza_lo_que_lo_excede(campo):
    minimo, maximo = LIMITES[campo]
    tipo = int if campo == "age" else float
    Entrada(**{**ENTRADA, campo: tipo(minimo)})
    Entrada(**{**ENTRADA, campo: tipo(maximo)})
    for fuera in (minimo - 1, maximo + 1):
        with pytest.raises(ValidationError):
            Entrada(**{**ENTRADA, campo: tipo(fuera)})


def test_los_limites_contienen_a_todas_las_personas_reales_salvo_los_registros_imposibles():
    df = limpiar(cargar_datos(), imprimir=False)
    for campo, (minimo, maximo) in LIMITES.items():
        assert df[campo].min() >= minimo and df[campo].max() <= maximo, campo


@pytest.mark.parametrize("campo", list(LIMITES))
def test_cada_campo_tiene_descripcion_con_unidad_y_un_ejemplo_valido(campo):
    info = Entrada.model_fields[campo]
    assert info.description and info.examples and LIMITES[campo][0] <= info.examples[0] <= LIMITES[campo][1]
    if campo not in ("age",):
        assert "kg" in info.description or "cm" in info.description


# --- Modelo entrenado y API ---

@entrenado
def test_el_modelo_elegido_y_las_metricas_publicadas():
    m = leer_metricas(CARPETA)["metricas"]
    assert m["modelo"] == "lasso" and (m["n_entrenamiento"], m["n_prueba"]) == (200, 50)
    assert m["comparacion_cv"]["lasso"]["cv_rmse"] == min(v["cv_rmse"] for n, v in m["comparacion_cv"].items() if "línea base" not in n)
    assert m["r2"] > m["baseline_media"]["r2"] + 0.5 and m["rmse"] < m["baseline_media"]["rmse"]
    assert m["r2_con_density_fuga"] > 0.95, "con Density el R² sería engañoso (fuga de información)"


@entrenado
def test_el_artefacto_guarda_el_rango_de_entrenamiento_de_las_13_variables():
    a = cargar_artefacto(CARPETA, "grasa_corporal")
    assert set(a["rango"]) == set(VARIABLES)
    train, _, _, _ = train_test_split(limpiar(cargar_datos(), imprimir=False)[VARIABLES], limpiar(cargar_datos(), imprimir=False)[OBJETIVO], test_size=0.2, random_state=SEMILLA)
    for campo, (minimo, maximo) in a["rango"].items():
        assert (minimo, maximo) == (train[campo].min(), train[campo].max()), campo
    assert a["rango"]["weight_kg"][1] < 125, "el entrenamiento no vio a la persona de 165 kg"


@entrenado
def test_una_sola_medida_fuera_del_rango_basta_para_avisar(cliente):
    rango = cargar_artefacto(CARPETA, "grasa_corporal")["rango"]
    for campo in ("weight_kg", "abdomen_cm", "wrist_cm"):
        maximo = rango[campo][1]
        dentro = cliente.post(URL, json={**ENTRADA, campo: maximo}).json()["texto"]
        assert "poco confiable" not in dentro, campo
        lejos = cliente.post(URL, json={**ENTRADA, campo: min(maximo + (maximo - rango[campo][0]) * 0.2, LIMITES[campo][1])}).json()["texto"]
        assert "poco confiable" in lejos, campo


@entrenado
def test_la_respuesta_redondea_a_un_decimal_y_no_modifica_la_entrada(cliente):
    cuerpo = cliente.post(URL, json=ENTRADA).json()
    assert cuerpo["prediccion"] == round(cuerpo["prediccion"], 1)
    crudo, _ = predecir_con_pipeline(CARPETA, "grasa_corporal", ENTRADA)
    assert cuerpo["prediccion"] == round(max(crudo, 0.0), 1) and str(cuerpo["prediccion"]) in cuerpo["texto"]
    assert "hombres adultos" in cuerpo["texto"] and "orientativa" in cuerpo["texto"]


@entrenado
def test_los_metadatos_del_modelo_registrado():
    info = REGISTRO["grasa_corporal"].info
    assert info["slug"] == "grasa_corporal" and info["unidad"] == "%" and info["tipo"] == "regresion"
    assert info["comandos"] == ["grasa corporal", "masa corporal", "porcentaje de grasa"] and REGISTRO["grasa_corporal"].entrada is Entrada


@entrenado
def test_el_artefacto_no_es_excesivo():
    assert (CARPETA / "modelo.joblib").stat().st_size < 5_000_000


# --- Entrenamiento (sin tocar el disco) ---

def test_entrenar_devuelve_todo_lo_que_despues_se_publica(reentrenado):
    assert set(reentrenado) == {"pipeline", "metricas", "rango", "entrada_ejemplo"}
    assert isinstance(reentrenado["entrada_ejemplo"]["age"], (int, np.integer))


def test_el_modelo_elegido_nunca_es_la_linea_base(reentrenado):
    assert reentrenado["metricas"]["modelo"] != "media (línea base)"


@pytest.mark.reproduce
@entrenado
def test_reentrenar_reproduce_exactamente_lo_publicado(reentrenado):
    """Cubre todo `entrenar()`: selección por validación cruzada, evaluación, línea base y el experimento de la fuga."""
    publicado = leer_metricas(CARPETA)
    artefacto = cargar_artefacto(CARPETA, "grasa_corporal")
    assert reentrenado["metricas"] == publicado["metricas"]
    assert reentrenado["entrada_ejemplo"] == publicado["entrada_ejemplo"]
    assert reentrenado["rango"] == artefacto["rango"]
    df = limpiar(cargar_datos(), imprimir=False)
    _, X_test, _, _ = train_test_split(df[VARIABLES], df[OBJETIVO], test_size=0.2, random_state=SEMILLA)
    assert reentrenado["pipeline"].predict(X_test) == pytest.approx(artefacto["pipeline"].predict(X_test), abs=1e-9)


def test_main_escribe_el_artefacto_las_metricas_y_las_figuras(tmp_path, monkeypatch, capsys):
    shutil.copy(train.CARPETA / "dataset.csv", tmp_path / "dataset.csv")
    monkeypatch.setattr(train, "CARPETA", tmp_path)
    train.main()
    a = joblib.load(tmp_path / "modelo.joblib")
    assert a["variables"] == VARIABLES and set(a["rango"]) == set(VARIABLES) and (tmp_path / "metricas.json").exists()
    assert sorted(p.name for p in (tmp_path / "figuras").glob("*.png")) == ["abdomen_vs_grasa.png", "correlacion.png", "distribucion_objetivo.png", "real_vs_predicho.png"]
    salida = capsys.readouterr().out
    assert "Modelo elegido: lasso" in salida and "Filas: 250  Columnas: 15" in salida
    if REGISTRO["grasa_corporal"].entrenado:
        assert a["metricas"] == leer_metricas(CARPETA)["metricas"]
