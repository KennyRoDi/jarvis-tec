"""Pruebas del modelo 06 · hepatitis C (specs/api_rest_spec.md §3). Usan el modelo entrenado real."""
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from core.modelos import REGISTRO, cargar_artefacto, leer_metricas, predecir_con_pipeline
from features.modelo_06_hepatitis.router import CARPETA, ORDEN, Entrada
from features.modelo_06_hepatitis.train import (CATEGORICAS, ENFERMEDAD, LABORATORIO, NUMERICAS, OBJETIVO, VARIABLES, candidatos,
                                                cargar_datos, dividir, entrenar, f1_macro_de, por_clase, sensibilidad_y_especificidad)

URL = "/api/modelos/hepatitis/predecir"
ENTRADA = {"age": 47, "sex": "m", "alb": 42.0, "alt": 23.0, "ast": 25.0, "bil": 7.3, "che": 8.3, "chol": 5.3,
           "crea": 77.0, "ggt": 23.0, "prot": 72.0}
entrenado = pytest.mark.skipif(not REGISTRO["hepatitis"].entrenado, reason="ejecutar features.modelo_06_hepatitis.train")


@pytest.fixture(scope="module")
def reentrenado():
    """Todo el entrenamiento (selección, evaluación, bootstrap, experimentos) en memoria, sin escribir nada."""
    return entrenar(cargar_datos(), figuras=None, imprimir=False)


def filas_completas(n=300, semilla=0):
    return cargar_datos().dropna(subset=VARIABLES).sample(n, random_state=semilla)[VARIABLES].to_dict("records")


def puntajes_lote(filas: list[dict]) -> pd.DataFrame:
    pipeline = cargar_artefacto(CARPETA, "hepatitis")["pipeline"]
    return pd.DataFrame(pipeline.predict_proba(pd.DataFrame(filas)), columns=pipeline.classes_)


# --- Preparación de datos (no requiere el modelo entrenado) ---

def test_los_datos_cargados():
    df = cargar_datos()
    assert len(df) == 608 and df[OBJETIVO].value_counts().to_dict() == {"donante": 533, "cirrosis": 30, "hepatitis": 24, "fibrosis": 21}
    assert "unnamed: 0" not in df.columns and set(df[OBJETIVO]) == set(ORDEN)


def test_la_clase_ambigua_de_donante_sospechoso_se_descarta():
    assert len(pd.read_csv(CARPETA / "dataset.csv")) == 615 and len(cargar_datos()) == 615 - 7


def test_el_alp_se_excluye_porque_sus_vacios_estan_todos_en_pacientes():
    """Artefacto de la recolección: ninguna de las 18 filas con ALP vacía es de un donante."""
    df = cargar_datos()
    assert df.alp.isna().sum() == 18 and not df[df.alp.isna()][OBJETIVO].eq("donante").any()
    assert "alp" not in VARIABLES and "alp_faltante" not in VARIABLES and len(VARIABLES) == 11


def test_la_particion_es_estratificada_y_conserva_las_clases_raras():
    train, test, y_train, y_test = dividir(cargar_datos())
    assert (len(train), len(test)) == (486, 122) and not set(train.index) & set(test.index)
    assert y_train.value_counts().to_dict() == {"donante": 426, "cirrosis": 24, "hepatitis": 19, "fibrosis": 17}
    assert y_test.value_counts().to_dict() == {"donante": 107, "cirrosis": 6, "hepatitis": 5, "fibrosis": 4}


def test_los_candidatos_imputan_con_la_mediana_y_ponderan_las_clases_raras():
    for nombre, pipeline in candidatos().items():
        transformadores = {n: (t, c) for n, t, c in pipeline.named_steps["pre"].transformers}
        assert transformadores["num"][0].named_steps["imputar"].strategy == "median", nombre
        assert transformadores["num"][1] == NUMERICAS and transformadores["cat"][1] == CATEGORICAS, nombre
    c = {n: p.named_steps["modelo"] for n, p in candidatos().items()}
    assert isinstance(c["clase mayoritaria (línea base)"], DummyClassifier)
    assert isinstance(c["regresión logística"], LogisticRegression) and c["regresión logística"].class_weight == "balanced"
    assert isinstance(c["random forest"], RandomForestClassifier) and c["random forest"].class_weight == "balanced_subsample"
    assert "svm rbf" not in c, "SVC(probability=True) está deprecado en scikit-learn 1.9"


def test_sensibilidad_y_especificidad_de_la_vista_binaria():
    y = ["donante", "donante", "donante", "hepatitis", "fibrosis", "cirrosis"]
    pred = ["donante", "hepatitis", "donante", "donante", "fibrosis", "cirrosis"]   # 1 falso positivo, 1 falso negativo
    assert sensibilidad_y_especificidad(y, pred) == {"sensibilidad": 0.6667, "especificidad": 0.6667}
    assert ENFERMEDAD == ["hepatitis", "fibrosis", "cirrosis"]


def test_las_metricas_por_clase_y_el_f1_macro_usan_el_orden_y_las_cuatro_clases():
    y = ["donante"] * 4 + ["hepatitis", "cirrosis"]
    pred = ["donante"] * 4 + ["donante", "cirrosis"]
    assert por_clase(y, pred)["hepatitis"]["recall"] == 0.0 and por_clase(y, pred)["cirrosis"]["recall"] == 1.0
    assert list(por_clase(y, pred)) == list(ORDEN)
    # una clase ausente (fibrosis) cuenta como 0 en el promedio macro: (F1 donante + 0 + 0 + 1) / 4
    assert f1_macro_de(y, pred) == pytest.approx((0.8889 + 0 + 0 + 1.0) / 4, abs=1e-3)


# --- Validación de la entrada (no requiere el modelo entrenado) ---

def test_la_entrada_acepta_a_todas_las_personas_reales_con_valores_completos():
    filas = filas_completas(n=590)
    assert len(filas) > 580
    for fila in filas:
        Entrada(**fila)


def test_los_limites_de_entrada_son_razonables_frente_a_los_datos():
    df = cargar_datos()
    propiedades = Entrada.model_json_schema()["properties"]
    for campo in NUMERICAS:
        minimo, maximo = df[campo].min(), df[campo].max()
        inferior = 0 if minimo < 1 else 0.4 * minimo
        assert inferior <= propiedades[campo]["minimum"] <= minimo, campo
        assert maximo <= propiedades[campo]["maximum"] <= 1.8 * maximo, campo


@pytest.mark.parametrize("cambio", [
    {"age": 17}, {"age": 101}, {"age": 47.5}, {"age": "47"}, {"sex": "x"}, {"sex": "M"}, {"alb": 5}, {"alt": 5000}, {"ast": "alto"},
    {"alp": 66.0}, {"category": "donante"}, {"crea": True},   # `alp` y otras variables que el modelo no usa se rechazan
])
def test_entradas_invalidas_dan_422(cliente, assert_error, cambio):
    assert_error(cliente.post(URL, json={**ENTRADA, **cambio}), 422, "VALIDACION")


def test_faltan_campos_da_422_y_todos_son_obligatorios(cliente, assert_error):
    assert_error(cliente.post(URL, json={k: v for k, v in ENTRADA.items() if k != "ggt"}), 422, "VALIDACION")
    esquema = Entrada.model_json_schema()
    assert set(esquema["required"]) == set(ENTRADA) and esquema.get("additionalProperties") is False


def test_los_metadatos_del_modelo_registrado():
    info = REGISTRO["hepatitis"].info
    assert info["tipo"] == "clasificacion" and info["unidad"] is None and info["slug"] == "hepatitis"


# --- Modelo entrenado y API ---

@entrenado
def test_el_modelo_guardado_reproduce_las_metricas_publicadas():
    _, test, _, y_test = dividir(cargar_datos())
    pipeline = joblib.load(CARPETA / "modelo.joblib")["pipeline"]
    m = leer_metricas(CARPETA)["metricas"]
    assert f1_macro_de(y_test, pipeline.predict(test[VARIABLES])) == pytest.approx(m["f1_macro"], abs=1e-3)
    assert m["f1_macro"] > m["baseline"]["f1_macro"] + 0.25


@entrenado
def test_las_metricas_publicadas_son_coherentes_y_con_intervalos_amplios():
    m = leer_metricas(CARPETA)["metricas"]
    matriz = np.array(m["matriz_confusion"])
    assert m["orden_clases"] == list(ORDEN) and matriz.shape == (4, 4) and matriz.sum() == m["n_prueba"] == 122
    assert matriz.sum(axis=1).tolist() == [m["casos_por_clase_prueba"][c] for c in ORDEN] == [107, 5, 4, 6]
    bajo, alto = m["ic95"]["f1_macro"]
    assert bajo < m["f1_macro"] < alto and alto - bajo > 0.25, "con 4-6 casos por clase el intervalo debe ser muy ancho"
    assert 0 <= m["enfermedad_vs_donante"]["sensibilidad"] <= 1 and m["oof_entrenamiento"]["por_clase"]["donante"]["support"] == 426


@entrenado
def test_el_alp_no_entra_al_modelo_y_su_aporte_a_los_arboles_era_un_artefacto():
    pipeline = cargar_artefacto(CARPETA, "hepatitis")["pipeline"]
    assert list(pipeline.feature_names_in_) == VARIABLES and "alp" not in pipeline.feature_names_in_
    e = leer_metricas(CARPETA)["metricas"]["experimento_alp_random_forest"]
    assert e["random forest con ALP imputada por la mediana"]["cv_f1_macro"] - e["random forest sin ALP (elegido)"]["cv_f1_macro"] > 0.02


@entrenado
def test_predecir(cliente):
    cuerpo = cliente.post(URL, json=ENTRADA).json()
    assert cuerpo["modelo"] == "hepatitis" and cuerpo["prediccion"] in ORDEN
    assert set(cuerpo["probabilidades"]) == set(ORDEN) and sum(cuerpo["probabilidades"].values()) == pytest.approx(1, abs=0.01)
    assert cuerpo["prediccion"] == max(cuerpo["probabilidades"], key=cuerpo["probabilidades"].get)
    assert f"como {cuerpo['prediccion']}" in cuerpo["texto"]
    for c in ORDEN:
        assert f"{c} {round(cuerpo['probabilidades'][c] * 100)} %" in cuerpo["texto"]


@entrenado
def test_el_texto_avisa_que_no_es_un_diagnostico_ni_probabilidades_calibradas(cliente):
    texto = cliente.post(URL, json=ENTRADA).json()["texto"]
    assert "no es un diagnóstico" in texto and "no reemplaza la valoración de un profesional de la salud" in texto
    assert "no son probabilidades calibradas" in texto
    assert "sin signos de enfermedad hepática" in texto, "un perfil de donante lo dice"


@entrenado
def test_un_perfil_tipico_de_cada_extremo_se_clasifica_bien(cliente):
    """Medianas reales de la clase como entrada: el donante típico es 'donante' y el cirrótico típico no lo es."""
    df = cargar_datos()
    for clase, esperado in [("donante", "donante"), ("cirrosis", "cirrosis")]:
        mediana = df[df[OBJETIVO] == clase][NUMERICAS].median().round(1).to_dict()
        fila = {**mediana, "age": int(mediana["age"]), "sex": "m"}
        assert cliente.post(URL, json=fila).json()["prediccion"] == esperado, clase


@entrenado
def test_el_perfil_de_laboratorio_tiene_sentido_clinico():
    """Más AST, GGT y bilirrubina y menos albúmina empujan al modelo lejos de 'donante' (sobre donantes reales)."""
    df = cargar_datos()
    donantes = df[df[OBJETIVO] == "donante"].dropna(subset=VARIABLES).sample(150, random_state=1)[VARIABLES].to_dict("records")
    sano = puntajes_lote(donantes)["donante"].to_numpy()
    danado = puntajes_lote([{**f, "ast": min(f["ast"] * 3, 380), "ggt": min(f["ggt"] * 4, 780), "bil": min(f["bil"] * 4, 290), "alb": f["alb"] - 8} for f in donantes])["donante"].to_numpy()
    assert np.mean(danado < sano) >= 0.9


@entrenado
def test_valores_fuera_de_rango_avisan_que_el_resultado_es_poco_confiable(cliente):
    assert "poco confiable" not in cliente.post(URL, json=ENTRADA).json()["texto"]
    assert "poco confiable" in cliente.post(URL, json={**ENTRADA, "ast": 399.0}).json()["texto"]
    assert "poco confiable" in cliente.post(URL, json={**ENTRADA, "age": 95}).json()["texto"]
    rango = cargar_artefacto(CARPETA, "hepatitis")["rango"]
    train, _, _, _ = dividir(cargar_datos())
    assert set(rango) == set(NUMERICAS)
    for campo, (minimo, maximo) in rango.items():
        assert (minimo, maximo) == (train[campo].min(), train[campo].max()), campo


@pytest.mark.reproduce
@entrenado
def test_reentrenar_reproduce_exactamente_lo_publicado(reentrenado):
    """Cubre todo `entrenar()`: selección, evaluación, bootstrap, predicciones fuera de muestra y experimentos."""
    publicado = leer_metricas(CARPETA)
    artefacto = cargar_artefacto(CARPETA, "hepatitis")
    assert reentrenado["metricas"] == publicado["metricas"]
    assert reentrenado["entrada_ejemplo"] == publicado["entrada_ejemplo"]
    assert reentrenado["rango"] == artefacto["rango"]
    _, test, _, _ = dividir(cargar_datos())
    esperado = artefacto["pipeline"].predict_proba(test[VARIABLES])
    assert reentrenado["pipeline"].predict_proba(test[VARIABLES]) == pytest.approx(esperado, abs=1e-9)


@entrenado
def test_info_incluye_metricas_esquema_y_ejemplo_valido(cliente):
    cuerpo = cliente.get("/api/modelos/hepatitis/info").json()
    assert {"f1_macro", "accuracy", "ic95", "por_clase", "enfermedad_vs_donante", "oof_entrenamiento"} <= set(cuerpo["metricas"])
    assert set(ENTRADA) == set(cuerpo["esquema_entrada"]["properties"]) == set(cuerpo["entrada_ejemplo"])
    assert isinstance(cuerpo["entrada_ejemplo"]["age"], int) and cliente.post(URL, json=cuerpo["entrada_ejemplo"]).status_code == 200


@entrenado
def test_el_artefacto_no_es_excesivo():
    assert (CARPETA / "modelo.joblib").stat().st_size < 5_000_000
