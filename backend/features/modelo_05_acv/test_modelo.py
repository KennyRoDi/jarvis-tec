"""Pruebas del modelo 05 · ACV (specs/api_rest_spec.md §3). Usan el modelo entrenado real."""
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score

from core.entrenamiento import umbral_optimo_f1
from core.modelos import REGISTRO, cargar_artefacto, leer_metricas, predecir_con_pipeline
from features.modelo_05_acv.router import CARPETA, Entrada
from features.modelo_05_acv.train import (CATEGORICAS, NUMERICAS, OBJETIVO, POSITIVO, VARIABLES, candidatos, cargar_datos,
                                          dividir, entrenar, estadisticas_al_umbral, REJILLA_F1)

URL = "/api/modelos/acv/predecir"
ENTRADA = {"age": 67.0, "hypertension": "Yes", "heart_disease": "No", "avg_glucose_level": 105.0, "bmi": 28.5}
JOVEN = {"age": 25.0, "hypertension": "No", "heart_disease": "No", "avg_glucose_level": 85.0, "bmi": 22.0}
entrenado = pytest.mark.skipif(not REGISTRO["acv"].entrenado, reason="ejecutar features.modelo_05_acv.train")


@pytest.fixture(scope="module")
def reentrenado():
    """Todo el entrenamiento (selección, umbrales, evaluación, bootstrap, experimento) en memoria, sin escribir nada."""
    return entrenar(cargar_datos(), figuras=None, imprimir=False)


def p_acv(fila: dict) -> float:
    return predecir_con_pipeline(CARPETA, "acv", fila)[1]["Yes"]


def p_lote(filas: list[dict]) -> np.ndarray:
    """Probabilidad de ACV de muchas filas a la vez (una sola llamada al pipeline)."""
    pipeline = cargar_artefacto(CARPETA, "acv")["pipeline"]
    return pipeline.predict_proba(pd.DataFrame(filas))[:, list(pipeline.classes_).index(POSITIVO)]


def filas_reales(n=600, semilla=0):
    return cargar_datos().dropna(subset=["bmi"]).sample(n, random_state=semilla)[VARIABLES].to_dict("records")


# --- Preparación de datos (no requiere el modelo entrenado) ---

def test_los_datos_cargados():
    df = cargar_datos()
    assert len(df) == 5110 and df[OBJETIVO].value_counts().to_dict() == {"No": 4861, "Yes": 249}
    assert df.bmi.isna().sum() == 201 and set(df.hypertension) == {"Yes", "No"}


def test_las_variables_son_las_5_clinicas_sin_datos_sensibles_ni_proxies():
    assert VARIABLES == ["age", "avg_glucose_level", "bmi", "hypertension", "heart_disease"]
    for excluida in ("gender", "ever_married", "work_type", "residence_type", "smoking_status", "bmi_faltante", OBJETIVO, "id"):
        assert excluida not in VARIABLES


def test_la_particion_es_estratificada_y_conserva_los_pocos_positivos():
    df = cargar_datos()
    train, test, y_train, y_test = dividir(df)
    assert (len(train), len(test)) == (4088, 1022) and not set(train.index) & set(test.index)
    assert ((y_train == POSITIVO).sum(), (y_test == POSITIVO).sum()) == (199, 50)


def test_los_candidatos_imputan_con_la_mediana_y_no_ponderan_las_clases():
    for nombre, pipeline in candidatos().items():
        transformadores = {n: (t, c) for n, t, c in pipeline.named_steps["pre"].transformers}
        assert transformadores["num"][0].named_steps["imputar"].strategy == "median", nombre
        assert transformadores["num"][1] == NUMERICAS and transformadores["cat"][1] == CATEGORICAS, nombre
        assert getattr(pipeline.named_steps["modelo"], "class_weight", None) is None, nombre
    assert isinstance(candidatos()["tasa base (línea base)"].named_steps["modelo"], DummyClassifier)


def test_estadisticas_al_umbral_incluye_el_borde_y_orienta_la_matriz():
    y = ["Yes", "Yes", "Yes", "No", "No", "No", "No"]
    p = [0.35, 0.34, 0.90, 0.35, 0.10, 0.20, 0.05]
    e = estadisticas_al_umbral(y, p, 0.35)
    assert e["matriz_confusion"] == [[3, 1], [1, 2]]
    assert (e["recall_yes"], e["precision_yes"], e["accuracy"]) == (0.6667, 0.6667, 0.7143)


def test_la_rejilla_del_umbral_f1_llega_por_debajo_de_0_05_porque_la_clase_es_rara():
    """Si los positivos tienen probabilidades muy bajas pero separables, el umbral perfecto es 0.01: la rejilla por
    defecto (desde 0.05) no lo alcanzaría y dejaría pasar F1 = 0."""
    y = np.array([1] * 50 + [0] * 950)
    p = np.concatenate([np.full(50, 0.03), np.full(950, 0.005)])
    assert umbral_optimo_f1(y, p, rejilla=REJILLA_F1) == 0.01
    assert umbral_optimo_f1(y, p) == 0.05, "la rejilla por defecto queda corta (por eso este modelo usa la suya)"


# --- Validación de la entrada (no requiere el modelo entrenado) ---

def test_la_entrada_acepta_a_todos_los_pacientes_reales_con_bmi():
    for fila in filas_reales(n=4000):
        Entrada(**fila)


def test_los_limites_de_entrada_son_razonables_frente_a_los_datos():
    df = cargar_datos()
    propiedades = Entrada.model_json_schema()["properties"]
    for campo in NUMERICAS:
        minimo, maximo = df[campo].min(), df[campo].max()
        assert propiedades[campo]["minimum"] <= minimo and (propiedades[campo]["minimum"] == 0 or propiedades[campo]["minimum"] >= 0.4 * minimo), campo
        assert maximo <= propiedades[campo]["maximum"] <= 1.8 * maximo, campo


@pytest.mark.parametrize("cambio", [
    {"age": -1}, {"age": 150}, {"bmi": 5}, {"avg_glucose_level": 1000}, {"hypertension": "Quizá"}, {"age": "67"}, {"age": True},
    {"hypertension": 1}, {"gender": "Female"}, {"smoking_status": "never smoked"},   # variables que el modelo no usa
])
def test_entradas_invalidas_dan_422(cliente, assert_error, cambio):
    assert_error(cliente.post(URL, json={**ENTRADA, **cambio}), 422, "VALIDACION")


def test_faltan_campos_da_422_y_todos_son_obligatorios(cliente, assert_error):
    assert_error(cliente.post(URL, json={k: v for k, v in ENTRADA.items() if k != "bmi"}), 422, "VALIDACION")
    esquema = Entrada.model_json_schema()
    assert set(esquema["required"]) == set(ENTRADA) and esquema.get("additionalProperties") is False


def test_los_metadatos_del_modelo_registrado():
    info = REGISTRO["acv"].info
    assert info["tipo"] == "clasificacion" and info["unidad"] is None and info["slug"] == "acv"


# --- Modelo entrenado y API ---

@entrenado
def test_el_modelo_guardado_reproduce_las_metricas_publicadas():
    _, test, _, y_test = dividir(cargar_datos())
    pipeline = joblib.load(CARPETA / "modelo.joblib")["pipeline"]
    auc = roc_auc_score(y_test == POSITIVO, pipeline.predict_proba(test[VARIABLES])[:, list(pipeline.classes_).index(POSITIVO)])
    m = leer_metricas(CARPETA)["metricas"]
    assert auc == pytest.approx(m["roc_auc"], abs=1e-3) and auc > m["baseline"]["roc_auc"] + 0.3


@entrenado
def test_las_probabilidades_estan_calibradas():
    m = leer_metricas(CARPETA)["metricas"]
    assert m["brier"] < m["brier_tasa_base"]
    for tramo in m["calibracion"]:
        assert abs(tramo["prob_media"] - tramo["tasa_real"]) < 0.04, tramo


@entrenado
def test_los_umbrales_y_el_tamizaje():
    """El umbral de sensibilidad detecta más casos que el de F1; con 0.5 el modelo no detectaría a nadie."""
    m = leer_metricas(CARPETA)["metricas"]
    assert 0 < m["umbral_sensibilidad"] < m["umbral"] < 0.5
    assert m["al_umbral_de_sensibilidad"]["recall_yes"] >= m["recall_objetivo"] > m["recall_yes"] > m["en_umbral_0_5"]["recall_yes"] == 0
    assert m["en_umbral_0_5"]["accuracy"] == pytest.approx(m["baseline"]["accuracy"], abs=0.001), "con 0.5 solo se predice 'No'"
    assert np.array(m["matriz_confusion"]).sum() == m["n_prueba"] == 1022


@entrenado
def test_los_intervalos_de_confianza_contienen_las_estimaciones():
    m = leer_metricas(CARPETA)["metricas"]
    for clave, valor in [("roc_auc", m["roc_auc"]), ("pr_auc", m["pr_auc"]), ("recall_yes", m["recall_yes"])]:
        bajo, alto = m["ic95"][clave]
        assert bajo < valor < alto, clave
    assert m["ic95"]["roc_auc"][1] - m["ic95"]["roc_auc"][0] > 0.05, "con ~50 positivos el intervalo debe ser ancho"


@entrenado
def test_predecir(cliente):
    cuerpo = cliente.post(URL, json=ENTRADA).json()
    assert cuerpo["modelo"] == "acv" and cuerpo["prediccion"] in ("Yes", "No")
    assert set(cuerpo["probabilidades"]) == {"Yes", "No"} and sum(cuerpo["probabilidades"].values()) == pytest.approx(1, abs=0.01)
    assert f"{round(cuerpo['probabilidades']['Yes'] * 100, 1)} por ciento" in cuerpo["texto"]
    assert "no reemplaza la valoración de un profesional de la salud" in cuerpo["texto"]


@entrenado
def test_los_tres_niveles_de_riesgo_cambian_exactamente_en_los_umbrales_guardados(cliente):
    """Se prueban las filas reales más cercanas a cada borde: bajo | umbral de sensibilidad | moderado | umbral F1 | alto."""
    a = cargar_artefacto(CARPETA, "acv")
    filas = filas_reales(4000)
    p = p_lote(filas)
    s, u = a["umbral_sensibilidad"], a["umbral"]
    borde = {  # fila real con la probabilidad más cercana al borde, por cada lado
        ("bajo", "No"): int(np.argmax(np.where(p < s, p, -1))),
        ("moderado", "No"): int(np.argmin(np.where(p >= s, p, 9))),
        ("moderado ", "No"): int(np.argmax(np.where(p < u, p, -1))),
        ("alto", "Yes"): int(np.argmin(np.where(p >= u, p, 9))),
    }
    for (nivel, clase), indice in borde.items():
        cuerpo = cliente.post(URL, json=filas[indice]).json()
        assert cuerpo["prediccion"] == clase and f"riesgo {nivel.strip()}" in cuerpo["texto"], (nivel, p[indice])
        assert sum(f"riesgo {n}" in cuerpo["texto"] for n in ("bajo", "moderado", "alto")) == 1, "un único nivel por texto"
    assert p[borde[("bajo", "No")]] < s <= p[borde[("moderado", "No")]] < p[borde[("moderado ", "No")]] < u <= p[borde[("alto", "Yes")]]


@entrenado
def test_el_perfil_de_riesgo_tiene_sentido():
    """La edad domina (mediana de 71 años con ACV contra 43): más edad, hipertensión y enfermedad cardíaca elevan el riesgo."""
    assert p_acv(ENTRADA) > 5 * p_acv(JOVEN)
    filas = filas_reales(300, 1)
    mejora_edad = np.mean(p_lote([{**f, "age": 75.0} for f in filas]) > p_lote([{**f, "age": 35.0} for f in filas]))
    mejora_hta = np.mean(p_lote([{**f, "hypertension": "Yes"} for f in filas]) >= p_lote([{**f, "hypertension": "No"} for f in filas]))
    assert mejora_edad >= 0.98 and mejora_hta >= 0.9


@entrenado
def test_medidas_fuera_de_rango_avisan_que_el_resultado_es_poco_confiable(cliente):
    assert "poco confiable" not in cliente.post(URL, json=ENTRADA).json()["texto"]
    assert "poco confiable" in cliente.post(URL, json={**ENTRADA, "age": 110.0}).json()["texto"]
    assert "poco confiable" in cliente.post(URL, json={**ENTRADA, "avg_glucose_level": 395.0}).json()["texto"]
    rango = cargar_artefacto(CARPETA, "acv")["rango"]
    # El rango de entrenamiento del bmi (10.3-97.6) cubre los límites de Entrada (10-100) dentro del margen del 5 %:
    # ningún bmi válido dispara el aviso (el aviso sí funciona para edad y glucosa, arriba).
    margen = (rango["bmi"][1] - rango["bmi"][0]) * 0.05
    assert rango["bmi"][0] - margen <= 10 and 100 <= rango["bmi"][1] + margen
    train, _, _, _ = dividir(cargar_datos())
    assert set(rango) == set(NUMERICAS)
    for campo, (minimo, maximo) in rango.items():
        assert (minimo, maximo) == (train[campo].min(), train[campo].max()), campo


@pytest.mark.reproduce
@entrenado
def test_reentrenar_reproduce_exactamente_lo_publicado(reentrenado):
    """Cubre todo `entrenar()`: selección, umbrales, evaluación, bootstrap y experimento de variables."""
    publicado = leer_metricas(CARPETA)
    artefacto = cargar_artefacto(CARPETA, "acv")
    assert reentrenado["metricas"] == publicado["metricas"]
    assert reentrenado["entrada_ejemplo"] == publicado["entrada_ejemplo"]
    assert reentrenado["umbral"] == artefacto["umbral"] == publicado["metricas"]["umbral"]
    assert reentrenado["umbral_sensibilidad"] == artefacto["umbral_sensibilidad"] == publicado["metricas"]["umbral_sensibilidad"]
    assert reentrenado["rango"] == artefacto["rango"]
    _, test, _, _ = dividir(cargar_datos())
    assert reentrenado["pipeline"].predict_proba(test[VARIABLES]) == pytest.approx(artefacto["pipeline"].predict_proba(test[VARIABLES]), abs=1e-9)


@entrenado
def test_info_incluye_metricas_esquema_y_ejemplo_valido(cliente):
    cuerpo = cliente.get("/api/modelos/acv/info").json()
    assert {"roc_auc", "pr_auc", "brier", "umbral", "umbral_sensibilidad", "ic95", "recall_yes"} <= set(cuerpo["metricas"])
    assert set(ENTRADA) == set(cuerpo["esquema_entrada"]["properties"]) == set(cuerpo["entrada_ejemplo"])
    assert cliente.post(URL, json=cuerpo["entrada_ejemplo"]).status_code == 200


@entrenado
def test_el_artefacto_no_es_excesivo():
    assert (CARPETA / "modelo.joblib").stat().st_size < 5_000_000
