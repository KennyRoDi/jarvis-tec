"""Pruebas del modelo 04 · churn (specs/api_rest_spec.md §3). Usan el modelo entrenado real."""
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score

from core.modelos import REGISTRO, cargar_artefacto, leer_metricas, predecir_con_pipeline
from features.modelo_04_churn.router import CARPETA, Entrada
from features.modelo_04_churn.train import (CATEGORICAS, NUMERICAS, OBJETIVO, POSITIVO, VARIABLES, candidatos, cargar_datos,
                                            dividir)

URL = "/api/modelos/churn/predecir"
ENTRADA = {
    "tenure": 12, "monthly_charges": 70.0, "contract": "Month-to-month", "internet_service": "Fiber optic",
    "payment_method": "Electronic check", "paperless_billing": "Yes", "tech_support": "No", "online_security": "No",
    "senior_citizen": "No",
}
SEGURO = {**ENTRADA, "tenure": 60, "contract": "Two year", "internet_service": "DSL", "payment_method": "Credit card (automatic)",
          "paperless_billing": "No", "tech_support": "Yes", "online_security": "Yes", "monthly_charges": 55.0}
entrenado = pytest.mark.skipif(not REGISTRO["churn"].entrenado, reason="ejecutar features.modelo_04_churn.train")


def p_abandona(fila: dict) -> float:
    return predecir_con_pipeline(CARPETA, "churn", fila)[1]["Yes"]


# --- Preparación de datos (no requiere el modelo entrenado) ---

def test_los_datos_cargados():
    df = cargar_datos()
    assert len(df) == 7043 and not df.isna().any().any()
    assert df.total_charges.eq(0).sum() == 11, "los 11 vacíos originales corresponden a clientes con 0 meses"
    assert set(df.senior_citizen) == {"Yes", "No"} and set(df[OBJETIVO]) == {"Yes", "No"}


def test_las_variables_no_incluyen_el_objetivo_ni_el_total_redundante():
    assert OBJETIVO not in VARIABLES and "total_charges" not in VARIABLES and len(VARIABLES) == 9
    assert set(VARIABLES) == set(NUMERICAS + CATEGORICAS)


def test_la_particion_es_estratificada_por_abandono():
    df = cargar_datos()
    train, test, y_train, y_test = dividir(df)
    assert (len(train), len(test)) == (5634, 1409) and not set(train.index) & set(test.index)
    base = (df[OBJETIVO] == POSITIVO).mean()
    assert (y_train == POSITIVO).mean() == pytest.approx(base, abs=0.002)
    assert (y_test == POSITIVO).mean() == pytest.approx(base, abs=0.002)


def test_los_candidatos_tratan_cada_variable_como_corresponde():
    """Numéricas escaladas y categóricas codificadas, sin ponderar clases (las probabilidades deben estar calibradas)."""
    for nombre, pipeline in candidatos().items():
        columnas = {n: c for n, _, c in pipeline.named_steps["pre"].transformers}
        assert columnas["num"] == NUMERICAS and columnas["cat"] == CATEGORICAS, nombre
        assert getattr(pipeline.named_steps["modelo"], "class_weight", None) is None, nombre
    assert isinstance(candidatos()["tasa base (línea base)"].named_steps["modelo"], DummyClassifier)


# --- Validación de la entrada (no requiere el modelo entrenado) ---

def test_la_entrada_acepta_a_todos_los_clientes_reales():
    """Los límites y la coherencia internet/servicios no deben rechazar clientes reales."""
    filas = cargar_datos()[VARIABLES].to_dict("records")
    for fila in filas:
        Entrada(**fila)


def test_los_limites_de_entrada_son_razonables_frente_a_los_datos():
    df = cargar_datos()
    propiedades = Entrada.model_json_schema()["properties"]
    for campo in NUMERICAS:
        minimo, maximo = df[campo].min(), df[campo].max()
        assert 0.4 * minimo <= propiedades[campo]["minimum"] <= minimo, campo
        assert maximo <= propiedades[campo]["maximum"] <= 1.8 * maximo, campo


@pytest.mark.parametrize("cambio", [
    {"contract": "Weekly"}, {"tenure": -1}, {"monthly_charges": 1000}, {"tenure": "mucho"}, {"senior_citizen": "Quizá"},
    {"internet_service": "No"},                                              # sin internet pero con soporte y seguridad
    {"internet_service": "No", "tech_support": "No internet service"},       # online_security sigue incoherente
    {"tech_support": "No internet service"},                                 # con internet no puede ser "No internet service"
])
def test_entradas_invalidas_dan_422(cliente, assert_error, cambio):
    assert_error(cliente.post(URL, json={**ENTRADA, **cambio}), 422, "VALIDACION")


def test_cliente_sin_internet_es_una_entrada_valida():
    sin = {**ENTRADA, "internet_service": "No", "tech_support": "No internet service", "online_security": "No internet service"}
    assert Entrada(**sin).internet_service == "No"


def test_faltan_campos_da_422(cliente, assert_error):
    assert_error(cliente.post(URL, json={k: v for k, v in ENTRADA.items() if k != "contract"}), 422, "VALIDACION")


# --- Modelo entrenado y API ---

@entrenado
def test_el_modelo_guardado_reproduce_las_metricas_publicadas():
    train, test, _, y_test = dividir(cargar_datos())
    pipeline = joblib.load(CARPETA / "modelo.joblib")["pipeline"]
    auc = roc_auc_score(y_test == POSITIVO, pipeline.predict_proba(test[VARIABLES])[:, list(pipeline.classes_).index(POSITIVO)])
    m = leer_metricas(CARPETA)["metricas"]
    assert auc == pytest.approx(m["roc_auc"], abs=1e-3)
    assert auc > m["baseline"]["roc_auc"] + 0.3


@entrenado
def test_las_probabilidades_estan_calibradas():
    """Sin pesos de clase la probabilidad mostrada debe parecerse a la tasa real (Brier mejor que predecir la tasa base)."""
    m = leer_metricas(CARPETA)["metricas"]
    assert m["brier"] < 0.19
    for tramo in m["calibracion"]:
        assert abs(tramo["prob_media"] - tramo["tasa_real"]) < 0.08, tramo


@entrenado
def test_el_umbral_elegido_mejora_el_recall_de_la_clase_que_se_va():
    m = leer_metricas(CARPETA)["metricas"]
    assert 0.15 < m["umbral"] < 0.5 and m["recall_yes"] > m["en_umbral_0_5"]["recall_yes"]
    assert np.array(m["matriz_confusion"]).sum() == m["n_prueba"] == 1409


@entrenado
def test_predecir(cliente):
    cuerpo = cliente.post(URL, json=ENTRADA).json()
    assert cuerpo["modelo"] == "churn" and cuerpo["prediccion"] in ("Yes", "No")
    assert set(cuerpo["probabilidades"]) == {"Yes", "No"}
    assert sum(cuerpo["probabilidades"].values()) == pytest.approx(1, abs=0.01)
    assert f"{round(cuerpo['probabilidades']['Yes'] * 100)} por ciento" in cuerpo["texto"]


@entrenado
def test_la_clase_depende_del_umbral_guardado_y_no_del_050(cliente):
    """Un cliente con probabilidad entre el umbral y 0.5 debe marcarse como riesgo alto."""
    umbral = cargar_artefacto(CARPETA, "churn")["umbral"]
    filas = cargar_datos().sample(400, random_state=0)[VARIABLES].to_dict("records")
    intermedio = next(f for f in filas if umbral + 0.01 <= p_abandona(f) <= 0.49)
    cuerpo = cliente.post(URL, json=intermedio).json()
    assert cuerpo["prediccion"] == "Yes" and "riesgo alto" in cuerpo["texto"]
    assert cliente.post(URL, json=SEGURO).json()["prediccion"] == "No"
    assert "riesgo bajo" in cliente.post(URL, json=SEGURO).json()["texto"]


@entrenado
def test_el_perfil_de_riesgo_tiene_sentido():
    """Coherencia con la exploración: contrato mes a mes, poca antigüedad y cheque electrónico elevan el riesgo."""
    assert p_abandona(ENTRADA) > p_abandona(SEGURO) + 0.3
    filas = cargar_datos().sample(100, random_state=1)[VARIABLES].to_dict("records")
    mejora = np.mean([p_abandona({**f, "contract": "Month-to-month"}) > p_abandona({**f, "contract": "Two year"}) for f in filas])
    assert mejora >= 0.95


@entrenado
def test_antiguedad_fuera_de_rango_avisa_que_el_resultado_es_poco_confiable(cliente):
    assert "poco confiable" in cliente.post(URL, json={**ENTRADA, "tenure": 110}).json()["texto"]
    assert "poco confiable" not in cliente.post(URL, json=ENTRADA).json()["texto"]


@entrenado
def test_info_incluye_metricas_esquema_y_ejemplo_valido(cliente):
    cuerpo = cliente.get("/api/modelos/churn/info").json()
    assert {"roc_auc", "pr_auc", "brier", "umbral", "recall_yes", "precision_yes", "f1_yes"} <= set(cuerpo["metricas"])
    assert cuerpo["metricas"]["orden_clases"] == ["No", "Yes"]
    assert set(ENTRADA) == set(cuerpo["esquema_entrada"]["properties"]) == set(cuerpo["entrada_ejemplo"])
    assert cliente.post(URL, json=cuerpo["entrada_ejemplo"]).status_code == 200


@entrenado
def test_el_artefacto_no_es_excesivo():
    assert (CARPETA / "modelo.joblib").stat().st_size < 5_000_000
