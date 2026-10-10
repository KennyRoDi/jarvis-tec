"""Pruebas del modelo 03 · vino (specs/api_rest_spec.md §3). Usan el modelo entrenado real."""
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from core.modelos import REGISTRO, leer_metricas, predecir_con_pipeline
from features.modelo_03_vino.router import CARPETA, ORDEN, Entrada
from features.modelo_03_vino.train import NUMERICAS, OBJETIVO, VARIABLES, agrupar_calidad, candidatos, cargar_datos, dividir

URL = "/api/modelos/vino/predecir"
ENTRADA = {
    "tipo": "white", "fixed_acidity": 7.0, "volatile_acidity": 0.3, "citric_acid": 0.32, "residual_sugar": 2.0,
    "chlorides": 0.045, "free_sulfur_dioxide": 30.0, "total_sulfur_dioxide": 115.0, "density": 0.994,
    "ph": 3.2, "sulphates": 0.5, "alcohol": 10.5,
}
ESQUINA = {**ENTRADA, "fixed_acidity": 3, "ph": 4.2, "sulphates": 2.1, "alcohol": 15.5}  # en el borde de lo permitido
entrenado = pytest.mark.skipif(not REGISTRO["vino"].entrenado, reason="ejecutar features.modelo_03_vino.train")


def probabilidades(fila: dict) -> dict:
    return predecir_con_pipeline(CARPETA, "vino", fila)[1]


# --- Preparación de datos (no requiere el modelo entrenado) ---

@pytest.mark.parametrize("puntaje, clase", [(3, "baja"), (5, "baja"), (6, "media"), (7, "alta"), (9, "alta")])
def test_agrupacion_de_la_calidad(puntaje, clase):
    assert agrupar_calidad(pd.Series([puntaje])).iloc[0] == clase


def test_no_quedan_duplicados_antes_de_dividir():
    """Con duplicados, filas idénticas caen en entrenamiento y prueba y la exactitud sube sin generalizar."""
    df = cargar_datos()
    assert not df.duplicated().any() and len(df) == 5329


def test_las_variables_no_incluyen_el_objetivo_ni_la_puntuacion():
    assert "quality" not in VARIABLES and OBJETIVO not in VARIABLES and len(VARIABLES) == 12


def test_la_particion_es_estratificada_y_no_comparte_filas():
    df = cargar_datos()
    X_train, X_test, y_train, y_test = dividir(df)
    assert (len(X_train), len(X_test)) == (4263, 1066) and not set(X_train.index) & set(X_test.index)
    for clase in ORDEN:
        assert (y_train == clase).mean() == pytest.approx((y_test == clase).mean(), abs=0.005)
        assert (y_test == clase).mean() == pytest.approx((df[OBJETIVO] == clase).mean(), abs=0.005)


def test_todos_los_candidatos_imputan_con_la_mediana_dentro_del_pipeline():
    """La imputación vive en el Pipeline (se ajusta solo con los datos de cada pliegue) y usa la mediana."""
    for nombre, pipeline in candidatos().items():
        transformadores = {n: t for n, t, _ in pipeline.named_steps["pre"].transformers}
        assert transformadores["num"].named_steps["imputar"].strategy == "median", nombre


# --- Modelo entrenado ---

@entrenado
def test_el_modelo_guardado_reproduce_las_metricas_publicadas():
    """El .joblib y el metricas.json deben corresponder al mismo entrenamiento (y superar claramente a la línea base)."""
    _, X_test, _, y_test = dividir(cargar_datos())
    pipeline = joblib.load(CARPETA / "modelo.joblib")["pipeline"]
    f1 = f1_score(y_test, pipeline.predict(X_test), average="macro")
    m = leer_metricas(CARPETA)["metricas"]
    assert f1 == pytest.approx(m["f1_macro"], abs=1e-3)
    assert f1 > m["baseline"]["f1_macro"] + 0.3


@entrenado
def test_la_matriz_de_confusion_esta_en_el_orden_baja_media_alta():
    """Las sumas por fila deben coincidir con las muestras reales de cada clase, en ese orden."""
    _, _, _, y_test = dividir(cargar_datos())
    metricas = leer_metricas(CARPETA)["metricas"]
    assert metricas["orden_clases"] == list(ORDEN)
    assert np.array(metricas["matriz_confusion"]).sum(axis=1).tolist() == [int((y_test == c).sum()) for c in ORDEN]


@entrenado
def test_el_imputador_se_ajusto_solo_con_el_entrenamiento():
    """Las medianas del imputador deben ser las de X_train (no las de todo el dataset: habría fuga)."""
    X_train, _, _, _ = dividir(cargar_datos())
    imputador = joblib.load(CARPETA / "modelo.joblib")["pipeline"].named_steps["pre"].named_transformers_["num"].named_steps["imputar"]
    assert imputador.strategy == "median"
    assert imputador.statistics_ == pytest.approx(X_train[NUMERICAS].median().to_numpy())


@entrenado
def test_predecir(cliente):
    cuerpo = cliente.post(URL, json=ENTRADA).json()
    assert cuerpo["modelo"] == "vino" and cuerpo["prediccion"] in ORDEN
    assert set(cuerpo["probabilidades"]) == set(ORDEN)
    assert sum(cuerpo["probabilidades"].values()) == pytest.approx(1, abs=0.01)
    assert cuerpo["prediccion"] == max(cuerpo["probabilidades"], key=cuerpo["probabilidades"].get)
    assert f"calidad {cuerpo['prediccion']}" in cuerpo["texto"] and "vino blanco" in cuerpo["texto"]


@entrenado
def test_el_texto_distingue_tinto_de_blanco(cliente):
    assert "vino tinto" in cliente.post(URL, json={**ENTRADA, "tipo": "red"}).json()["texto"]


@entrenado
def test_aviso_de_clasificacion_poco_concluyente(cliente):
    """Aparece cuando ninguna clase supera el 50 % y no aparece cuando el modelo está seguro."""
    filas = cargar_datos().dropna(subset=NUMERICAS).sample(300, random_state=0)[VARIABLES].to_dict("records")
    con_dudas = next(f for f in filas if max(probabilidades(f).values()) < 0.45)
    seguro = next(f for f in filas if max(probabilidades(f).values()) > 0.75)
    assert "poco concluyente" in cliente.post(URL, json=con_dudas).json()["texto"]
    assert "poco concluyente" not in cliente.post(URL, json=seguro).json()["texto"]


@entrenado
def test_medidas_en_la_esquina_avisan_que_el_resultado_es_poco_confiable(cliente):
    assert "poco confiable" in cliente.post(URL, json=ESQUINA).json()["texto"]
    assert "poco confiable" not in cliente.post(URL, json=ENTRADA).json()["texto"]


@entrenado
def test_mas_alcohol_aumenta_la_probabilidad_de_calidad_alta():
    """Coherencia con la exploración: el alcohol es la variable más asociada a la calidad (r = 0.47)."""
    filas = cargar_datos().sample(40, random_state=0)[VARIABLES].to_dict("records")
    mejora = np.mean([probabilidades({**f, "alcohol": min(f["alcohol"] + 2, 15)})["alta"]
                      > probabilidades({**f, "alcohol": max(f["alcohol"] - 2, 8)})["alta"] for f in filas])
    assert mejora >= 0.9


# --- Validación de la entrada (no requiere el modelo entrenado) ---

def test_la_entrada_acepta_todos_los_vinos_reales_del_dataset():
    """Los límites no deben rechazar vinos reales (las filas con nulos quedan fuera: la API exige los 12 datos)."""
    filas = cargar_datos(quitar_duplicados=False).dropna(subset=NUMERICAS)[VARIABLES].to_dict("records")
    assert len(filas) > 6400
    for fila in filas:
        Entrada(**fila)


def test_los_limites_de_entrada_son_razonables_frente_a_los_datos():
    """Ni tan estrechos que rechacen vinos reales ni tan amplios que dejen pasar datos absurdos."""
    datos = cargar_datos(quitar_duplicados=False)
    propiedades = Entrada.model_json_schema()["properties"]
    for campo in NUMERICAS:
        minimo, maximo = datos[campo].min(), datos[campo].max()
        p = propiedades[campo]
        assert 0.4 * minimo <= p["minimum"] <= minimo, campo
        assert maximo <= p["maximum"] <= 1.6 * maximo, campo


@pytest.mark.parametrize("cambio", [{"tipo": "rosado"}, {"alcohol": 40}, {"ph": 14}, {"density": -1}, {"alcohol": "mucho"},
                                    {"free_sulfur_dioxide": 200, "total_sulfur_dioxide": 100}])
def test_entradas_invalidas_dan_422(cliente, assert_error, cambio):
    assert_error(cliente.post(URL, json={**ENTRADA, **cambio}), 422, "VALIDACION")


def test_faltan_campos_da_422(cliente, assert_error):
    assert_error(cliente.post(URL, json={k: v for k, v in ENTRADA.items() if k != "alcohol"}), 422, "VALIDACION")


@entrenado
def test_info_incluye_metricas_matriz_y_esquema(cliente):
    cuerpo = cliente.get("/api/modelos/vino/info").json()
    assert {"accuracy", "f1_macro", "precision_macro", "recall_macro"} <= set(cuerpo["metricas"])
    assert np.array(cuerpo["metricas"]["matriz_confusion"]).shape == (3, 3)
    assert set(ENTRADA) == set(cuerpo["esquema_entrada"]["properties"])
    assert cuerpo["esquema_entrada"]["properties"]["tipo"]["enum"] == ["red", "white"]
    assert set(ENTRADA) == set(cuerpo["entrada_ejemplo"])


@entrenado
def test_el_artefacto_no_es_excesivo():
    assert (CARPETA / "modelo.joblib").stat().st_size < 5_000_000
