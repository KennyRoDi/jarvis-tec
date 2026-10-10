"""Pruebas del modelo 03 · vino (specs/api_rest_spec.md §3). Usan el modelo entrenado real."""
import shutil

import joblib
import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.svm import SVC

from core.modelos import REGISTRO, cargar_artefacto, leer_metricas, predecir_con_pipeline
from features.modelo_03_vino import train
from features.modelo_03_vino.router import CARPETA, ORDEN, UMBRAL_POCO_CONCLUYENTE, Entrada
from features.modelo_03_vino.train import (NUMERICAS, OBJETIVO, SEMILLA, VARIABLES, agrupar_calidad, candidatos, cargar_datos, dividir, elegibles, entender,
                                           entrenar, explorar, sin_duplicados)

URL = "/api/modelos/vino/predecir"
ENTRADA = {
    "tipo": "white", "fixed_acidity": 7.0, "volatile_acidity": 0.3, "citric_acid": 0.32, "residual_sugar": 2.0,
    "chlorides": 0.045, "free_sulfur_dioxide": 30.0, "total_sulfur_dioxide": 115.0, "density": 0.994,
    "ph": 3.2, "sulphates": 0.5, "alcohol": 10.5,
}
ESQUINA = {**ENTRADA, "fixed_acidity": 3, "ph": 4.2, "sulphates": 2.1, "alcohol": 15.5}  # en el borde de lo permitido
entrenado = pytest.mark.skipif(not REGISTRO["vino"].entrenado, reason="ejecutar features.modelo_03_vino.train")


@pytest.fixture(scope="module")
def reentrenado():
    """Todo el entrenamiento (selección, evaluación, experimento de duplicados) en memoria, sin escribir nada."""
    return entrenar(cargar_datos(quitar_duplicados=False), figuras=None, imprimir=False)


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


# --- Datos y candidatos (no requieren el modelo entrenado) ---

def test_los_datos_crudos_tienen_duplicados_y_nulos_documentados():
    crudo = cargar_datos(quitar_duplicados=False)
    assert len(crudo) == 6497 and crudo.tipo.value_counts().to_dict() == {"white": 4898, "red": 1599}
    assert int(crudo.duplicated().sum()) == 1168 and len(sin_duplicados(crudo)) == 5329
    limpio = cargar_datos()
    assert limpio[VARIABLES].isna().sum()[lambda s: s > 0].to_dict() == {"fixed_acidity": 10, "volatile_acidity": 8, "citric_acid": 3,
                                                                          "residual_sugar": 2, "chlorides": 2, "ph": 9, "sulphates": 4}
    assert int(limpio[VARIABLES].isna().any(axis=1).sum()) == 34
    assert limpio[OBJETIVO].value_counts().to_dict() == {"media": 2327, "baja": 1991, "alta": 1011}


def test_los_nombres_de_las_columnas_estan_normalizados():
    assert VARIABLES == NUMERICAS + ["tipo"] and len(NUMERICAS) == 11
    assert {"fixed_acidity", "free_sulfur_dioxide", "total_sulfur_dioxide", "ph"} <= set(NUMERICAS) and "quality" not in cargar_datos()[VARIABLES]


def test_los_candidatos_son_una_linea_base_y_cuatro_modelos():
    c = candidatos()
    assert list(c) == ["mayoritaria (línea base)", "regresión logística", "svm rbf", "random forest", "gradient boosting"]
    modelos = {n: p.named_steps["modelo"] for n, p in c.items()}
    assert isinstance(modelos["mayoritaria (línea base)"], DummyClassifier) and isinstance(modelos["regresión logística"], LogisticRegression)
    assert isinstance(modelos["svm rbf"], SVC) and isinstance(modelos["random forest"], RandomForestClassifier)
    assert isinstance(modelos["gradient boosting"], HistGradientBoostingClassifier) and modelos["gradient boosting"].early_stopping is False
    bosque = modelos["random forest"]
    assert (bosque.n_estimators, bosque.min_samples_leaf, bosque.max_depth, bosque.class_weight) == (100, 5, 16, "balanced_subsample")
    assert modelos["svm rbf"].C == 3 and modelos["svm rbf"].class_weight == "balanced"


def test_la_svm_no_estima_probabilidades_y_por_eso_no_puede_ser_el_modelo_elegido():
    """`probability=True` está deprecado en scikit-learn 1.9: sin él la SVM no ofrece `predict_proba`, que la API necesita."""
    c = candidatos()
    assert not hasattr(c["svm rbf"], "predict_proba") and c["svm rbf"].named_steps["modelo"].probability in (False, "deprecated")
    for nombre in ("regresión logística", "random forest", "gradient boosting"):
        assert hasattr(c[nombre], "predict_proba"), nombre


def test_solo_pueden_ser_elegidos_los_modelos_que_calculan_probabilidades():
    assert elegibles() == ["regresión logística", "random forest", "gradient boosting"]


def test_el_entrenamiento_no_emite_avisos_de_deprecacion(recwarn):
    X_train, _, y_train, _ = dividir(cargar_datos())
    candidatos()["svm rbf"].fit(X_train.head(300), y_train.head(300))
    assert not [w for w in recwarn if issubclass(w.category, FutureWarning)]


def test_las_clases_estan_ponderadas_salvo_la_linea_base_y_el_orden_natural():
    assert ORDEN == ("baja", "media", "alta") and train.ORDEN == list(ORDEN)
    c = candidatos()
    assert c["regresión logística"].named_steps["modelo"].class_weight == "balanced"
    assert c["gradient boosting"].named_steps["modelo"].class_weight == "balanced"


def test_entender_y_explorar(capsys, tmp_path):
    crudo = cargar_datos(quitar_duplicados=False)
    entender(cargar_datos(), len(crudo))
    salida = capsys.readouterr().out
    assert "Filas crudas: 6497  Filas sin duplicados: 5329 (1168 duplicadas descartadas)" in salida
    assert "filas con algún nulo: 34" in salida and "{'baja': 0.374, 'media': 0.437, 'alta': 0.19}" in salida
    explorar(cargar_datos(), tmp_path)
    assert sorted(p.name for p in tmp_path.glob("*.png")) == ["clases.png", "correlacion.png", "variables_por_clase.png"]
    assert "alcohol" in capsys.readouterr().out


# --- Entrada estricta, límites, descripciones y textos ---

LIMITES = {"fixed_acidity": (3, 17), "volatile_acidity": (0.05, 2), "citric_acid": (0, 2), "residual_sugar": (0.3, 70), "chlorides": (0.005, 0.7),
           "free_sulfur_dioxide": (0.5, 320), "total_sulfur_dioxide": (5, 460), "density": (0.98, 1.05), "ph": (2.6, 4.2),
           "sulphates": (0.2, 2.1), "alcohol": (7.5, 15.5)}


def test_la_entrada_es_estricta_y_los_12_campos_son_obligatorios():
    esquema = Entrada.model_json_schema()
    assert esquema.get("additionalProperties") is False and set(esquema["required"]) == set(ENTRADA) == set(VARIABLES)


@pytest.mark.parametrize("cambio", [{"alcohol": "10.5"}, {"alcohol": True}, {"alcohol": None}, {"tipo": "White"}, {"tipo": 1}, {"calidad": "alta"},
                                    {"quality": 6}, {"ph": "3.2"}])
def test_tipos_laxos_y_campos_desconocidos_dan_422(cliente, assert_error, cambio):
    assert_error(cliente.post(URL, json={**ENTRADA, **cambio}), 422, "VALIDACION")


@pytest.mark.parametrize("campo", list(LIMITES))
def test_el_limite_de_cada_campo_es_inclusivo_y_rechaza_lo_que_lo_excede(campo):
    minimo, maximo = LIMITES[campo]
    base = {**ENTRADA, "free_sulfur_dioxide": 5.0, "total_sulfur_dioxide": 400.0} if "sulfur" in campo else ENTRADA
    for dentro in (minimo, maximo):
        if campo == "free_sulfur_dioxide" and dentro > 400:
            continue
        Entrada(**{**base, campo: float(dentro)})
    for fuera in (minimo - 0.001, maximo + 0.001):
        with pytest.raises(ValidationError):
            Entrada(**{**base, campo: float(fuera)})


def test_el_dioxido_de_azufre_libre_puede_igualar_pero_no_superar_al_total():
    Entrada(**{**ENTRADA, "free_sulfur_dioxide": 50.0, "total_sulfur_dioxide": 50.0})
    with pytest.raises(ValidationError, match="free_sulfur_dioxide"):
        Entrada(**{**ENTRADA, "free_sulfur_dioxide": 50.1, "total_sulfur_dioxide": 50.0})


@pytest.mark.parametrize("campo", VARIABLES)
def test_cada_campo_tiene_descripcion_y_un_ejemplo_valido(campo):
    info = Entrada.model_fields[campo]
    assert info.description and info.examples
    Entrada(**{**ENTRADA, campo: info.examples[0]})


@pytest.mark.parametrize("campo, unidad", [("fixed_acidity", "g/dm³"), ("residual_sugar", "g/dm³"), ("free_sulfur_dioxide", "mg/dm³"),
                                           ("total_sulfur_dioxide", "mg/dm³"), ("density", "g/cm³"), ("alcohol", "% vol.")])
def test_las_unidades_del_formulario(campo, unidad):
    assert unidad in Entrada.model_fields[campo].description


# --- Modelo entrenado y API ---

@entrenado
def test_el_modelo_elegido_las_metricas_y_el_experimento_de_duplicados_publicados():
    m = leer_metricas(CARPETA)["metricas"]
    assert m["modelo"] == "random forest" and (m["n_entrenamiento"], m["n_prueba"]) == (4263, 1066)
    cv = m["comparacion_cv"]
    assert cv[m["modelo"]]["cv_f1_macro"] == max(v["cv_f1_macro"] for n, v in cv.items() if "línea base" not in n)
    e = m["experimento_duplicados"]
    assert e["filas_duplicadas"] == 1168 and e["prueba_con_copia_en_entrenamiento"] == 356 and e["accuracy_con_duplicados"] > m["accuracy"] + 0.08


@entrenado
def test_el_artefacto_guarda_el_rango_de_las_11_medidas_y_las_variables():
    a = cargar_artefacto(CARPETA, "vino")
    X_train, _, _, _ = dividir(cargar_datos())
    assert a["variables"] == VARIABLES and set(a["rango"]) == set(NUMERICAS)
    for campo, (minimo, maximo) in a["rango"].items():
        assert (minimo, maximo) == (X_train[campo].min(), X_train[campo].max()), campo


@entrenado
def test_las_probabilidades_del_texto_van_en_el_orden_de_calidad_y_coinciden_con_las_de_la_respuesta(cliente):
    cuerpo = cliente.post(URL, json=ENTRADA).json()
    posiciones = [cuerpo["texto"].index(f"{c} {round(cuerpo['probabilidades'][c] * 100)} %") for c in ORDEN]
    assert posiciones == sorted(posiciones)
    assert f"con una probabilidad del {round(cuerpo['probabilidades'][cuerpo['prediccion']] * 100)} por ciento" in cuerpo["texto"]


@entrenado
@pytest.mark.parametrize("clase", ORDEN)
def test_el_texto_da_la_probabilidad_de_la_clase_predicha_para_cada_clase(cliente, clase):
    """Busca vinos reales que el modelo clasifique en cada clase: el porcentaje del texto es el de esa clase y no el de otra."""
    filas = cargar_datos().dropna(subset=NUMERICAS).sample(400, random_state=1)[VARIABLES].to_dict("records")
    fila = next(f for f in filas if predecir_con_pipeline(CARPETA, "vino", f)[0] == clase)
    cuerpo = cliente.post(URL, json=fila).json()
    assert cuerpo["prediccion"] == clase
    assert f"calidad {clase}, con una probabilidad del {round(cuerpo['probabilidades'][clase] * 100)} por ciento" in cuerpo["texto"]


@entrenado
def test_el_umbral_de_poco_concluyente_es_el_50_por_ciento():
    assert UMBRAL_POCO_CONCLUYENTE == 0.5


@entrenado
def test_el_aviso_de_extrapolacion_se_dispara_con_una_sola_medida_fuera_del_rango(cliente):
    rango = cargar_artefacto(CARPETA, "vino")["rango"]
    for campo in ("alcohol", "residual_sugar", "chlorides"):
        minimo, maximo = rango[campo]
        lejos = min(maximo + (maximo - minimo) * 0.2, LIMITES[campo][1])
        assert "poco confiable" in cliente.post(URL, json={**ENTRADA, campo: lejos}).json()["texto"], campo
        assert "poco confiable" not in cliente.post(URL, json={**ENTRADA, campo: min(max(ENTRADA[campo], minimo), maximo)}).json()["texto"], campo


@entrenado
def test_la_api_no_altera_la_entrada_y_devuelve_lo_mismo_que_el_pipeline(cliente):
    cuerpo = cliente.post(URL, json=ENTRADA).json()
    clase, probs = predecir_con_pipeline(CARPETA, "vino", ENTRADA)
    assert cuerpo["prediccion"] == clase and cuerpo["probabilidades"] == probs


@entrenado
def test_los_metadatos_del_modelo_registrado():
    info = REGISTRO["vino"].info
    assert info["slug"] == "vino" and info["tipo"] == "clasificacion" and info["unidad"] is None
    assert info["comandos"] == ["calidad del vino", "clasificar vino", "que tan bueno es el vino"] and REGISTRO["vino"].entrada is Entrada


# --- Entrenamiento (sin tocar el disco) ---

def test_entrenar_devuelve_todo_lo_que_despues_se_publica(reentrenado):
    assert set(reentrenado) == {"pipeline", "metricas", "rango", "entrada_ejemplo"}
    assert set(reentrenado["entrada_ejemplo"]) == set(VARIABLES)


def test_el_modelo_elegido_ofrece_probabilidades_y_no_es_la_linea_base(reentrenado):
    assert hasattr(reentrenado["pipeline"], "predict_proba") and "línea base" not in reentrenado["metricas"]["modelo"]


def test_el_entrenamiento_no_usa_la_prueba_ni_para_elegir_ni_para_ajustar(reentrenado):
    """Se alteran las medidas de las filas de prueba (la partición solo depende de las etiquetas): la selección por validación
    cruzada y el modelo elegido no deben cambiar."""
    base = sin_duplicados(cargar_datos(quitar_duplicados=False))
    _, X_test, _, _ = dividir(base)
    base.loc[X_test.index, "alcohol"] = base.loc[X_test.index, "alcohol"] * 2
    otro = entrenar(base, figuras=None, imprimir=False)
    assert otro["metricas"]["comparacion_cv"] == reentrenado["metricas"]["comparacion_cv"] and otro["metricas"]["modelo"] == reentrenado["metricas"]["modelo"]
    assert otro["metricas"]["accuracy"] != reentrenado["metricas"]["accuracy"]


def test_main_escribe_el_artefacto_las_metricas_y_las_figuras(tmp_path, monkeypatch, capsys):
    shutil.copy(train.CARPETA / "dataset.csv", tmp_path / "dataset.csv")
    monkeypatch.setattr(train, "CARPETA", tmp_path)
    train.main()
    a = joblib.load(tmp_path / "modelo.joblib")
    assert a["variables"] == VARIABLES and set(a["rango"]) == set(NUMERICAS) and (tmp_path / "metricas.json").exists()
    assert sorted(p.name for p in (tmp_path / "figuras").glob("*.png")) == ["clases.png", "correlacion.png", "matriz_confusion.png", "variables_por_clase.png"]
    salida = capsys.readouterr().out
    assert "Modelo elegido: random forest" in salida and "Filas crudas: 6497" in salida and "Por clase:" in salida


@pytest.mark.reproduce
@entrenado
def test_reentrenar_reproduce_exactamente_lo_publicado(reentrenado):
    """Cubre todo `entrenar()`: selección, evaluación por clase, línea base y el experimento de los duplicados."""
    publicado = leer_metricas(CARPETA)
    artefacto = cargar_artefacto(CARPETA, "vino")
    assert reentrenado["metricas"] == publicado["metricas"] and reentrenado["entrada_ejemplo"] == publicado["entrada_ejemplo"]
    assert reentrenado["rango"] == artefacto["rango"]
    _, X_test, _, _ = dividir(cargar_datos())
    assert reentrenado["pipeline"].predict_proba(X_test) == pytest.approx(artefacto["pipeline"].predict_proba(X_test), abs=1e-9)
