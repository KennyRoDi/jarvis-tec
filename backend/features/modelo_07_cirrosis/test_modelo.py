"""Pruebas del modelo 07 · cirrosis (specs/api_rest_spec.md §3). Usan el modelo entrenado real."""
import joblib
import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

from core.modelos import REGISTRO, cargar_artefacto, leer_metricas
from features.modelo_07_cirrosis.router import CARPETA, ETAPAS, Entrada
from features.modelo_07_cirrosis.train import (CATEGORICAS, NUMERICAS, OBJETIVO, ORDEN, SOLO_EN_EXPERIMENTOS, VARIABLES, candidatos,
                                                cargar_datos, casos_completos, dividir, entender, entrenar, explorar, f1_macro_de, metricas_ordinales, por_clase)

URL = "/api/modelos/cirrosis/predecir"
ENTRADA = {"age": 50.0, "sex": "F", "ascites": "N", "hepatomegaly": "Y", "spiders": "N", "edema": "N", "bilirubin": 1.4,
           "cholesterol": 310.0, "albumin": 3.5, "copper": 74.0, "alk_phos": 1277.0, "sgot": 116.0, "tryglicerides": 108.0,
           "platelets": 257.0, "prothrombin": 10.6}
entrenado = pytest.mark.skipif(not REGISTRO["cirrosis"].entrenado, reason="ejecutar features.modelo_07_cirrosis.train")


@pytest.fixture(scope="module")
def reentrenado():
    """Todo el entrenamiento (selección, evaluación, bootstrap, experimentos) en memoria, sin escribir nada."""
    return entrenar(cargar_datos(), figuras=None, imprimir=False)


def filas_completas(n=200, semilla=0):
    return casos_completos(cargar_datos()).sample(n, random_state=semilla)[VARIABLES].to_dict("records")


def puntajes_lote(filas: list[dict]) -> pd.DataFrame:
    pipeline = cargar_artefacto(CARPETA, "cirrosis")["pipeline"]
    return pd.DataFrame(pipeline.predict_proba(pd.DataFrame(filas)), columns=[int(c) for c in pipeline.classes_])


def etapa_esperada(filas: list[dict]) -> np.ndarray:
    p = puntajes_lote(filas)
    return sum(e * p[e] for e in ETAPAS).to_numpy()


# --- Preparación de datos (no requiere el modelo entrenado) ---

def test_los_datos_cargados():
    df = cargar_datos()
    assert len(df) == 412 and df[OBJETIVO].value_counts().sort_index().to_dict() == {1: 21, 2: 92, 3: 155, 4: 144}
    assert 26 < df.age.min() < 27 and 78 < df.age.max() < 79, "la edad se convierte de días a años"
    assert "id" not in df.columns and "drug" not in df.columns


def test_se_entrena_solo_con_los_pacientes_completos():
    completos = casos_completos(cargar_datos())
    assert len(completos) == 276 and not completos[VARIABLES].isna().any().any()
    assert completos[OBJETIVO].value_counts().sort_index().to_dict() == {1: 12, 2: 59, 3: 111, 4: 94}


def test_el_seguimiento_posterior_no_es_una_variable_del_modelo():
    """`n_days` y `status` solo existen después de la consulta: se usan nada más en el experimento."""
    assert SOLO_EN_EXPERIMENTOS == ["n_days", "status"]
    for excluida in SOLO_EN_EXPERIMENTOS + ["drug", "id", "stage", OBJETIVO]:
        assert excluida not in VARIABLES
    assert len(VARIABLES) == 15 and set(VARIABLES) == set(NUMERICAS + CATEGORICAS)


def test_los_faltantes_del_bloque_fuera_del_ensayo_no_dependen_de_la_etapa():
    """Diagnóstico de artefacto: la distribución de etapas es casi la misma con y sin el bloque de 100 pacientes sin ascitis."""
    df = cargar_datos()
    bloque = df[df.ascites.isna()][OBJETIVO].value_counts(normalize=True).sort_index()
    resto = df[df.ascites.notna()][OBJETIVO].value_counts(normalize=True).sort_index()
    assert len(df[df.ascites.isna()]) == 100
    assert (bloque - resto).abs().max() < 0.05


def test_la_particion_es_estratificada_y_conserva_la_etapa_rara():
    train, test, y_train, y_test = dividir(casos_completos(cargar_datos()))
    assert (len(train), len(test)) == (220, 56) and not set(train.index) & set(test.index)
    assert y_train.value_counts().sort_index().to_dict() == {1: 10, 2: 47, 3: 88, 4: 75}
    assert y_test.value_counts().sort_index().to_dict() == {1: 2, 2: 12, 3: 23, 4: 19}


def test_los_candidatos_imputan_ponderan_las_etapas_y_la_linea_base_es_la_mayoritaria():
    for nombre, pipeline in candidatos().items():
        t = {n: (tr, c) for n, tr, c in pipeline.named_steps["pre"].transformers}
        assert t["num"][0].named_steps["imputar"].strategy == "median" and t["num"][1] == NUMERICAS, nombre
        assert t["cat"][0].named_steps["imputar"].strategy == "most_frequent" and t["cat"][1] == CATEGORICAS, nombre
    c = {n: p.named_steps["modelo"] for n, p in candidatos().items()}
    assert isinstance(c["clase mayoritaria (línea base)"], DummyClassifier)
    assert c["regresión logística"].class_weight == "balanced" and isinstance(c["regresión logística"], LogisticRegression)
    assert c["random forest"].class_weight == "balanced_subsample" and isinstance(c["random forest"], RandomForestClassifier)
    assert c["gradient boosting"].class_weight == "balanced" and isinstance(c["gradient boosting"], HistGradientBoostingClassifier)


def test_las_metricas_ordinales_penalizan_los_errores_grandes():
    y = [1, 2, 3, 4]
    assert metricas_ordinales(y, y) == {"mae_etapas": 0.0, "dentro_de_1_etapa": 1.0, "kappa_cuadratico": 1.0}
    cerca, lejos = metricas_ordinales(y, [2, 3, 4, 3]), metricas_ordinales(y, [4, 4, 1, 1])
    assert cerca["mae_etapas"] == 1.0 and cerca["dentro_de_1_etapa"] == 1.0
    assert lejos["mae_etapas"] == 2.5 and lejos["dentro_de_1_etapa"] == 0.0   # errores de 3, 2, 2 y 3 etapas
    assert lejos["kappa_cuadratico"] < cerca["kappa_cuadratico"] < 1.0


def test_las_metricas_por_clase_usan_las_cuatro_etapas_en_orden():
    y, pred = [1, 1, 2, 3, 4, 4], [1, 2, 2, 3, 3, 3]
    assert list(por_clase(y, pred)) == ["1", "2", "3", "4"] and por_clase(y, pred)["4"]["recall"] == 0.0
    assert por_clase(y, pred)["1"]["recall"] == 0.5
    assert f1_macro_de(y, pred) == pytest.approx((0.6667 + 0.6667 + 0.5 + 0.0) / 4, abs=1e-3)
    assert ORDEN == [1, 2, 3, 4]


# --- Validación de la entrada (no requiere el modelo entrenado) ---

def test_la_entrada_acepta_a_todos_los_pacientes_completos():
    for fila in casos_completos(cargar_datos())[VARIABLES].to_dict("records"):
        Entrada(**fila)


def test_los_limites_de_entrada_son_razonables_frente_a_los_datos():
    completos = casos_completos(cargar_datos())
    propiedades = Entrada.model_json_schema()["properties"]
    for campo in NUMERICAS:
        minimo, maximo = completos[campo].min(), completos[campo].max()
        inferior = 0 if minimo < 1 else 0.5 * minimo
        assert inferior <= propiedades[campo]["minimum"] <= minimo, campo
        assert maximo <= propiedades[campo]["maximum"] <= 1.5 * maximo, campo


@pytest.mark.parametrize("campo", NUMERICAS)
def test_el_limite_de_cada_campo_es_inclusivo_y_rechaza_lo_que_lo_excede(campo):
    propiedades = Entrada.model_json_schema()["properties"][campo]
    Entrada(**{**ENTRADA, campo: propiedades["minimum"]})
    Entrada(**{**ENTRADA, campo: propiedades["maximum"]})
    for fuera in (propiedades["minimum"] - 0.001, propiedades["maximum"] + 0.001):
        with pytest.raises(ValidationError):
            Entrada(**{**ENTRADA, campo: fuera})


@pytest.mark.parametrize("campo", NUMERICAS)
def test_cada_campo_numerico_rechaza_cadenas_y_booleanos_por_ser_estricto(cliente, assert_error, campo):
    assert_error(cliente.post(URL, json={**ENTRADA, campo: "10"}), 422, "VALIDACION")
    assert_error(cliente.post(URL, json={**ENTRADA, campo: True}), 422, "VALIDACION")


@pytest.mark.parametrize("campo, valor", [("sex", "m"), ("sex", "X"), ("ascites", "S"), ("ascites", "y"), ("hepatomegaly", "Si"),
                                          ("spiders", 1), ("edema", "no"), ("edema", "YES")])
def test_los_campos_categoricos_solo_aceptan_sus_valores(cliente, assert_error, campo, valor):
    assert_error(cliente.post(URL, json={**ENTRADA, campo: valor}), 422, "VALIDACION")


def test_las_categorias_aceptadas_son_exactamente_las_de_los_datos():
    props = Entrada.model_json_schema()["properties"]
    df = cargar_datos().dropna(subset=CATEGORICAS)
    for campo in CATEGORICAS:
        assert set(props[campo]["enum"]) == set(df[campo].unique()), campo


@pytest.mark.parametrize("extra", [{"n_days": 1500}, {"status": "C"}, {"drug": "Placebo"}, {"id": 1}, {"stage": 3}])
def test_el_seguimiento_posterior_y_otros_campos_no_se_aceptan(cliente, assert_error, extra):
    assert_error(cliente.post(URL, json={**ENTRADA, **extra}), 422, "VALIDACION")


def test_faltan_campos_da_422_y_todos_son_obligatorios(cliente, assert_error):
    assert_error(cliente.post(URL, json={k: v for k, v in ENTRADA.items() if k != "copper"}), 422, "VALIDACION")
    esquema = Entrada.model_json_schema()
    assert set(esquema["required"]) == set(ENTRADA) == set(VARIABLES) and esquema.get("additionalProperties") is False


def test_los_metadatos_del_modelo_registrado():
    info = REGISTRO["cirrosis"].info
    assert info["tipo"] == "clasificacion" and info["unidad"] is None and info["slug"] == "cirrosis"


# --- Modelo entrenado y API ---

@entrenado
def test_el_modelo_guardado_reproduce_las_metricas_publicadas():
    _, test, _, y_test = dividir(casos_completos(cargar_datos()))
    pipeline = joblib.load(CARPETA / "modelo.joblib")["pipeline"]
    m = leer_metricas(CARPETA)["metricas"]
    assert f1_macro_de(y_test, pipeline.predict(test[VARIABLES])) == pytest.approx(m["f1_macro"], abs=1e-3)
    assert m["f1_macro"] > m["baseline"]["f1_macro"] + 0.2 and m["kappa_cuadratico"] > m["baseline"]["kappa_cuadratico"] + 0.2


@entrenado
def test_las_metricas_publicadas_son_coherentes_e_incluyen_los_intervalos():
    m = leer_metricas(CARPETA)["metricas"]
    matriz = np.array(m["matriz_confusion"])
    assert m["orden_clases"] == ORDEN and matriz.shape == (4, 4) and matriz.sum() == m["n_prueba"] == 56
    assert matriz.sum(axis=1).tolist() == [m["casos_por_clase_prueba"][str(e)] for e in ETAPAS] == [2, 12, 23, 19]
    for clave in ("f1_macro", "dentro_de_1_etapa", "kappa_cuadratico"):
        bajo, alto = m["ic95"][clave]
        assert bajo <= m[clave] <= alto, clave
    assert m["ic95"]["f1_macro"][1] - m["ic95"]["f1_macro"][0] > 0.2, "con 56 pacientes el intervalo debe ser muy ancho"


@entrenado
def test_los_pacientes_incompletos_no_mejoran_el_modelo_y_el_seguimiento_posterior_no_entra():
    """Decisiones documentadas: se entrena solo con completos (los incompletos no suman) y n_days/status quedan fuera."""
    m = leer_metricas(CARPETA)["metricas"]
    for modelo, e in m["experimento_pacientes_incompletos"].items():
        assert e["con_incompletos_imputados"]["cv_f1_macro"] - e["solo_completos"]["cv_f1_macro"] < 0.02, modelo
    pipeline = cargar_artefacto(CARPETA, "cirrosis")["pipeline"]
    assert list(pipeline.feature_names_in_) == VARIABLES
    v = m["experimento_variables"]
    assert v["solo N_Days y Status (seguimiento posterior, no se usa)"]["cv_f1_macro"] < 0.3, "por sí solas informan poco"
    assert v["7 básicas (las disponibles en todos los pacientes)"]["cv_f1_macro"] < v["15 (todas las de la consulta inicial)"]["cv_f1_macro"] - 0.05


@entrenado
def test_predecir(cliente):
    cuerpo = cliente.post(URL, json=ENTRADA).json()
    assert cuerpo["modelo"] == "cirrosis" and cuerpo["prediccion"] in ETAPAS and isinstance(cuerpo["prediccion"], int)
    assert set(cuerpo["probabilidades"]) == {"1", "2", "3", "4"} and sum(cuerpo["probabilidades"].values()) == pytest.approx(1, abs=0.01)
    assert str(cuerpo["prediccion"]) == max(cuerpo["probabilidades"], key=cuerpo["probabilidades"].get)
    assert f"etapa histológica {cuerpo['prediccion']} de 4" in cuerpo["texto"]


@entrenado
def test_el_texto_avisa_que_no_es_un_diagnostico_y_lista_las_etapas_en_orden(cliente):
    cuerpo = cliente.post(URL, json=ENTRADA).json()
    texto = cuerpo["texto"]
    assert "no es un diagnóstico" in texto and "biopsia" in texto and "no son probabilidades calibradas" in texto
    posiciones = [texto.index(f"etapa {e} {round(cuerpo['probabilidades'][str(e)] * 100)} %") for e in ETAPAS]
    assert posiciones == sorted(posiciones)


@entrenado
def test_la_api_no_altera_la_entrada_y_devuelve_lo_mismo_que_el_pipeline(cliente):
    pipeline = cargar_artefacto(CARPETA, "cirrosis")["pipeline"]
    filas = filas_completas(40, semilla=5)
    esperado = puntajes_lote(filas)
    for i, fila in enumerate(filas):
        cuerpo = cliente.post(URL, json=fila).json()
        for e in ETAPAS:
            assert cuerpo["probabilidades"][str(e)] == pytest.approx(float(esperado.iloc[i][e]), abs=5e-5), (fila, e)
        assert cuerpo["prediccion"] == int(pipeline.predict(pd.DataFrame([fila]))[0]), fila


@entrenado
def test_el_perfil_tipico_de_la_etapa_4_tiene_mayor_etapa_esperada_que_el_de_la_etapa_1():
    completos = casos_completos(cargar_datos())
    perfiles = {}
    for etapa in (1, 4):
        grupo = completos[completos[OBJETIVO] == etapa]
        perfil = {c: float(grupo[c].median()) for c in NUMERICAS}
        perfil.update({c: grupo[c].mode().iloc[0] for c in CATEGORICAS})
        perfiles[etapa] = perfil
    esperada = etapa_esperada([perfiles[1], perfiles[4]])
    assert esperada[1] > esperada[0] + 0.5


@entrenado
def test_mas_bilirrubina_y_menos_albumina_elevan_la_etapa_esperada_en_pacientes_reales():
    filas = filas_completas(200, semilla=1)
    peor = [{**f, "bilirubin": min(f["bilirubin"] * 3, 39), "albumin": max(f["albumin"] - 0.6, 1.6)} for f in filas]
    assert np.mean(etapa_esperada(peor) > etapa_esperada(filas)) >= 0.8


@entrenado
def test_valores_fuera_de_rango_avisan_que_el_resultado_es_poco_confiable(cliente):
    assert "poco confiable" not in cliente.post(URL, json=ENTRADA).json()["texto"]
    assert "poco confiable" in cliente.post(URL, json={**ENTRADA, "bilirubin": 39.0}).json()["texto"]
    assert "poco confiable" in cliente.post(URL, json={**ENTRADA, "age": 95.0}).json()["texto"]
    rango = cargar_artefacto(CARPETA, "cirrosis")["rango"]
    train, _, _, _ = dividir(casos_completos(cargar_datos()))
    assert set(rango) == set(NUMERICAS)
    for campo, (minimo, maximo) in rango.items():
        assert (minimo, maximo) == (train[campo].min(), train[campo].max()), campo


@pytest.mark.reproduce
@entrenado
def test_reentrenar_reproduce_exactamente_lo_publicado(reentrenado):
    """Cubre todo `entrenar()`: selección, evaluación, bootstrap, predicciones fuera de muestra y experimentos."""
    publicado = leer_metricas(CARPETA)
    artefacto = cargar_artefacto(CARPETA, "cirrosis")
    assert reentrenado["metricas"] == publicado["metricas"]
    assert reentrenado["entrada_ejemplo"] == publicado["entrada_ejemplo"]
    assert reentrenado["rango"] == artefacto["rango"]
    _, test, _, _ = dividir(casos_completos(cargar_datos()))
    esperado = artefacto["pipeline"].predict_proba(test[VARIABLES])
    assert reentrenado["pipeline"].predict_proba(test[VARIABLES]) == pytest.approx(esperado, abs=1e-9)


@entrenado
def test_info_incluye_metricas_esquema_y_ejemplo_valido(cliente):
    cuerpo = cliente.get("/api/modelos/cirrosis/info").json()
    assert {"f1_macro", "accuracy", "kappa_cuadratico", "dentro_de_1_etapa", "mae_etapas", "ic95", "por_clase", "oof_entrenamiento"} <= set(cuerpo["metricas"])
    assert set(ENTRADA) == set(cuerpo["esquema_entrada"]["properties"]) == set(cuerpo["entrada_ejemplo"])
    assert cliente.post(URL, json=cuerpo["entrada_ejemplo"]).status_code == 200


@entrenado
def test_el_artefacto_no_es_excesivo():
    assert (CARPETA / "modelo.joblib").stat().st_size < 5_000_000


# --- Unidades, avisos, bordes del aviso de rango y contrato OpenAPI ---

@pytest.mark.parametrize("campo, unidad", [
    ("age", "años"), ("bilirubin", "mg/dL"), ("cholesterol", "mg/dL"), ("albumin", "g/dL"), ("copper", "µg/día"),
    ("alk_phos", "U/L"), ("sgot", "U/L"), ("tryglicerides", "mg/dL"), ("platelets", "10³/µL"), ("prothrombin", "segundos"),
])
def test_cada_campo_documenta_su_unidad(campo, unidad):
    """SGOT va en U/L y las plaquetas en 10³/µL (la revisión independiente corrigió U/mL y 'miles por mL')."""
    assert unidad in Entrada.model_json_schema()["properties"][campo]["description"]


@entrenado
@pytest.mark.parametrize("campo", ["age", "bilirubin"])
def test_el_aviso_de_rango_empieza_justo_pasado_el_margen_del_5_por_ciento(cliente, campo):
    minimo, maximo = cargar_artefacto(CARPETA, "cirrosis")["rango"][campo]
    margen = (maximo - minimo) * 0.05
    adentro, afuera = {**ENTRADA, campo: maximo + margen * 0.9}, {**ENTRADA, campo: maximo + margen * 1.1}
    assert "poco confiable" not in cliente.post(URL, json=adentro).json()["texto"]
    assert "poco confiable" in cliente.post(URL, json=afuera).json()["texto"]


@entrenado
def test_el_aviso_menciona_la_biopsia_y_que_la_estimacion_puede_equivocarse(cliente):
    texto = cliente.post(URL, json=ENTRADA).json()["texto"]
    assert "se determina por biopsia" in texto and "puede equivocarse por una etapa o más" in texto


def test_el_contrato_openapi_declara_la_respuesta_de_prediccion(cliente):
    esquema = cliente.get("/openapi.json").json()["paths"][URL]["post"]["responses"]["200"]["content"]["application/json"]["schema"]
    assert esquema["$ref"].endswith("RespuestaPrediccion")
    assert "cirrosis" in REGISTRO["cirrosis"].info["nombre"].lower()


def test_entender_describe_el_bloque_fuera_del_ensayo_y_los_pacientes_completos(capsys):
    entender(cargar_datos())
    salida = capsys.readouterr().out
    assert "Pacientes con etapa: 412 | con las 15 variables: 276 | incompletos: 136" in salida
    assert "Bloque de 100 pacientes fuera del ensayo" in salida and "{1: 0.05, 2: 0.25, 3: 0.35, 4: 0.35}" in salida
    assert "{1: 2644.0, 2: 2409.5, 3: 1810.0, 4: 1207.0}" in salida


def test_explorar_genera_las_figuras_del_analisis(tmp_path):
    explorar(cargar_datos(), tmp_path)
    assert sorted(p.name for p in tmp_path.glob("*.png")) == ["etapas.png", "laboratorio_por_etapa.png", "seguimiento_posterior.png", "signos_por_etapa.png"]


@entrenado
def test_el_experimento_de_quitar_cada_variable_cubre_las_15_y_no_hay_ninguna_decisiva():
    """Las diferencias son menores que la desviación entre pliegues (~0.06): el modelo reparte su información entre variables redundantes."""
    m = leer_metricas(CARPETA)["metricas"]
    e = m["experimento_sin_cada_variable"]
    assert set(e["diferencia_al_quitar"]) == set(VARIABLES)
    assert e["cv_f1_macro_15_variables"] == m["experimento_variables"]["15 (todas las de la consulta inicial)"]["cv_f1_macro"]
    assert max(abs(v) for v in e["diferencia_al_quitar"].values()) < 0.05
