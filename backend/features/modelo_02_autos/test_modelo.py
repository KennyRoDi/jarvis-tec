"""Pruebas del modelo 02 · autos (specs/api_rest_spec.md §3). Usan el modelo entrenado real."""
import shutil

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from core.modelos import REGISTRO, cargar_artefacto, leer_metricas, predecir_con_pipeline
from features.modelo_02_autos import train
from features.modelo_02_autos.razon import RazonAlPrecioActual
from features.modelo_02_autos.router import CARPETA, DUENOS_RAROS, Entrada, formato
from features.modelo_02_autos.train import (CATEGORICAS, CON_RANGO, NUMERICAS, OBJETIVO, SEMILLA, VARIABLES, candidatos, cargar_datos, dividir,
                                            entender, entrenar, es_linea_base, explorar)

URL = "/api/modelos/autos/predecir"
ENTRADA = {
    "year": 2014, "present_price": 5.59, "kms_driven": 27000, "fuel_type": "Petrol",
    "seller_type": "Dealer", "transmission": "Manual", "owner": 0,
}
entrenado = pytest.mark.skipif(not REGISTRO["autos"].entrenado, reason="ejecutar features.modelo_02_autos.train")


@pytest.fixture(scope="module")
def reentrenado():
    """Todo el entrenamiento en memoria. Se prohíbe escribir: guardar figuras, el artefacto o cualquier archivo hace fallar la prueba."""
    def prohibido(*a, **k):
        raise AssertionError("entrenar() no debe escribir en disco")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(train, "guardar_figura", prohibido)
        mp.setattr(train, "guardar_modelo", prohibido)
        mp.setattr(plt, "savefig", prohibido)
        mp.setattr(joblib, "dump", prohibido)
        resultado = entrenar(cargar_datos(), figuras=None, imprimir=False)
    return resultado  # el parche solo vale durante el entrenamiento: otras pruebas sí escriben (main)


def precio(cliente, **cambios) -> float:
    return cliente.post(URL, json={**ENTRADA, **cambios}).json()["prediccion"]


# --- Datos (no requieren el modelo entrenado) ---

def test_los_datos_cargados_no_tienen_duplicados_ni_nombres_de_auto():
    df = cargar_datos()
    assert len(df) == 299 and list(df.columns) == ["year", "selling_price", "present_price", "kms_driven", "fuel_type", "seller_type", "transmission", "owner"]
    assert not df.duplicated().any() and df.isna().sum().sum() == 0
    crudo = cargar_datos(quitar_duplicados=False)
    assert len(crudo) == 301 and int(crudo.duplicated().sum()) == 2, "hay 2 autos repetidos que podrían caer en entrenamiento y prueba"
    assert VARIABLES == NUMERICAS + CATEGORICAS and "car_name" not in crudo.columns and OBJETIVO not in VARIABLES


def test_la_razon_de_reventa_sobre_agencia_nunca_supera_uno_y_baja_con_la_antiguedad():
    """Es el hecho en que se apoya el modelo de razón: acotada y casi independiente de la escala del auto."""
    df = cargar_datos()
    razon = df[OBJETIVO] / df.present_price
    assert 0.1 < razon.min() < 0.12 and razon.max() < 1 and razon.median() == pytest.approx(0.65, abs=0.01)
    por_anio = razon.groupby(df.year).median()
    assert por_anio.loc[2017] > por_anio.loc[2013] > por_anio.loc[2008]


def test_las_categorias_raras_que_justifican_el_aviso():
    df = cargar_datos()
    assert df.fuel_type.value_counts().to_dict() == {"Petrol": 239, "Diesel": 58, "CNG": 2}
    assert df.owner.value_counts().sort_index().to_dict() == {0: 288, 1: 10, 3: 1} and DUENOS_RAROS == 2


def test_la_particion_es_80_20_con_la_semilla_fija():
    X_train, X_test, y_train, y_test = dividir(cargar_datos())
    assert (len(X_train), len(X_test)) == (239, 60) and SEMILLA == 42 and not set(X_train.index) & set(X_test.index)
    assert list(X_train.columns) == VARIABLES and y_train.name == OBJETIVO


# --- El regresor de razón y los candidatos ---

def datos_de_razon():
    X = pd.DataFrame({"present_price": [2.0, 4.0, 10.0, 20.0], "x": [1.0, 2.0, 3.0, 4.0]})
    return X, pd.Series([1.0, 3.0, 5.0, 16.0])  # razones: 0.5, 0.75, 0.5, 0.8


def test_el_regresor_de_razon_predice_agencia_por_razon():
    X, y = datos_de_razon()
    modelo = RazonAlPrecioActual(DummyRegressor(strategy="median")).fit(X, y)
    assert modelo.modelo_.constant_ == pytest.approx(0.625), "la mediana de las razones 0.5, 0.5, 0.75, 0.8"
    assert modelo.predict(X) == pytest.approx(0.625 * X.present_price.to_numpy())
    nuevo = pd.DataFrame({"present_price": [100.0], "x": [9.0]})
    assert modelo.predict(nuevo) == pytest.approx([62.5]), "escala con el precio de agencia aunque no haya visto uno tan alto"


def test_el_regresor_de_razon_con_logaritmo_y_sin_modificar_los_datos():
    X, y = datos_de_razon()
    copia_X, copia_y = X.copy(), y.copy()
    modelo = RazonAlPrecioActual(DummyRegressor(strategy="mean"), log=True).fit(X, y)
    razones = y.to_numpy() / X.present_price.to_numpy()
    assert modelo.modelo_.constant_ == pytest.approx(np.log(razones).mean())
    assert modelo.predict(X) == pytest.approx(np.exp(np.log(razones).mean()) * X.present_price.to_numpy()) and (modelo.predict(X) > 0).all()
    assert X.equals(copia_X) and y.equals(copia_y)


def test_el_regresor_de_razon_se_puede_clonar_y_no_comparte_el_modelo_interno():
    from sklearn.base import clone
    original = RazonAlPrecioActual(LinearRegression(), log=True)
    copia = clone(original)
    assert copia.log is True and copia.modelo is not original.modelo
    X, y = datos_de_razon()
    copia.fit(X, y)
    assert not hasattr(original, "modelo_")
    assert copia.modelo_ is not copia.modelo and not hasattr(copia.modelo, "coef_"), "se ajusta una copia, no el modelo recibido"


def test_los_candidatos_son_dos_lineas_base_y_seis_modelos_con_y_sin_razon():
    c = candidatos()
    assert list(c) == ["precio medio (línea base)", "regla de depreciación (línea base)", "regresión lineal", "random forest", "gradient boosting",
                       "random forest (razón)", "gradient boosting (razón)", "regresión lineal (log-razón)"]
    assert [n for n in c if es_linea_base(n)] == ["precio medio (línea base)", "regla de depreciación (línea base)"]
    assert isinstance(c["precio medio (línea base)"].named_steps["modelo"], DummyRegressor)
    regla = c["regla de depreciación (línea base)"]
    assert isinstance(regla, RazonAlPrecioActual) and regla.modelo.named_steps["modelo"].strategy == "median"
    for nombre in ("random forest (razón)", "gradient boosting (razón)", "regresión lineal (log-razón)"):
        assert isinstance(c[nombre], RazonAlPrecioActual), nombre
    assert c["regresión lineal (log-razón)"].log is True and c["random forest (razón)"].log is False
    assert not any(isinstance(c[n], RazonAlPrecioActual) for n in ("regresión lineal", "random forest", "gradient boosting"))


def test_los_modelos_comparten_hiperparametros_con_y_sin_razon():
    c = candidatos()
    bosque = c["random forest"].named_steps["modelo"]
    assert isinstance(bosque, RandomForestRegressor) and (bosque.n_estimators, bosque.random_state) == (200, SEMILLA)
    assert c["random forest (razón)"].modelo.named_steps["modelo"].get_params() == bosque.get_params()
    boosting = c["gradient boosting"].named_steps["modelo"]
    assert isinstance(boosting, HistGradientBoostingRegressor) and boosting.early_stopping is False and boosting.learning_rate == 0.05
    assert c["gradient boosting (razón)"].modelo.named_steps["modelo"].get_params() == boosting.get_params()


def test_el_preprocesamiento_va_dentro_del_pipeline_y_cubre_las_siete_variables():
    pipe = candidatos()["random forest"]
    transformadores = {n: cols for n, _, cols in pipe.named_steps["preprocesador"].transformers}
    assert transformadores == {"num": NUMERICAS, "cat": CATEGORICAS}


def test_entender_y_explorar(capsys, tmp_path):
    entender(cargar_datos(), 301)
    salida = capsys.readouterr().out
    assert "Filas crudas: 301  Filas sin duplicados: 299 (2 duplicadas descartadas)  Columnas: 8" in salida
    assert "Combustible: {'Petrol': 239, 'Diesel': 58, 'CNG': 2}" in salida and "mínimo 0.11, mediana 0.65, máximo 0.99" in salida
    explorar(cargar_datos(), tmp_path)
    assert sorted(p.name for p in tmp_path.glob("*.png")) == ["correlacion.png", "distribucion_objetivo.png", "present_vs_selling.png", "razon_por_anio.png"]
    assert "'present_price': 0.88" in capsys.readouterr().out


# --- Entrada estricta, límites, descripciones y textos ---

LIMITES = {"year": (1990, 2026), "present_price": (0, 150), "kms_driven": (0, 1_000_000), "owner": (0, 3)}


def test_la_entrada_es_estricta_y_los_7_campos_son_obligatorios():
    esquema = Entrada.model_json_schema()
    assert esquema.get("additionalProperties") is False and set(esquema["required"]) == set(ENTRADA) == set(VARIABLES)


@pytest.mark.parametrize("campo", list(ENTRADA))
def test_cada_campo_rechaza_cadenas_numeros_como_texto_y_booleanos(cliente, assert_error, campo):
    assert_error(cliente.post(URL, json={**ENTRADA, campo: str(ENTRADA[campo]) + "x"}), 422, "VALIDACION")
    assert_error(cliente.post(URL, json={**ENTRADA, campo: True}), 422, "VALIDACION")
    assert_error(cliente.post(URL, json={**ENTRADA, campo: None}), 422, "VALIDACION")


@pytest.mark.parametrize("cambio", [{"year": 2014.5}, {"year": "2014"}, {"kms_driven": 27000.5}, {"owner": 0.5}, {"fuel_type": "Agua"}, {"fuel_type": "petrol"},
                                    {"seller_type": "Dealer "}, {"transmission": "Semi"}, {"modelo": "ritz"}, {"car_name": "ritz"}, {"selling_price": 3}])
def test_tipos_laxos_valores_desconocidos_y_campos_extra_dan_422(cliente, assert_error, cambio):
    assert_error(cliente.post(URL, json={**ENTRADA, **cambio}), 422, "VALIDACION")


@pytest.mark.parametrize("campo", list(LIMITES))
def test_el_limite_de_cada_campo_numerico_rechaza_lo_que_lo_excede(campo):
    minimo, maximo = LIMITES[campo]
    tipo = float if campo == "present_price" else int
    Entrada(**{**ENTRADA, campo: tipo(maximo)})
    if campo == "present_price":
        with pytest.raises(ValidationError):
            Entrada(**{**ENTRADA, campo: 0.0})  # gt=0: el cero se rechaza
        Entrada(**{**ENTRADA, campo: 0.01})
    else:
        Entrada(**{**ENTRADA, campo: tipo(minimo)})
        with pytest.raises(ValidationError):
            Entrada(**{**ENTRADA, campo: tipo(minimo - 1)})
    with pytest.raises(ValidationError):
        Entrada(**{**ENTRADA, campo: tipo(maximo + 1)})


def test_los_limites_contienen_a_todos_los_autos_reales():
    df = cargar_datos(quitar_duplicados=False)
    for campo, (minimo, maximo) in LIMITES.items():
        assert df[campo].min() >= minimo and df[campo].max() <= maximo, campo


@pytest.mark.parametrize("campo, texto", [("year", "fabricación"), ("present_price", "lakhs"), ("kms_driven", "km"), ("fuel_type", "gas natural"),
                                          ("seller_type", "agencia"), ("transmission", "automática"), ("owner", "dueños")])
def test_cada_campo_tiene_descripcion_y_un_ejemplo_valido(campo, texto):
    info = Entrada.model_fields[campo]
    assert texto in info.description and info.examples
    Entrada(**{**ENTRADA, campo: info.examples[0]})


def test_formato_de_numeros():
    assert formato(348000.0) == "348 000" and formato(4764000.4) == "4 764 000" and formato(0.4) == "0"


@entrenado
def test_predecir(cliente):
    cuerpo = cliente.post(URL, json=ENTRADA).json()
    assert cuerpo["modelo"] == "autos" and cuerpo["unidad"] == "lakhs INR" and cuerpo["probabilidades"] is None
    assert 0 < cuerpo["prediccion"] < ENTRADA["present_price"]
    assert "poco confiable" not in cuerpo["texto"]


@entrenado
def test_el_texto_da_el_precio_en_rupias_y_el_porcentaje_del_precio_de_agencia(cliente):
    cuerpo = cliente.post(URL, json=ENTRADA).json()
    v = cuerpo["prediccion"]
    assert f"estimado del vehículo es de {v} lakhs de rupias indias, unas {formato(v * 100_000)} rupias" in cuerpo["texto"]
    assert f"alrededor del {round(v / 5.59 * 100)} por ciento de su precio de agencia" in cuerpo["texto"]


@entrenado
def test_la_prediccion_se_redondea_a_dos_decimales_y_coincide_con_el_pipeline(cliente):
    crudo, _ = predecir_con_pipeline(CARPETA, "autos", ENTRADA)
    assert abs(crudo - round(crudo, 2)) > 1e-9, "el valor sin redondear tiene más decimales: la prueba sí ejercita el redondeo"
    assert precio(cliente) == round(crudo, 2)


@entrenado
def test_el_precio_de_reventa_nunca_supera_al_de_agencia_en_ningun_auto_real(cliente):
    """Propiedad del modelo de razón: la razón aprendida es menor que 1, así que reventa < agencia."""
    filas = cargar_datos().sample(60, random_state=0)[VARIABLES].to_dict("records")
    for fila in filas:
        assert 0 < cliente.post(URL, json=fila).json()["prediccion"] < fila["present_price"]


@entrenado
def test_un_auto_mas_caro_que_cualquiera_del_entrenamiento_se_estima_con_sentido(cliente):
    """Un árbol en niveles no podría pasar de lo visto (35 lakhs de reventa); con la razón, 90 de agencia da bastante más."""
    cuerpo = cliente.post(URL, json={**ENTRADA, "present_price": 90.0, "year": 2012, "kms_driven": 78000, "fuel_type": "Diesel", "transmission": "Automatic"}).json()
    assert 35 < cuerpo["prediccion"] < 90
    assert "poco confiable" not in cuerpo["texto"], "90 está dentro del rango visto (hasta 92.6)"


@entrenado
def test_la_prediccion_crece_con_el_precio_de_agencia_y_baja_con_la_antiguedad(cliente):
    assert precio(cliente, present_price=12.0) > precio(cliente) > precio(cliente, present_price=2.0)
    assert precio(cliente, year=2017) > precio(cliente, year=2014) > precio(cliente, year=2008)


@entrenado
def test_los_avisos_de_rango_y_de_categorias_raras(cliente):
    rango = cargar_artefacto(CARPETA, "autos")["rango"]
    for campo, fuera in (("year", 1991), ("kms_driven", 900_000), ("present_price", 140.0)):
        assert fuera < rango[campo][0] or fuera > rango[campo][1]
        assert "Atención: el año, el kilometraje o el precio de agencia está fuera del rango de los autos con que se entrenó" in cliente.post(URL, json={**ENTRADA, campo: fuera}).json()["texto"], campo
    for cambio in ({"fuel_type": "CNG"}, {"owner": 2}, {"owner": 3}):
        assert "muy pocos autos de gas natural o con varios dueños" in cliente.post(URL, json={**ENTRADA, **cambio}).json()["texto"], cambio
    for cambio in ({"fuel_type": "Diesel"}, {"owner": 1}):
        assert "poco confiable" not in cliente.post(URL, json={**ENTRADA, **cambio}).json()["texto"], cambio


@entrenado
def test_el_margen_del_aviso_es_el_5_por_ciento_del_rango(cliente):
    minimo, maximo = cargar_artefacto(CARPETA, "autos")["rango"]["kms_driven"]
    ancho = maximo - minimo
    assert "poco confiable" not in cliente.post(URL, json={**ENTRADA, "kms_driven": int(maximo + 0.04 * ancho)}).json()["texto"]
    assert "poco confiable" in cliente.post(URL, json={**ENTRADA, "kms_driven": int(maximo + 0.06 * ancho)}).json()["texto"]


@entrenado
def test_los_metadatos_del_modelo_registrado():
    info = REGISTRO["autos"].info
    assert info["slug"] == "autos" and info["unidad"] == "lakhs INR" and info["tipo"] == "regresion"
    assert info["comandos"] == ["precio de un auto", "precio de un carro", "cuanto vale mi carro"] and REGISTRO["autos"].entrada is Entrada


@entrenado
def test_info_incluye_metricas_y_esquema(cliente):
    cuerpo = cliente.get("/api/modelos/autos/info").json()
    assert {"r2", "mae", "rmse", "ic95", "comparacion_cv", "experimento_extrapolacion"} <= set(cuerpo["metricas"])
    assert set(ENTRADA) == set(cuerpo["esquema_entrada"]["properties"]) == set(cuerpo["entrada_ejemplo"])
    assert cuerpo["esquema_entrada"]["properties"]["fuel_type"]["enum"] == ["Petrol", "Diesel", "CNG"]
    assert isinstance(cuerpo["entrada_ejemplo"]["year"], int) and cliente.post(URL, json=cuerpo["entrada_ejemplo"]).status_code == 200


# --- Modelo entrenado: lo publicado ---

@entrenado
def test_el_artefacto_guarda_el_rango_de_los_tres_campos_y_la_marca_de_reentrenado():
    a = cargar_artefacto(CARPETA, "autos")
    df = cargar_datos()
    assert a["variables"] == VARIABLES and a["reentrenado_con_todo"] is True and isinstance(a["pipeline"], RazonAlPrecioActual)
    assert set(a["rango"]) == set(CON_RANGO) == {"year", "present_price", "kms_driven"}
    for campo, (minimo, maximo) in a["rango"].items():
        assert (minimo, maximo) == (df[campo].min(), df[campo].max()), campo


@entrenado
def test_el_modelo_elegido_las_metricas_y_la_comparacion_publicadas():
    m = leer_metricas(CARPETA)["metricas"]
    cv = m["comparacion_cv"]
    assert m["modelo"] == "random forest (razón)" and (m["n_entrenamiento"], m["n_prueba"]) == (239, 60)
    assert cv[m["modelo"]]["cv_rmse"] == min(v["cv_rmse"] for n, v in cv.items() if not es_linea_base(n)) and m["cv_r2_media"] == cv[m["modelo"]]["cv_r2"]
    assert m["rmse"] < m["baseline_depreciacion"]["rmse"] < m["baseline_media"]["rmse"] and m["r2"] > 0.9
    bajo, alto = m["ic95"]["r2"]
    assert bajo < m["r2"] < alto and alto - bajo > 0.02, "con 60 autos el intervalo no puede ser estrecho"


@entrenado
def test_la_razon_gana_con_claridad_en_validacion_cruzada_y_en_extrapolacion():
    m = leer_metricas(CARPETA)["metricas"]
    cv = m["comparacion_cv"]
    assert cv["random forest (razón)"]["cv_rmse"] < 0.6 * cv["random forest"]["cv_rmse"], "la razón casi reduce a la mitad el error del bosque en niveles"
    e = m["experimento_extrapolacion"]
    assert e["autos_caros_de_prueba"] == 28 and e["random forest"]["rmse"] > 3 * e[m["modelo"]]["rmse"]
    assert e["random forest"]["sesgo"] < -3 and abs(e[m["modelo"]]["sesgo"]) < 1, "en niveles el árbol subestima los autos caros; con razón casi no hay sesgo"


@entrenado
def test_el_modelo_en_niveles_pierde_en_la_prueba_aunque_el_r2_antiguo_era_alto():
    """El R² de 0.962 publicado antes dependía de la partición: con los duplicados quitados el bosque en niveles saca menos en la prueba."""
    otros = leer_metricas(CARPETA)["metricas"]["prueba_de_los_demas_candidatos"]
    assert otros["random forest"]["r2"] < 0.6 and set(otros) == set(candidatos()) - {"random forest (razón)"}


@entrenado
def test_el_artefacto_no_es_excesivo():
    assert (CARPETA / "modelo.joblib").stat().st_size < 5_000_000


# --- Entrenamiento (sin tocar el disco) ---

def test_entrenar_devuelve_todo_lo_que_despues_se_publica(reentrenado):
    assert set(reentrenado) == {"pipeline", "metricas", "rango", "entrada_ejemplo"}
    assert set(reentrenado["entrada_ejemplo"]) == set(VARIABLES) and set(reentrenado["rango"]) == set(CON_RANGO)


def test_el_modelo_elegido_nunca_es_una_linea_base(reentrenado):
    assert not es_linea_base(reentrenado["metricas"]["modelo"])


def test_el_modelo_final_se_reentrena_con_todos_los_autos(reentrenado):
    df = cargar_datos()
    esperado = candidatos()[reentrenado["metricas"]["modelo"]].fit(df[VARIABLES], df[OBJETIVO])
    assert reentrenado["pipeline"].predict(df[VARIABLES]) == pytest.approx(esperado.predict(df[VARIABLES]))


def test_el_entrenamiento_no_usa_la_prueba_ni_para_elegir_ni_para_ajustar(reentrenado):
    """Se alteran los precios de las filas de prueba (la partición no depende del precio): la selección no debe cambiar."""
    df = cargar_datos()
    _, X_test, _, _ = dividir(df)
    alterado = df.copy()
    alterado.loc[X_test.index, OBJETIVO] = alterado.loc[X_test.index, OBJETIVO] * 3
    otro = entrenar(alterado, figuras=None, imprimir=False)
    assert otro["metricas"]["comparacion_cv"] == reentrenado["metricas"]["comparacion_cv"] and otro["metricas"]["modelo"] == reentrenado["metricas"]["modelo"]
    assert otro["metricas"]["rmse"] != reentrenado["metricas"]["rmse"]


def test_main_escribe_el_artefacto_las_metricas_y_las_figuras(tmp_path, monkeypatch, capsys):
    shutil.copy(train.CARPETA / "dataset.csv", tmp_path / "dataset.csv")
    monkeypatch.setattr(train, "CARPETA", tmp_path)
    train.main()
    a = joblib.load(tmp_path / "modelo.joblib")
    assert a["reentrenado_con_todo"] is True and a["variables"] == VARIABLES and set(a["rango"]) == set(CON_RANGO) and (tmp_path / "metricas.json").exists()
    assert sorted(p.name for p in (tmp_path / "figuras").glob("*.png")) == ["correlacion.png", "distribucion_objetivo.png", "present_vs_selling.png",
                                                                            "razon_por_anio.png", "real_vs_predicho.png"]
    salida = capsys.readouterr().out
    assert "Modelo elegido: random forest (razón)" in salida and "Filas crudas: 301" in salida and "Extrapolación a autos caros" in salida


@pytest.mark.reproduce
@entrenado
def test_reentrenar_reproduce_exactamente_lo_publicado(reentrenado):
    """Cubre todo `entrenar()`: selección por validación cruzada repetida, evaluación con IC, líneas base y el experimento de extrapolación."""
    publicado = leer_metricas(CARPETA)
    artefacto = cargar_artefacto(CARPETA, "autos")
    assert reentrenado["metricas"] == publicado["metricas"] and reentrenado["entrada_ejemplo"] == publicado["entrada_ejemplo"]
    assert reentrenado["rango"] == artefacto["rango"]
    X = cargar_datos()[VARIABLES]
    assert reentrenado["pipeline"].predict(X) == pytest.approx(artefacto["pipeline"].predict(X), abs=1e-9)
