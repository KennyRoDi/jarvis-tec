"""Modelo 07 · Clasificación de la etapa histológica de la cirrosis biliar primaria (1 a 4).

Ejecutar desde backend/:  python -m features.modelo_07_cirrosis.train
Dataset: https://www.kaggle.com/fedesoriano/cirrhosis-prediction-dataset (ensayo clínico de la Clínica Mayo, 418 pacientes)

Sigue las 6 etapas de la rúbrica; la redacción de cada etapa va en analisis.md.
No es una herramienta de diagnóstico: la etapa se determina por biopsia.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score, classification_report
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from core.entrenamiento import carpeta_figuras, guardar_figura, guardar_modelo, intervalo_bootstrap, metricas_clasificacion

CARPETA = Path(__file__).parent
OBJETIVO = "etapa"
ORDEN = [1, 2, 3, 4]  # etapa histológica: 1 (más temprana) a 4 (cirrosis)
NUMERICAS = ["age", "bilirubin", "cholesterol", "albumin", "copper", "alk_phos", "sgot", "tryglicerides", "platelets", "prothrombin"]
CATEGORICAS = ["sex", "ascites", "hepatomegaly", "spiders", "edema"]
VARIABLES = NUMERICAS + CATEGORICAS  # 15 variables de la consulta inicial
# Excluidas: `n_days` y `status` (seguimiento POSTERIOR: fuga de información), `drug` (tratamiento aleatorizado, vacío en los
# pacientes fuera del ensayo) e `id`.
SEMILLA = 42
CV = dict(scoring="f1_macro", n_jobs=-1, error_score="raise")
SOLO_EN_EXPERIMENTOS = ["n_days", "status"]  # nunca entran al modelo: solo para mostrar cuánto inflarían la métrica
BASICAS = ["age", "sex", "edema", "bilirubin", "albumin", "platelets", "prothrombin"]  # disponibles en los 412 pacientes
CONJUNTOS = {
    "15 (todas las de la consulta inicial)": VARIABLES,
    "13 (sin colesterol ni triglicéridos)": [v for v in VARIABLES if v not in ("cholesterol", "tryglicerides")],
    "7 básicas (las disponibles en todos los pacientes)": BASICAS,
    "15 + N_Days y Status (seguimiento posterior: no existe en la consulta, no se usa)": VARIABLES + SOLO_EN_EXPERIMENTOS,
    "solo N_Days y Status (seguimiento posterior, no se usa)": SOLO_EN_EXPERIMENTOS,
}

# 1. Análisis del problema: estimar la etapa histológica (1 a 4) de un paciente con cirrosis biliar primaria a partir de signos
#    clínicos y análisis de laboratorio, sin biopsia. Clasificación multiclase ordinal. Ver analisis.md.


def cargar_datos() -> pd.DataFrame:
    """Lee el CSV crudo: edad en años, nombres snake_case y descarta los 6 pacientes sin etapa. Conserva `n_days` y `status`
    solo para el experimento de fuga; `drug` y `id` se descartan."""
    df = pd.read_csv(CARPETA / "dataset.csv").drop(columns=["ID", "Drug"])
    df.columns = [c.lower() for c in df.columns]
    df["age"] = df["age"] / 365.25  # el original trae la edad en días
    df = df.dropna(subset=["stage"]).reset_index(drop=True)
    df[OBJETIVO] = df.pop("stage").astype(int)
    return df


def casos_completos(df: pd.DataFrame) -> pd.DataFrame:
    """Los 276 pacientes con las 15 variables: el modelo se entrena solo con ellos (ver `experimento_pacientes_incompletos`)."""
    return df.dropna(subset=VARIABLES).reset_index(drop=True)


def entender(df: pd.DataFrame) -> None:
    """2. Entendimiento de los datos."""
    completos = casos_completos(df)
    print(f"Pacientes con etapa: {len(df)} | con las 15 variables: {len(completos)} | incompletos: {len(df) - len(completos)}")
    print("Etapas (todos):", df[OBJETIVO].value_counts().sort_index().to_dict(), "| completos:", completos[OBJETIVO].value_counts().sort_index().to_dict())
    bloque = df[df["ascites"].isna()]
    print(f"\nBloque de {len(bloque)} pacientes fuera del ensayo (sin ascitis, cobre, SGOT...): etapas",
          bloque[OBJETIVO].value_counts(normalize=True).sort_index().round(2).to_dict(), "| resto:",
          df[df["ascites"].notna()][OBJETIVO].value_counts(normalize=True).sort_index().round(2).to_dict())
    print("Nulos:", df[VARIABLES].isna().sum()[lambda s: s > 0].to_dict())
    print("\nEdad (años):", df.age.describe().round(1).to_dict())
    print("N_Days mediana por etapa (seguimiento posterior):", df.groupby(OBJETIVO).n_days.median().to_dict())
    print("Resumen de laboratorio:\n", completos[NUMERICAS].describe().round(1).T)


def explorar(df: pd.DataFrame, figuras: Path) -> None:
    """3. Exploración de los datos."""
    completos = casos_completos(df)
    sns.countplot(data=completos, x=OBJETIVO)
    plt.title("Pacientes completos por etapa")
    guardar_figura(figuras, "etapas")

    fig, ejes = plt.subplots(2, 4, figsize=(15, 7))
    for eje, variable in zip(ejes.ravel(), ["bilirubin", "albumin", "copper", "platelets", "prothrombin", "alk_phos", "sgot", "age"]):
        sns.boxplot(data=completos, x=OBJETIVO, y=variable, ax=eje)
        eje.set_title(variable)
        if variable in ("copper", "alk_phos", "bilirubin"):
            eje.set_yscale("log")
    guardar_figura(figuras, "laboratorio_por_etapa")

    fig, ejes = plt.subplots(1, 4, figsize=(14, 3.6))
    for eje, variable in zip(ejes, ["hepatomegaly", "spiders", "ascites", "edema"]):
        pd.crosstab(completos[OBJETIVO], completos[variable], normalize="index").plot.bar(stacked=True, ax=eje, legend=variable == "edema")
        eje.set_title(variable)
        eje.set_xlabel("etapa")
    guardar_figura(figuras, "signos_por_etapa")

    seguimiento = df.groupby(OBJETIVO).agg(dias=("n_days", "median"), fallecidos=("status", lambda s: (s == "D").mean()))
    seguimiento.plot.bar(subplots=True, legend=False, figsize=(6, 5))
    guardar_figura(figuras, "seguimiento_posterior")


def armar(estimador, variables: list[str] | None = None, escalar: bool = True) -> Pipeline:
    """4. Modelo: imputación (mediana / más frecuente) + escala de las numéricas, one-hot de las categóricas y estimador."""
    variables = VARIABLES if variables is None else variables
    categoricas = [v for v in variables if v in CATEGORICAS + ["status"]]
    numericas = [v for v in variables if v not in categoricas]
    columnas = ColumnTransformer([
        ("num", Pipeline([("imputar", SimpleImputer(strategy="median"))] + ([("escala", StandardScaler())] if escalar else [])), numericas),
        ("cat", Pipeline([("imputar", SimpleImputer(strategy="most_frequent")), ("codificar", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categoricas),
    ])
    return Pipeline([("pre", columnas), ("modelo", estimador)])


def candidatos() -> dict[str, Pipeline]:
    """Clases ponderadas (la etapa 1 tiene 9 pacientes en el entrenamiento): los puntajes NO son probabilidades calibradas."""
    return {
        "clase mayoritaria (línea base)": armar(DummyClassifier(strategy="most_frequent")),
        "regresión logística": armar(LogisticRegression(max_iter=5000, class_weight="balanced")),
        "random forest": armar(RandomForestClassifier(n_estimators=200, min_samples_leaf=3, max_depth=8,
                                                      class_weight="balanced_subsample", random_state=SEMILLA, n_jobs=-1), escalar=False),
        "gradient boosting": armar(HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05, max_depth=3, class_weight="balanced",
                                                                  early_stopping=False, random_state=SEMILLA), escalar=False),
    }


def dividir(completos: pd.DataFrame):
    """Partición 80/20 **estratificada** de los pacientes completos (la etapa 1 tiene solo 12)."""
    return train_test_split(completos, completos[OBJETIVO], test_size=0.2, stratify=completos[OBJETIVO], random_state=SEMILLA)


def metricas_ordinales(y_real, y_pred) -> dict:
    """La etapa es ordinal: equivocarse por 3 etapas es peor que por 1."""
    y, p = np.asarray(y_real), np.asarray(y_pred)
    return {"mae_etapas": round(float(np.abs(y - p).mean()), 4), "dentro_de_1_etapa": round(float((np.abs(y - p) <= 1).mean()), 4),
            "kappa_cuadratico": round(float(cohen_kappa_score(y, p, weights="quadratic")), 4)}


def por_clase(y_real, y_pred) -> dict:
    informe = classification_report(y_real, y_pred, labels=ORDEN, output_dict=True, zero_division=0)
    return {str(c): {k: round(float(v), 3) for k, v in informe[str(c)].items()} for c in ORDEN}


def f1_macro_de(y, pred) -> float:
    return float(f1_score(y, pred, labels=ORDEN, average="macro", zero_division=0))


def evaluar(pipeline: Pipeline, X_test, y_test, figuras: Path | None = None) -> dict:
    """5. Evaluación sobre el conjunto de prueba (solo 56 pacientes: 2 o 3 en la etapa 1)."""
    pred = pipeline.predict(X_test)
    metricas = metricas_clasificacion(y_test, pred, orden=ORDEN)
    metricas.update(metricas_ordinales(y_test, pred))
    metricas["por_clase"] = por_clase(y_test, pred)
    metricas["ic95"] = {
        "f1_macro": intervalo_bootstrap(np.asarray(y_test), pred, f1_macro_de),
        "dentro_de_1_etapa": intervalo_bootstrap(np.asarray(y_test), pred, lambda y, p: float((np.abs(y - p) <= 1).mean())),
        "kappa_cuadratico": intervalo_bootstrap(np.asarray(y_test), pred, lambda y, p: float(cohen_kappa_score(y, p, weights="quadratic"))),
    }
    if figuras is None:  # sin efectos en disco (pruebas)
        return metricas
    sns.heatmap(confusion_matrix(y_test, pred, labels=ORDEN), annot=True, fmt="d", cmap="Blues", xticklabels=ORDEN, yticklabels=ORDEN)
    plt.xlabel("Etapa predicha")
    plt.ylabel("Etapa real")
    plt.title("Matriz de confusión (prueba)")
    guardar_figura(figuras, "matriz_confusion")
    return metricas


def cv_con_incompletas(df_completo_train: pd.DataFrame, incompletos: pd.DataFrame, validacion, estimador_fn, escalar: bool) -> tuple[float, float]:
    """F1 macro en pliegues de pacientes completos, entrenando con o sin los pacientes incompletos (imputados) añadidos."""
    f1s = []
    for entrena, valida in validacion.split(df_completo_train, df_completo_train[OBJETIVO]):
        base = df_completo_train.iloc[entrena]
        datos = pd.concat([base, incompletos]) if len(incompletos) else base
        modelo = armar(estimador_fn(), escalar=escalar).fit(datos[VARIABLES], datos[OBJETIVO])
        f1s.append(f1_macro_de(df_completo_train.iloc[valida][OBJETIVO], modelo.predict(df_completo_train.iloc[valida][VARIABLES])))
    return float(np.mean(f1s)), float(np.std(f1s))


def entrenar(df: pd.DataFrame, figuras: Path | None = None, imprimir: bool = True) -> dict:
    """Etapas 4 y 5 completas, **sin escribir nada en disco** si `figuras` es None: las pruebas lo reentrenan y
    exigen reproducir exactamente lo publicado (metricas.json y el artefacto)."""
    log = print if imprimir else (lambda *a, **k: None)
    completos = casos_completos(df)
    incompletos = df[~df.index.isin(df.dropna(subset=VARIABLES).index)]
    train, test, y_train, y_test = dividir(completos)
    X_train, X_test = train[VARIABLES], test[VARIABLES]

    # La selección usa solo el entrenamiento (validación cruzada estratificada repetida) con el F1 macro.
    validacion = RepeatedStratifiedKFold(n_splits=5, n_repeats=4, random_state=SEMILLA)
    comparacion = {}
    for nombre, pipe in candidatos().items():
        f1 = cross_val_score(pipe, X_train, y_train, cv=validacion, **CV)
        bal = cross_val_score(pipe, X_train, y_train, cv=validacion, **{**CV, "scoring": "balanced_accuracy"})
        comparacion[nombre] = {"cv_f1_macro": round(float(f1.mean()), 4), "cv_f1_desv": round(float(f1.std()), 4), "cv_balanced_accuracy": round(float(bal.mean()), 4)}
        log(f"{nombre:32s} F1 macro cv = {f1.mean():.3f} ± {f1.std():.3f}   exactitud balanceada cv = {bal.mean():.3f}")
    ganador = max((n for n in comparacion if "línea base" not in n), key=lambda n: comparacion[n]["cv_f1_macro"])
    log(f"\nModelo elegido: {ganador}")
    pipeline = candidatos()[ganador].fit(X_train, y_train)

    metricas = evaluar(pipeline, X_test, y_test, figuras)
    metricas["modelo"] = ganador
    metricas["comparacion_cv"] = comparacion
    base = candidatos()["clase mayoritaria (línea base)"].fit(X_train, y_train)
    metricas["baseline"] = {k: v for k, v in {**metricas_clasificacion(y_test, base.predict(X_test), orden=ORDEN), **metricas_ordinales(y_test, base.predict(X_test))}.items() if k != "matriz_confusion"}
    metricas["n_entrenamiento"], metricas["n_prueba"] = len(X_train), len(X_test)
    metricas["casos_por_clase_prueba"] = {str(c): int((y_test == c).sum()) for c in ORDEN}
    metricas["casos_por_clase_entrenamiento"] = {str(c): int((y_train == c).sum()) for c in ORDEN}

    # Predicciones fuera de muestra del entrenamiento (220 pacientes): estimaciones menos ruidosas que la prueba de 56.
    oof = cross_val_predict(candidatos()[ganador], X_train, y_train, cv=StratifiedKFold(5, shuffle=True, random_state=SEMILLA))
    metricas["oof_entrenamiento"] = {**{k: v for k, v in metricas_clasificacion(y_train, oof, orden=ORDEN).items() if k in ("f1_macro", "accuracy", "recall_macro")},
                                      **metricas_ordinales(y_train, oof), "por_clase": por_clase(y_train, oof)}

    # Experimento 1: ¿qué variables hacen falta? (regresión logística; las filas son siempre las mismas, solo cambian las columnas).
    experimento = {}
    for nombre, cols in CONJUNTOS.items():
        f1 = cross_val_score(armar(LogisticRegression(max_iter=5000, class_weight="balanced"), cols), train[cols], y_train, cv=validacion, **CV)
        experimento[nombre] = {"variables": len(cols), "cv_f1_macro": round(float(f1.mean()), 4), "cv_f1_desv": round(float(f1.std()), 4)}
        log(f"  {nombre:66s} F1 macro cv = {f1.mean():.3f} ± {f1.std():.3f}")
    metricas["experimento_variables"] = experimento

    # Experimento 2: ¿sirven los 136 pacientes incompletos (fuera del ensayo o con vacíos)? Mismos pliegues, con y sin ellos.
    parametros = candidatos()["random forest"].named_steps["modelo"].get_params()
    experimento_inc = {}
    for nombre, (fn, escala) in {"regresión logística": (lambda: LogisticRegression(max_iter=5000, class_weight="balanced"), True),
                                 "random forest": (lambda: RandomForestClassifier(**parametros), False)}.items():
        solo, ds = cv_con_incompletas(train, incompletos.iloc[:0], validacion, fn, escala)
        con, dc = cv_con_incompletas(train, incompletos, validacion, fn, escala)
        experimento_inc[nombre] = {"solo_completos": {"cv_f1_macro": round(solo, 4), "cv_f1_desv": round(ds, 4)},
                                   "con_incompletos_imputados": {"cv_f1_macro": round(con, 4), "cv_f1_desv": round(dc, 4)}, "pacientes_incompletos": len(incompletos)}
        log(f"  {nombre:20s} solo completos {solo:.3f} ± {ds:.3f} | con {len(incompletos)} incompletos imputados {con:.3f} ± {dc:.3f}")
    metricas["experimento_pacientes_incompletos"] = experimento_inc

    # Experimento 3: ¿qué variables usa realmente el modelo? F1 macro (regresión logística, validación cruzada 5 × 4, solo
    # entrenamiento) quitando una variable cada vez. Las medianas por etapa describen los datos; esto mide lo que el modelo usa.
    completa = cross_val_score(armar(LogisticRegression(max_iter=5000, class_weight="balanced")), X_train, y_train, cv=validacion, **CV).mean()
    sin_cada = {}
    for variable in VARIABLES:
        cols = [v for v in VARIABLES if v != variable]
        f1 = cross_val_score(armar(LogisticRegression(max_iter=5000, class_weight="balanced"), cols), train[cols], y_train, cv=validacion, **CV).mean()
        sin_cada[variable] = round(float(f1 - completa), 4)  # diferencia respecto de las 15 variables (negativa = la variable ayuda)
    metricas["experimento_sin_cada_variable"] = {"cv_f1_macro_15_variables": round(float(completa), 4), "diferencia_al_quitar": sin_cada}
    log("  Diferencia de F1 macro al quitar cada variable:", {k: v for k, v in sorted(sin_cada.items(), key=lambda kv: kv[1])})

    return {
        "pipeline": pipeline, "metricas": metricas,
        "rango": {c: [float(X_train[c].min()), float(X_train[c].max())] for c in NUMERICAS},  # rango visto en entrenamiento
        "entrada_ejemplo": X_test.iloc[[0]].to_dict("records")[0],
    }


def main() -> None:
    figuras = carpeta_figuras(CARPETA)
    df = cargar_datos()
    entender(df)
    explorar(df, figuras)
    r = entrenar(df, figuras)
    guardar_modelo(CARPETA, r["pipeline"], r["metricas"], entrada_ejemplo=r["entrada_ejemplo"], variables=VARIABLES, rango=r["rango"])
    print("\nMétricas:", {k: v for k, v in r["metricas"].items() if k not in ("comparacion_cv", "experimento_variables", "experimento_pacientes_incompletos", "experimento_sin_cada_variable", "por_clase", "oof_entrenamiento", "ic95")})
    # 6. Conclusión: ver analisis.md


if __name__ == "__main__":
    main()
