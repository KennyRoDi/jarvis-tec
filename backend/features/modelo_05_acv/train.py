"""Modelo 05 · Clasificación del riesgo de accidente cerebrovascular (ACV).

Ejecutar desde backend/:  python -m features.modelo_05_acv.train
Dataset: https://www.kaggle.com/fedesoriano/stroke-prediction-dataset (5 110 pacientes; 4.9 % con ACV)

Sigue las 6 etapas de la rúbrica; la redacción de cada etapa va en analisis.md.
No es una herramienta de diagnóstico: estima un riesgo estadístico con fines educativos.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix, f1_score, make_scorer,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from core.entrenamiento import (carpeta_figuras, guardar_figura, guardar_modelo, intervalo_bootstrap, umbral_optimo_f1,
                                umbral_para_recall)

CARPETA = Path(__file__).parent
OBJETIVO = "acv"
POSITIVO, ORDEN = "Yes", ["No", "Yes"]  # se predice la probabilidad de tener un ACV ("Yes")
NUMERICAS = ["age", "avg_glucose_level"]
CATEGORICAS = ["hypertension", "heart_disease"]
# 4 variables clínicas. El `bmi` se excluye: su único aporte medible es un artefacto (ver `experimento_variables`).
VARIABLES = NUMERICAS + CATEGORICAS
RECALL_OBJETIVO = 0.8  # umbral de sensibilidad del tamizaje
SEMILLA = 42
REJILLA_F1 = np.round(np.arange(0.01, 0.951, 0.01), 2)  # la clase es rara: el óptimo puede estar por debajo de 0.05
# Con etiquetas de texto "average_precision" falla y scikit-learn devuelve nan en silencio: se indica la clase positiva
# y se exige que cualquier error sea explícito (error_score="raise").
PR_AUC = make_scorer(average_precision_score, response_method="predict_proba", pos_label=POSITIVO)
CV = dict(scoring="roc_auc", n_jobs=-1, error_score="raise")

TODAS = ["gender", "age", "hypertension", "heart_disease", "ever_married", "work_type", "residence_type",
         "avg_glucose_level", "bmi", "smoking_status"]
CONJUNTOS = {
    "10 (todas)": TODAS,
    "9 (sin género)": [c for c in TODAS if c != "gender"],
    "5 (las 4 elegidas + bmi imputado)": VARIABLES + ["bmi"],
    "4 (las elegidas, sin bmi)": VARIABLES,
    "1 (solo la edad)": ["age"],
    "4 + indicador de bmi faltante (artefacto, no se usa)": VARIABLES + ["bmi_faltante"],
}

# 1. Análisis del problema: estimar la probabilidad de que un paciente sufra un ACV a partir de datos clínicos básicos.
#    Clasificación binaria supervisada muy desbalanceada (≈ 5 % positivos). Ver analisis.md.


def cargar_datos() -> pd.DataFrame:
    """Lee el CSV crudo, normaliza nombres y valores. `bmi` conserva sus nulos (se imputan dentro del Pipeline)."""
    df = pd.read_csv(CARPETA / "dataset.csv").drop(columns="id")
    df = df.rename(columns={"Residence_type": "residence_type"})
    for columna in ("hypertension", "heart_disease"):
        df[columna] = df[columna].map({0: "No", 1: "Yes"})
    df[OBJETIVO] = df.pop("stroke").map({0: "No", 1: "Yes"})
    df["bmi_faltante"] = df["bmi"].isna().astype(int)  # solo para cuantificar el artefacto; no es una variable del modelo
    return df


def entender(df: pd.DataFrame) -> None:
    """2. Entendimiento de los datos."""
    print(f"Filas: {len(df)}  Columnas: {df.shape[1]}  Duplicados: {int(df.drop(columns=OBJETIVO).duplicated().sum())}")
    print("ACV:", df[OBJETIVO].value_counts().to_dict(), "| tasa:", round((df[OBJETIVO] == POSITIVO).mean(), 4))
    print("\nNulos:", df[TODAS].isna().sum()[lambda s: s > 0].to_dict())
    tasa_faltante = (df[df.bmi_faltante == 1][OBJETIVO] == POSITIVO).mean()
    tasa_presente = (df[df.bmi_faltante == 0][OBJETIVO] == POSITIVO).mean()
    print(f"Tasa de ACV con bmi faltante: {tasa_faltante:.3f} | con bmi: {tasa_presente:.3f} (el dato faltante es informativo)")
    print("\nResumen numérico:\n", df[NUMERICAS].describe().round(1).T)


def explorar(df: pd.DataFrame, figuras: Path) -> None:
    """3. Exploración de los datos."""
    positivo = df[OBJETIVO] == POSITIVO
    por_edad = df.assign(grupo=pd.cut(df.age, [0, 18, 40, 60, 70, 80, 100])).groupby("grupo", observed=True)[OBJETIVO].apply(lambda s: (s == POSITIVO).mean())
    por_edad.plot.bar(color="#e76f51")
    plt.ylabel("Tasa de ACV")
    plt.title("Tasa de ACV por grupo de edad")
    plt.xticks(rotation=30)
    guardar_figura(figuras, "tasa_por_edad")

    fig, ejes = plt.subplots(1, 3, figsize=(13, 4))
    for eje, columna in zip(ejes, NUMERICAS):
        sns.histplot(data=df, x=columna, hue=OBJETIVO, stat="density", common_norm=False, bins=30, ax=eje)
        eje.set_title(columna)
    guardar_figura(figuras, "numericas_por_clase")

    fig, ejes = plt.subplots(1, 3, figsize=(12, 3.5))
    for eje, columna in zip(ejes, ["hypertension", "heart_disease", "bmi_faltante"]):
        (positivo.groupby(df[columna]).mean()).plot.bar(ax=eje, color="#2a9d8f")
        eje.set_ylabel("Tasa de ACV")
        eje.set_title(columna)
    guardar_figura(figuras, "tasa_por_categoria")
    print("\nTasa por grupo de edad:", por_edad.round(3).to_dict())
    print("Edad mediana: sin ACV", df[~positivo].age.median(), "| con ACV", df[positivo].age.median())


def armar(estimador, variables: list[str] | None = None, escalar: bool = True) -> Pipeline:
    """4. Modelo: imputación (mediana) + escala de las numéricas, one-hot de las categóricas y estimador, en un Pipeline."""
    variables = VARIABLES if variables is None else variables
    numericas = [v for v in variables if v in ("age", "avg_glucose_level", "bmi", "bmi_faltante")]  # bmi solo en experimentos
    categoricas = [v for v in variables if v not in numericas]
    columnas = ColumnTransformer([
        ("num", Pipeline([("imputar", SimpleImputer(strategy="median"))] + ([("escala", StandardScaler())] if escalar else [])), numericas),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categoricas),
    ])
    return Pipeline([("pre", columnas), ("modelo", estimador)])


def candidatos() -> dict[str, Pipeline]:
    """Sin pesos de clase: la probabilidad mostrada debe estar calibrada; el desbalance se trata con los umbrales."""
    return {
        "tasa base (línea base)": armar(DummyClassifier(strategy="prior")),
        "regresión logística": armar(LogisticRegression(max_iter=3000)),
        "random forest": armar(RandomForestClassifier(n_estimators=200, min_samples_leaf=20, max_depth=6,
                                                      random_state=SEMILLA, n_jobs=-1), escalar=False),
        "gradient boosting": armar(HistGradientBoostingClassifier(max_iter=120, learning_rate=0.05, max_depth=3,
                                                                  early_stopping=False, random_state=SEMILLA), escalar=False),
    }


def dividir(df: pd.DataFrame):
    """Partición 80/20 **estratificada** por ACV (con solo 249 positivos, cada uno cuenta)."""
    return train_test_split(df, df[OBJETIVO], test_size=0.2, stratify=df[OBJETIVO], random_state=SEMILLA)


def estadisticas_al_umbral(y_real, p_yes, umbral: float) -> dict:
    pred = np.where(np.asarray(p_yes) >= umbral, POSITIVO, "No")
    tn, fp, fn, tp = confusion_matrix(y_real, pred, labels=ORDEN).ravel()
    return {
        "umbral": umbral,
        "accuracy": round(float((pred == np.asarray(y_real)).mean()), 4),
        "precision_yes": round(float(precision_score(y_real, pred, pos_label=POSITIVO, zero_division=0)), 4),
        "recall_yes": round(float(recall_score(y_real, pred, pos_label=POSITIVO, zero_division=0)), 4),
        "f1_yes": round(float(f1_score(y_real, pred, pos_label=POSITIVO, zero_division=0)), 4),
        "precision_macro": round(float(precision_score(y_real, pred, average="macro", zero_division=0)), 4),
        "recall_macro": round(float(recall_score(y_real, pred, average="macro", zero_division=0)), 4),
        "f1_macro": round(float(f1_score(y_real, pred, average="macro", zero_division=0)), 4),
        "matriz_confusion": [[int(tn), int(fp)], [int(fn), int(tp)]],  # filas = real (No, Yes); columnas = predicho
    }


def evaluar(pipeline: Pipeline, X_test, y_test, umbral: float, umbral_sensibilidad: float, figuras: Path | None = None) -> dict:
    """5. Evaluación sobre el conjunto de prueba (los umbrales vienen del entrenamiento, no de la prueba)."""
    p_yes = pipeline.predict_proba(X_test)[:, list(pipeline.classes_).index(POSITIVO)]
    y_bin = (np.asarray(y_test) == POSITIVO).astype(int)
    prevalencia = float(y_bin.mean())
    metricas = {
        "roc_auc": round(float(roc_auc_score(y_bin, p_yes)), 4),
        "pr_auc": round(float(average_precision_score(y_bin, p_yes)), 4),
        "pr_auc_azar": round(prevalencia, 4),
        "brier": round(float(brier_score_loss(y_bin, p_yes)), 4),
        "brier_tasa_base": round(prevalencia * (1 - prevalencia), 4),
        "orden_clases": ORDEN,
        **estadisticas_al_umbral(y_test, p_yes, umbral),
        "al_umbral_de_sensibilidad": estadisticas_al_umbral(y_test, p_yes, umbral_sensibilidad),
        "en_umbral_0_5": estadisticas_al_umbral(y_test, p_yes, 0.5),
    }
    # Con ~50 positivos en la prueba las métricas son ruidosas: intervalos de confianza del 95 % por bootstrap.
    recall_al = lambda u: (lambda y, p: float((p[y == 1] >= u).mean()))  # noqa: E731
    metricas["ic95"] = {
        "roc_auc": intervalo_bootstrap(y_bin, p_yes, roc_auc_score),
        "pr_auc": intervalo_bootstrap(y_bin, p_yes, average_precision_score),
        "recall_yes": intervalo_bootstrap(y_bin, p_yes, recall_al(umbral)),
        "recall_yes_al_umbral_de_sensibilidad": intervalo_bootstrap(y_bin, p_yes, recall_al(umbral_sensibilidad)),
    }
    frac, media = calibration_curve(y_bin, p_yes, n_bins=5, strategy="quantile")
    metricas["calibracion"] = [{"prob_media": round(float(m), 3), "tasa_real": round(float(f), 3)} for m, f in zip(media, frac)]
    if figuras is None:  # sin efectos en disco (pruebas)
        return metricas

    fig, ejes = plt.subplots(2, 2, figsize=(11, 9))
    fpr, tpr, _ = roc_curve(y_bin, p_yes)
    ejes[0, 0].plot(fpr, tpr, label=f"AUC = {metricas['roc_auc']}")
    ejes[0, 0].plot([0, 1], [0, 1], "k--")
    ejes[0, 0].set(xlabel="Falsos positivos", ylabel="Verdaderos positivos", title="Curva ROC (prueba)")
    ejes[0, 0].legend()
    precision, recall, _ = precision_recall_curve(y_bin, p_yes)
    ejes[0, 1].plot(recall, precision, label=f"AUC PR = {metricas['pr_auc']}")
    ejes[0, 1].axhline(prevalencia, color="k", linestyle="--", label=f"azar = {prevalencia:.3f}")
    ejes[0, 1].set(xlabel="Recall", ylabel="Precisión", title="Curva precisión-recall (prueba)")
    ejes[0, 1].legend()
    ejes[1, 0].plot(media, frac, "o-", label="modelo")
    ejes[1, 0].plot([0, max(media.max(), frac.max())] * 1, [0, max(media.max(), frac.max())], "k--", label="ideal")
    ejes[1, 0].set(xlabel="Probabilidad predicha", ylabel="Tasa real de ACV", title="Calibración (prueba)")
    ejes[1, 0].legend()
    sns.heatmap(metricas["matriz_confusion"], annot=True, fmt="d", cmap="Blues", xticklabels=ORDEN, yticklabels=ORDEN, ax=ejes[1, 1])
    ejes[1, 1].set(xlabel="Predicho", ylabel="Real", title=f"Matriz de confusión (umbral {umbral})")
    guardar_figura(figuras, "evaluacion")
    return metricas


def entrenar(df: pd.DataFrame, figuras: Path | None = None, imprimir: bool = True) -> dict:
    """Etapas 4 y 5 completas, **sin escribir nada en disco** si `figuras` es None: las pruebas lo reentrenan y
    exigen reproducir exactamente lo publicado (metricas.json y el artefacto)."""
    log = print if imprimir else (lambda *a, **k: None)
    train, test, y_train, y_test = dividir(df)
    X_train, X_test = train[VARIABLES], test[VARIABLES]

    # La selección usa solo el entrenamiento (validación cruzada estratificada repetida) con el AUC ROC.
    validacion = RepeatedStratifiedKFold(n_splits=5, n_repeats=4, random_state=SEMILLA)
    comparacion = {}
    for nombre, pipe in candidatos().items():
        auc = cross_val_score(pipe, X_train, y_train, cv=validacion, **CV)
        pr = cross_val_score(pipe, X_train, y_train, cv=validacion, **{**CV, "scoring": PR_AUC})
        comparacion[nombre] = {"cv_roc_auc": round(float(auc.mean()), 4), "cv_roc_auc_desv": round(float(auc.std()), 4), "cv_pr_auc": round(float(pr.mean()), 4)}
        log(f"{nombre:26s} AUC ROC cv = {auc.mean():.4f} ± {auc.std():.4f}   AUC PR cv = {pr.mean():.4f}")
    ganador = max((n for n in comparacion if "línea base" not in n), key=lambda n: comparacion[n]["cv_roc_auc"])
    log(f"\nModelo elegido: {ganador}")
    pipeline = candidatos()[ganador].fit(X_train, y_train)

    # Umbrales con predicciones fuera de muestra del entrenamiento (nunca de la prueba).
    columna_yes = sorted(set(y_train)).index(POSITIVO)  # cross_val_predict ordena las clases alfabéticamente
    oof = cross_val_predict(candidatos()[ganador], X_train, y_train, method="predict_proba",
                            cv=StratifiedKFold(5, shuffle=True, random_state=SEMILLA))[:, columna_yes]
    y_oof = (np.asarray(y_train) == POSITIVO).astype(int)
    umbral = umbral_optimo_f1(y_oof, oof, rejilla=REJILLA_F1)
    umbral_sensibilidad = umbral_para_recall(y_oof, oof, RECALL_OBJETIVO)
    log(f"Umbral F1 = {umbral} | umbral de sensibilidad (recall >= {RECALL_OBJETIVO}) = {umbral_sensibilidad}")

    metricas = evaluar(pipeline, X_test, y_test, umbral, umbral_sensibilidad, figuras)
    metricas["modelo"] = ganador
    metricas["umbral_sensibilidad"] = umbral_sensibilidad
    metricas["recall_objetivo"] = RECALL_OBJETIVO
    metricas["comparacion_cv"] = comparacion
    base = candidatos()["tasa base (línea base)"].fit(X_train, y_train)
    metricas["baseline"] = {"roc_auc": round(float(roc_auc_score((np.asarray(y_test) == POSITIVO), base.predict_proba(X_test)[:, 1])), 4),
                            "accuracy": round(float((np.asarray(y_test) == "No").mean()), 4), "recall_yes": 0.0}
    metricas["n_entrenamiento"], metricas["n_prueba"] = len(X_train), len(X_test)
    metricas["positivos_entrenamiento"], metricas["positivos_prueba"] = int((y_train == POSITIVO).sum()), int((y_test == POSITIVO).sum())

    # Experimento: ¿qué variables aportan? (regresión logística, AUC ROC de validación cruzada, solo entrenamiento)
    experimento = {}
    for nombre, cols in CONJUNTOS.items():
        auc = cross_val_score(armar(LogisticRegression(max_iter=3000), cols), train[cols], y_train, cv=validacion, **CV)
        experimento[nombre] = {"variables": len(cols), "cv_roc_auc": round(float(auc.mean()), 4), "cv_roc_auc_desv": round(float(auc.std()), 4)}
        log(f"  {nombre:52s} AUC cv = {auc.mean():.4f} ± {auc.std():.4f}")
    metricas["experimento_variables"] = experimento

    # Por qué se excluye el bmi: con la imputación por la mediana el Random Forest aísla el pico del valor imputado y
    # "descubre" que el bmi faltante predice el ACV (un artefacto de la recolección que la aplicación no puede dar).
    bosque = candidatos()["random forest"].named_steps["modelo"]
    experimento_bmi = {}
    for nombre, cols in {"random forest sin bmi (elegido)": VARIABLES, "random forest con bmi imputado por la mediana": VARIABLES + ["bmi"]}.items():
        auc = cross_val_score(armar(RandomForestClassifier(**bosque.get_params()), cols, escalar=False), train[cols], y_train, cv=validacion, **CV)
        experimento_bmi[nombre] = {"cv_roc_auc": round(float(auc.mean()), 4), "cv_roc_auc_desv": round(float(auc.std()), 4)}
        log(f"  {nombre:52s} AUC cv = {auc.mean():.4f} ± {auc.std():.4f}")
    metricas["experimento_bmi_random_forest"] = experimento_bmi

    return {
        "pipeline": pipeline, "metricas": metricas, "umbral": umbral, "umbral_sensibilidad": umbral_sensibilidad,
        "rango": {c: [float(X_train[c].min()), float(X_train[c].max())] for c in NUMERICAS},  # rango visto en entrenamiento
        "entrada_ejemplo": X_test.iloc[[0]].to_dict("records")[0],
    }


def main() -> None:
    figuras = carpeta_figuras(CARPETA)
    df = cargar_datos()
    entender(df)
    explorar(df, figuras)
    r = entrenar(df, figuras)
    guardar_modelo(CARPETA, r["pipeline"], r["metricas"], entrada_ejemplo=r["entrada_ejemplo"], variables=VARIABLES,
                   rango=r["rango"], umbral=r["umbral"], umbral_sensibilidad=r["umbral_sensibilidad"])
    print("\nMétricas:", {k: v for k, v in r["metricas"].items() if k not in ("comparacion_cv", "calibracion", "experimento_variables", "ic95")})
    # 6. Conclusión: ver analisis.md


if __name__ == "__main__":
    main()
