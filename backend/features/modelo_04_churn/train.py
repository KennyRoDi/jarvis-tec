"""Modelo 04 · Clasificación de abandono de clientes de telefonía (churn).

Ejecutar desde backend/:  python -m features.modelo_04_churn.train
Dataset: https://github.com/IBM/telco-customer-churn-on-icp4d (7 043 clientes; IBM)

Sigue las 6 etapas de la rúbrica; la redacción de cada etapa va en analisis.md.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix, f1_score, make_scorer,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from core.entrenamiento import carpeta_figuras, guardar_figura, guardar_modelo, umbral_optimo_f1

CARPETA = Path(__file__).parent
OBJETIVO = "churn"
POSITIVO, ORDEN = "Yes", ["No", "Yes"]  # se predice la probabilidad de abandonar ("Yes")
NUMERICAS = ["tenure", "monthly_charges"]
CATEGORICAS = ["contract", "internet_service", "payment_method", "paperless_billing", "tech_support", "online_security",
               "senior_citizen"]
VARIABLES = NUMERICAS + CATEGORICAS  # 9 variables: ver el experimento `experimento_variables` (analisis.md)
SEMILLA = 42
# Con etiquetas de texto, el scorer "average_precision" falla y scikit-learn devuelve nan en silencio:
# se indica la clase positiva y se exige que cualquier error sea explícito (error_score="raise").
PR_AUC = make_scorer(average_precision_score, response_method="predict_proba", pos_label=POSITIVO)
CV = dict(scoring="roc_auc", n_jobs=-1, error_score="raise")

# Conjuntos de variables del experimento (la AUC de validación cruzada decide cuántas hacen falta).
CONJUNTOS = {
    "18 (todas, sin total_charges)": None,  # se arma en conjuntos_de_variables()
    "9 (las elegidas)": VARIABLES,
    "6 mínimas": ["tenure", "monthly_charges", "contract", "internet_service", "payment_method", "paperless_billing"],
    "3 (antigüedad, mensualidad, contrato)": ["tenure", "monthly_charges", "contract"],
}

# 1. Análisis del problema: estimar la probabilidad de que un cliente abandone la compañía telefónica para
#    poder retenerlo. Clasificación binaria supervisada desbalanceada (≈ 26 % abandona). Ver analisis.md.


def cargar_datos() -> pd.DataFrame:
    """Lee el CSV crudo y normaliza nombres a snake_case. `total_charges` se conserva solo para el experimento."""
    df = pd.read_csv(CARPETA / "dataset.csv")
    df.columns = ["customerID"] + [
        "".join("_" + c.lower() if c.isupper() else c for c in col).lstrip("_") for col in df.columns[1:]]
    df = df.drop(columns="customerID")
    df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce").fillna(0)  # 11 vacíos = clientes con 0 meses
    df["senior_citizen"] = df["senior_citizen"].map({0: "No", 1: "Yes"})
    return df


def todas_las_variables(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in (OBJETIVO, "total_charges")]


def entender(df: pd.DataFrame) -> None:
    """2. Entendimiento de los datos."""
    print(f"Filas: {len(df)}  Columnas: {df.shape[1]}  Variables candidatas: {len(todas_las_variables(df))}")
    print("\nNulos:", int(df.isna().sum().sum()), "| filas repetidas en las variables (clientes distintos, mismo perfil):",
          int(df.drop(columns=OBJETIVO).duplicated().sum()))
    print("Abandono:", df[OBJETIVO].value_counts(normalize=True).round(3).to_dict())
    print("\nResumen numérico:\n", df[["tenure", "monthly_charges", "total_charges"]].describe().round(1).T)
    r = df.total_charges.corr(df.tenure * df.monthly_charges)
    print(f"\nCorrelación total_charges vs tenure*monthly_charges: {r:.4f} (redundante, se descarta)")


def explorar(df: pd.DataFrame, figuras: Path) -> None:
    """3. Exploración de los datos."""
    tasa = lambda col: df.groupby(col)[OBJETIVO].apply(lambda s: (s == POSITIVO).mean())  # noqa: E731
    fig, ejes = plt.subplots(1, 3, figsize=(13, 4))
    for eje, col in zip(ejes, ["contract", "internet_service", "payment_method"]):
        tasa(col).sort_values().plot.barh(ax=eje, color="#e76f51")
        eje.set_xlabel("Tasa de abandono")
        eje.set_title(col)
    guardar_figura(figuras, "abandono_por_categoria")

    fig, ejes = plt.subplots(1, 2, figsize=(10, 4))
    sns.histplot(data=df, x="tenure", hue=OBJETIVO, bins=36, stat="density", common_norm=False, ax=ejes[0])
    ejes[0].set_title("Antigüedad (meses)")
    sns.histplot(data=df, x="monthly_charges", hue=OBJETIVO, bins=36, stat="density", common_norm=False, ax=ejes[1])
    ejes[1].set_title("Mensualidad (USD)")
    guardar_figura(figuras, "antiguedad_y_mensualidad")
    print("\nTasa de abandono por contrato:", tasa("contract").round(3).to_dict())
    print("Tasa por internet:", tasa("internet_service").round(3).to_dict(), "| por pago:", tasa("payment_method").round(3).to_dict())
    print("Mediana de antigüedad:", df.groupby(OBJETIVO).tenure.median().to_dict(), "| mensualidad:", df.groupby(OBJETIVO).monthly_charges.median().round(1).to_dict())


def armar(estimador, variables: list[str] | None = None, escalar: bool = True) -> Pipeline:
    """4. Modelo: escala de las numéricas + one-hot de las categóricas + estimador, en un Pipeline."""
    variables = VARIABLES if variables is None else variables
    numericas = [v for v in variables if v in ("tenure", "monthly_charges", "total_charges")]
    categoricas = [v for v in variables if v not in numericas]
    columnas = ColumnTransformer([
        ("num", StandardScaler() if escalar else "passthrough", numericas),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categoricas),
    ])
    return Pipeline([("pre", columnas), ("modelo", estimador)])


def candidatos() -> dict[str, Pipeline]:
    """Sin pesos de clase: las probabilidades deben estar calibradas; el desbalance se trata con el umbral."""
    return {
        "tasa base (línea base)": armar(DummyClassifier(strategy="prior")),
        "regresión logística": armar(LogisticRegression(max_iter=3000)),
        "random forest": armar(RandomForestClassifier(n_estimators=150, min_samples_leaf=10, max_depth=10,
                                                      random_state=SEMILLA, n_jobs=-1), escalar=False),
        "gradient boosting": armar(HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, early_stopping=False,
                                                                  random_state=SEMILLA), escalar=False),
    }


def dividir(df: pd.DataFrame):
    """Partición 80/20 **estratificada** por abandono (conserva la proporción de clientes que se van)."""
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


def evaluar(pipeline: Pipeline, X_test, y_test, umbral: float, figuras: Path | None = None) -> dict:
    """5. Evaluación sobre el conjunto de prueba (el umbral viene del entrenamiento, no de la prueba)."""
    p_yes = pipeline.predict_proba(X_test)[:, list(pipeline.classes_).index(POSITIVO)]
    y_bin = (np.asarray(y_test) == POSITIVO).astype(int)
    metricas = {
        "roc_auc": round(float(roc_auc_score(y_bin, p_yes)), 4),
        "pr_auc": round(float(average_precision_score(y_bin, p_yes)), 4),
        "brier": round(float(brier_score_loss(y_bin, p_yes)), 4),
        "orden_clases": ORDEN,
        **estadisticas_al_umbral(y_test, p_yes, umbral),
        "en_umbral_0_5": estadisticas_al_umbral(y_test, p_yes, 0.5),
    }
    frac, media = calibration_curve(y_bin, p_yes, n_bins=6, strategy="quantile")
    metricas["calibracion"] = [{"prob_media": round(float(m), 3), "tasa_real": round(float(f), 3)} for m, f in zip(media, frac)]

    if figuras is None:  # sin efectos en disco (pruebas)
        return metricas
    fig, ejes = plt.subplots(1, 3, figsize=(14, 4))
    fpr, tpr, _ = roc_curve(y_bin, p_yes)
    ejes[0].plot(fpr, tpr, label=f"AUC = {metricas['roc_auc']}")
    ejes[0].plot([0, 1], [0, 1], "k--")
    ejes[0].set(xlabel="Falsos positivos", ylabel="Verdaderos positivos", title="Curva ROC (prueba)")
    ejes[0].legend()
    ejes[1].plot(media, frac, "o-", label="modelo")
    ejes[1].plot([0, 1], [0, 1], "k--", label="ideal")
    ejes[1].set(xlabel="Probabilidad predicha", ylabel="Tasa real de abandono", title="Calibración (prueba)")
    ejes[1].legend()
    sns.heatmap(metricas["matriz_confusion"], annot=True, fmt="d", cmap="Blues", xticklabels=ORDEN, yticklabels=ORDEN, ax=ejes[2])
    ejes[2].set(xlabel="Predicho", ylabel="Real", title=f"Matriz de confusión (umbral {umbral})")
    guardar_figura(figuras, "evaluacion")
    return metricas


def entrenar(df: pd.DataFrame, figuras: Path | None = None, imprimir: bool = True) -> dict:
    """Etapas 4 y 5 completas, **sin escribir nada en disco** si `figuras` es None: las pruebas lo reentrenan y
    exigen reproducir exactamente lo publicado (metricas.json y el artefacto)."""
    log = print if imprimir else (lambda *a, **k: None)
    train, test, y_train, y_test = dividir(df)
    X_train, X_test = train[VARIABLES], test[VARIABLES]

    # La selección usa solo el entrenamiento (validación cruzada estratificada repetida) con la AUC ROC.
    validacion = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEMILLA)
    comparacion = {}
    for nombre, pipe in candidatos().items():
        auc = cross_val_score(pipe, X_train, y_train, cv=validacion, **CV)
        pr = cross_val_score(pipe, X_train, y_train, cv=validacion, **{**CV, "scoring": PR_AUC})
        comparacion[nombre] = {"cv_roc_auc": round(float(auc.mean()), 4), "cv_roc_auc_desv": round(float(auc.std()), 4), "cv_pr_auc": round(float(pr.mean()), 4)}
        log(f"{nombre:26s} AUC ROC cv = {auc.mean():.4f} ± {auc.std():.4f}   AUC PR cv = {pr.mean():.4f}")
    ganador = max((n for n in comparacion if "línea base" not in n), key=lambda n: comparacion[n]["cv_roc_auc"])
    log(f"\nModelo elegido: {ganador}")
    pipeline = candidatos()[ganador].fit(X_train, y_train)

    # Umbral: maximiza el F1 de "Yes" con predicciones fuera de muestra del entrenamiento (no de la prueba).
    columna_yes = sorted(set(y_train)).index(POSITIVO)  # cross_val_predict ordena las clases alfabéticamente
    oof = cross_val_predict(candidatos()[ganador], X_train, y_train, method="predict_proba",
                            cv=StratifiedKFold(5, shuffle=True, random_state=SEMILLA))[:, columna_yes]
    umbral = umbral_optimo_f1((np.asarray(y_train) == POSITIVO).astype(int), oof)
    log(f"Umbral elegido (F1 de 'Yes' fuera de muestra): {umbral}")

    metricas = evaluar(pipeline, X_test, y_test, umbral, figuras)
    metricas["modelo"] = ganador
    metricas["comparacion_cv"] = comparacion
    base = candidatos()["tasa base (línea base)"].fit(X_train, y_train)
    metricas["baseline"] = {"roc_auc": round(float(roc_auc_score((np.asarray(y_test) == POSITIVO), base.predict_proba(X_test)[:, 1])), 4),
                            "accuracy": round(float((np.asarray(y_test) == "No").mean()), 4), "recall_yes": 0.0}
    metricas["n_entrenamiento"], metricas["n_prueba"] = len(X_train), len(X_test)

    # Experimento: ¿cuántas variables hacen falta? (regresión logística, AUC ROC de validación cruzada, solo entrenamiento)
    experimento = {}
    for nombre, cols in CONJUNTOS.items():
        cols = todas_las_variables(df) if cols is None else cols
        auc = cross_val_score(armar(LogisticRegression(max_iter=3000), cols), train[cols], y_train, cv=validacion, **CV)
        experimento[nombre] = {"variables": len(cols), "cv_roc_auc": round(float(auc.mean()), 4), "cv_roc_auc_desv": round(float(auc.std()), 4)}
        log(f"  {nombre:40s} AUC cv = {auc.mean():.4f} ± {auc.std():.4f}")
    metricas["experimento_variables"] = experimento

    return {
        "pipeline": pipeline, "metricas": metricas, "umbral": umbral,
        "rango": {c: [float(X_train[c].min()), float(X_train[c].max())] for c in NUMERICAS},  # rango visto en entrenamiento
        "entrada_ejemplo": X_test.iloc[[0]].to_dict("records")[0],
    }


def main() -> None:
    figuras = carpeta_figuras(CARPETA)
    df = cargar_datos()
    entender(df)
    explorar(df, figuras)
    r = entrenar(df, figuras)
    guardar_modelo(CARPETA, r["pipeline"], r["metricas"], entrada_ejemplo=r["entrada_ejemplo"],
                   variables=VARIABLES, rango=r["rango"], umbral=r["umbral"])
    print("\nMétricas:", {k: v for k, v in r["metricas"].items() if k not in ("comparacion_cv", "calibracion", "experimento_variables")})
    # 6. Conclusión: ver analisis.md


if __name__ == "__main__":
    main()
