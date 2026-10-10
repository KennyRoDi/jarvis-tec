"""Los notebooks de Colab (`modelo_XX_*/notebook.ipynb`) son un extra: deben ser válidos, autocontenidos y sin errores guardados."""
import ast
import re
from pathlib import Path

import nbformat
import pytest

from core.modelos import REGISTRO

FEATURES = Path(__file__).resolve().parents[1] / "features"
NOTEBOOKS = sorted(FEATURES.glob("modelo_*/notebook.ipynb"))
EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿⭐⬆✅❌️]")
SECCIONES = ("Análisis del Problema", "### Librerías", "Entendimiento de los Datos", "Exploración", "Ajuste del Modelo", "Conclusiones")


def cargar(ruta):
    return nbformat.read(ruta, as_version=4)


def codigo(nb):
    return [c.source for c in nb.cells if c.cell_type == "code"]


def test_hay_al_menos_un_notebook():
    assert NOTEBOOKS, "se esperaba al menos un notebook en backend/features/modelo_*/"


@pytest.mark.parametrize("ruta", NOTEBOOKS, ids=lambda r: r.parent.name)
def test_el_notebook_es_valido_y_tiene_las_secciones_de_la_plantilla(ruta):
    nb = cargar(ruta)
    nbformat.validate(nb)
    texto = "\n".join(c.source for c in nb.cells if c.cell_type == "markdown")
    for seccion in SECCIONES:
        assert seccion in texto, f"falta la sección '{seccion}'"
    assert "IC-6200" in texto and "Efren Jimenez Delgado" in texto


@pytest.mark.parametrize("ruta", NOTEBOOKS, ids=lambda r: r.parent.name)
def test_el_notebook_no_usa_emojis(ruta):
    assert not any(EMOJI.search(c.source) for c in cargar(ruta).cells)


@pytest.mark.parametrize("ruta", NOTEBOOKS, ids=lambda r: r.parent.name)
def test_el_notebook_es_autocontenido_y_su_codigo_compila(ruta):
    """No importa nada del repositorio (en Colab solo se sube el CSV) y cada celda es Python válido."""
    for fuente in codigo(cargar(ruta)):
        limpio = "\n".join("pass" if l.lstrip().startswith(("%", "!")) else l for l in fuente.splitlines())
        arbol = ast.parse(limpio)
        for nodo in ast.walk(arbol):
            modulos = [a.name for a in nodo.names] if isinstance(nodo, ast.Import) else [nodo.module] if isinstance(nodo, ast.ImportFrom) else []
            assert not any(m and m.split(".")[0] in ("core", "features", "main") for m in modulos), f"importa del repositorio: {modulos}"


@pytest.mark.parametrize("ruta", NOTEBOOKS, ids=lambda r: r.parent.name)
def test_el_notebook_se_guardo_ejecutado_y_sin_errores(ruta):
    celdas = [c for c in cargar(ruta).cells if c.cell_type == "code"]
    assert all(c.get("execution_count") for c in celdas), "ejecutar el generador de herramientas/notebooks/ para guardar las salidas"
    assert not [o for c in celdas for o in c.outputs if o.output_type == "error"]


@pytest.mark.parametrize("ruta", NOTEBOOKS, ids=lambda r: r.parent.name)
def test_el_notebook_busca_el_csv_en_sample_data_y_pide_subirlo(ruta):
    fuentes = "\n".join(codigo(cargar(ruta)))
    assert 'sample_data/dataset.csv' in fuentes and "files.upload()" in fuentes and "pd.read_csv(RUTA" in fuentes


def test_cada_modelo_entrenado_tiene_su_notebook():
    faltan = [m.id for m in REGISTRO.values() if m.entrenado and not (m.carpeta / "notebook.ipynb").exists()]
    assert not faltan, f"sin notebook: {faltan}"
