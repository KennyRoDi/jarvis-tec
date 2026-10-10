"""Constructor de los notebooks de Colab de cada modelo (uno por carpeta `modelo_XX_*/notebook.ipynb`).

Cada notebook es autocontenido: no importa nada del repositorio, de modo que se pueda subir a Google Colab con solo el
`dataset.csv` del modelo. El notebook reproduce, en una versión didáctica, lo que hace `train.py`; no lo reemplaza.

Uso (desde la raíz del repo, con un entorno que tenga nbclient e ipykernel):
    python herramientas/notebooks/modelo_08_grasa_corporal.py
Ejecuta el notebook en una carpeta temporal (con el CSV en `sample_data/`) y guarda el resultado con las salidas.
"""
import shutil
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient

RAIZ = Path(__file__).resolve().parents[2]
FEATURES = RAIZ / "backend" / "features"

CABECERA = """# <center>**Tecnológico de Costa Rica**</center>

***IC-6200 / Inteligencia artificial***

Profesor

*   **Efren Jimenez Delgado**

Proyecto JarvisTEC, I Semestre 2026"""


class Cuaderno:
    def __init__(self):
        self.celdas = [nbformat.v4.new_markdown_cell(CABECERA)]

    def md(self, texto: str) -> None:
        self.celdas.append(nbformat.v4.new_markdown_cell(texto.strip("\n")))

    def code(self, texto: str) -> None:
        self.celdas.append(nbformat.v4.new_code_cell(texto.strip("\n")))

    def portada(self, problema: str, titulo: str, subtitulo: str, temas: str, herramientas: str, hilo: list, nota: str,
                introduccion: str) -> None:
        """Las primeras celdas: análisis del problema, título con el hilo conductor y descripción."""
        self.md(f"## Análisis del Problema\n\n{problema}")
        tabla = "\n".join(f"| {e} | {p} |" for e, p in hilo)
        self.md(f"# {titulo}\n## {subtitulo}\n\n---\n\n**Temas:** {temas}  \n**Herramientas:** {herramientas}\n\n---\n\n"
                f"### El hilo conductor\n\n| Etapa | Pregunta central |\n|---|---|\n{tabla}\n\n> Nota: {nota}")
        self.md(f"{introduccion}\n\n### Autores\n\n   * Equipo JarvisTEC (proyecto de la primera etapa)")

    def datos(self, carpeta_modelo: str, descripcion: str) -> None:
        """Celda que ubica el CSV: primero `sample_data/` (panel de archivos de Colab), luego la carpeta actual y,
        si no está, pide subirlo."""
        self.md(f"Los datos están en el archivo `dataset.csv` de la carpeta `backend/features/{carpeta_modelo}/` del repositorio. {descripcion}\n\n"
                "En Colab hay dos formas de usarlo: arrastrarlo al panel de archivos (queda en `sample_data/`) o subirlo cuando la celda lo pida.")
        self.code('''# Ubicar el archivo de datos (Colab o ejecución local)
import os

RUTA = next((r for r in ("sample_data/dataset.csv", "dataset.csv") if os.path.exists(r)), None)
if RUTA is None:
    try:
        from google.colab import files
        print("Seleccione el archivo dataset.csv")
        RUTA = next(iter(files.upload()))
    except ImportError:
        raise FileNotFoundError("No se encontró dataset.csv: colóquelo en sample_data/ o en la carpeta actual")
print("Archivo de datos:", RUTA)''')

    def guardar(self, carpeta_modelo: str, ejecutar: bool = True) -> Path:
        nb = nbformat.v4.new_notebook(cells=self.celdas)
        nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}, "colab": {"provenance": []}}
        destino = FEATURES / carpeta_modelo / "notebook.ipynb"
        if ejecutar:
            with tempfile.TemporaryDirectory() as tmp:
                (Path(tmp) / "sample_data").mkdir()
                shutil.copy(FEATURES / carpeta_modelo / "dataset.csv", Path(tmp) / "sample_data" / "dataset.csv")
                NotebookClient(nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": tmp}}).execute()
        nbformat.write(nb, destino)
        return destino
