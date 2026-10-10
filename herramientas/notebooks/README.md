# Notebooks de Colab

Cada modelo tiene un `notebook.ipynb` en su carpeta (`backend/features/modelo_XX_<slug>/`). Es un extra para quien quiera probar el modelo en
Google Colab: reproduce de forma didáctica lo que hace `train.py`, no lo reemplaza. La estructura sigue `Ejemplo_Plantilla.ipynb`.

## Cómo se usa en Colab
1. Abrir Colab y subir `notebook.ipynb` (Archivo, Subir notebook).
2. Subir `dataset.csv` de la carpeta del modelo al panel de archivos (queda en `sample_data/`), o esperar a que la celda de datos lo pida.
3. Ejecutar todo (Entorno de ejecución, Ejecutar todas). Solo usa pandas, scikit-learn, seaborn y matplotlib, que Colab ya trae.

## Cómo se regenera
```bash
pip install nbclient ipykernel      # solo para regenerar
python herramientas/notebooks/modelo_08_grasa_corporal.py
```
`nb.py` copia el `dataset.csv` a una carpeta temporal (`sample_data/`), ejecuta el notebook y guarda el resultado con las salidas.

## Reglas
- Autocontenido: no importa nada del repositorio (`core`, `features`). `backend/tests/test_notebooks.py` lo comprueba.
- Español, sin emojis, y el estilo de la plantilla: análisis del problema, hilo conductor, librerías, secciones numeradas, conclusiones.
- Las cifras deben coincidir con `metricas.json` (se comprobó ejecutando cada notebook con scikit-learn 1.6, el de Colab, y con la versión del repositorio).
- Afirmar en el texto solo lo que muestran las salidas del propio notebook.
