# data/

Datos crudos compartidos o demasiado grandes para vivir dentro de una feature (descargas de Kaggle, ZIPs).
Los CSV de este directorio **no se suben a git** (`.gitignore`); `bash data/descargar_datasets.sh` los regenera.
Cada modelo usa su propio `backend/features/modelo_XX_<slug>/dataset.csv`, ya limpio o recortado a partir de aquí.
