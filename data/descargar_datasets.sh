#!/usr/bin/env bash
# Descarga los datasets de Kaggle de los 10 modelos. Requiere el token en ~/.kaggle/ (ver README.md de la raíz).
# Uso (desde la raíz del repo):  bash data/descargar_datasets.sh
# Los datasets pequeños ya están en cada carpeta de modelo; esto es para regenerarlos o para el S&P 500 completo.
set -euo pipefail
cd "$(dirname "$0")/.."
F=backend/features
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT

bajar() {  # $1 = dataset de Kaggle, $2 = archivo origen, $3 = destino
  kaggle datasets download -d "$1" -f "$2" -p "$T" --unzip -q
  mv "$T/$2" "$3"; echo "ok $3"
}
bajar team-ai/bitcoin-price-prediction "bitcoin_price_Training - Training.csv" $F/modelo_01_bitcoin/dataset.csv
bajar rajyellow46/wine-quality winequalityN.csv $F/modelo_03_vino/dataset.csv
bajar fedesoriano/stroke-prediction-dataset healthcare-dataset-stroke-data.csv $F/modelo_05_acv/dataset.csv
bajar fedesoriano/hepatitis-c-dataset HepatitisCdata.csv $F/modelo_06_hepatitis/dataset.csv
bajar fedesoriano/cirrhosis-prediction-dataset cirrhosis.csv $F/modelo_07_cirrosis/dataset.csv
bajar fedesoriano/body-fat-prediction-dataset bodyfat.csv $F/modelo_08_grasa_corporal/dataset.csv
bajar neuromusic/avocado-prices avocado.csv $F/modelo_09_aguacate/dataset.csv
# S&P 500 completo (30 MB) en data/; el recorte de dataset.csv lo genera el train.py del modelo 10
kaggle datasets download -d camnugent/sandp500 -f all_stocks_5yr.csv -p data --unzip -q && echo "ok data/all_stocks_5yr.csv"
