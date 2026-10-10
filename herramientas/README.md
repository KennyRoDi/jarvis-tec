# herramientas/ — utilidades de desarrollo (no forman parte de la API)

Se usaron para desarrollar los modelos en serie. Se ejecutan **desde la raíz del repo** salvo que se indique.

## `cerrar_modelo.py` — cierra un modelo en todos los documentos

Marca los criterios de `SPEC.md` (excepto "Aplicación"), agrega la sección "Resultado" y actualiza las tablas de estado de
`specs/modelos_spec.md`, `README.md`, `specs/alcance_spec.md` y `CLAUDE.md` (orden de desarrollo y estado).

```bash
python3 herramientas/cerrar_modelo.py . <num> <slug> <siguiente_num|-> "<resultado>" "<estado modelos_spec>" "<readme>" "<fila CLAUDE.md>"
# ejemplo: python3 herramientas/cerrar_modelo.py . 06 hepatitis 07 "texto del resultado" "✅ entrenado (falta interfaz)" "✅ F1 = 0.xx" "✅ entrenado (...); falta interfaz"
```
Revisar siempre el `git diff` después (las expresiones regulares asumen el formato actual de las tablas).

## `mutar.py` — control de mutaciones

Rompe a propósito una protección (en `train.py` o `router.py`), ejecuta las pruebas del modelo y dice si alguna falla.
Las mutaciones se listan en `mutaciones_<carpeta del modelo>.py` (ver `mutaciones_modelo_05_acv.py` como ejemplo).

```bash
cd backend && ../venv/bin/python ../herramientas/mutar.py modelo_05_acv 0 13      # mutaciones 0..12
```
Siempre restaura los archivos al terminar. Un mutante "NO DETECTADA" exige una prueba nueva o declararlo equivalente.
