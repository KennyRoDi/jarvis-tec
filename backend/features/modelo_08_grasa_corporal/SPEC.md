# SPEC — Modelo 08 · Predicción del porcentaje de grasa corporal

> **Responsable:** Dev A · **Contrato:** [`specs/api_rest_spec.md`](../../../specs/api_rest_spec.md) §3 ·
> **Trazabilidad:** [`specs/alcance_spec.md`](../../../specs/alcance_spec.md) (R3, A3, A4) · **Referencia completa:** [`modelo_02_autos`](../modelo_02_autos/)
>
> Lee este archivo antes de tocar la carpeta. Al terminar, marca los criterios de aceptación y actualiza el
> estado en `CLAUDE.md` y en `specs/modelos_spec.md`.

## Objetivo

- **Pregunta:** ¿Qué porcentaje de grasa corporal tiene una persona según sus medidas? (en el enunciado: "masa corporal")
- **Tipo:** Regresión
- **Variable objetivo:** `BodyFat` (%)

## Datos

- **Fuente:** https://www.kaggle.com/fedesoriano/body-fat-prediction-dataset (`bodyfat.csv`). ⚠️ En el enunciado este enlace está intercambiado con el de hepatitis
- **Archivo:** `dataset.csv` en esta carpeta. Si el original es grande, va en `data/` y aquí solo el recorte.
- ✅ Verificado: 252 filas, sin nulos, correlación `Density`–`BodyFat` = −0.988 (confirma la fuga). Variables: `Density, BodyFat, Age`, `Weight` (lb), `Height` (in) y 10 circunferencias en cm.
- ⚠️ **Excluir `Density`**: `BodyFat` se calcula a partir de ella (ecuación de Siri). Incluirla es fuga de información y daría un R² casi perfecto que no vale nada.
- Hay valores atípicos (✅ verificado: `BodyFat` = 0 %, `Height` mínima 29.5 in, `Weight` máximo 363 lb): revisarlos en la exploración.

## Enfoque sugerido

- Candidatos: Regresión lineal, Ridge/Lasso y Random Forest.
- Validación cruzada (pocos datos). Línea base: la media.

## Contrato del endpoint

- `POST /api/modelos/grasa_corporal/predecir` · `GET /api/modelos/grasa_corporal/info`
- **Entrada (`Entrada` en `router.py`):** `age`, `weight`, `height` y las circunferencias (documentar las unidades en el formulario).
- **Salida:** `prediccion` número en %.
- **¿Se ejecuta solo con la voz?** No: el comando abre el formulario.

Cuando definas la entrada, quita `extra="allow"` de `Entrada` y usa `Field`/`Literal` con rangos y valores
permitidos: de ahí sale el formulario de la interfaz (`esquema_entrada`).

## Comandos de voz (`MODELO_INFO["comandos"]`)

- "grasa corporal"
- "masa corporal"

Frases cortas y en minúscula; no deben coincidir con las de otro modelo.

## Archivos

| Archivo          | Contenido                                                                |
|------------------|--------------------------------------------------------------------------|
| `dataset.csv`    | Datos                                                                    |
| `train.py`       | 6 etapas de la rúbrica → genera `modelo.joblib`, `metricas.json`, `figuras/` |
| `router.py`      | `MODELO_INFO`, `Entrada` y `POST /predecir`                              |
| `test_modelo.py` | Pruebas del endpoint con el modelo real                                  |
| `analisis.md`    | Redacción académica de las 6 etapas (pasa a LaTeX)                       |

## Referencias sugeridas

- _Pendiente: al menos un artículo científico que justifique el algoritmo elegido._

## Criterios de aceptación

Un modelo vale 5 pts (creación) + 1 (aplicación) + 1 (API) solo si cumple **todo** lo siguiente.

**Creación del modelo**
- [ ] `dataset.csv` disponible y `python -m features.modelo_08_grasa_corporal.train` corre sin errores
- [ ] Entendimiento y exploración: estadísticas impresas y al menos 2 figuras en `figuras/`
- [ ] Modelo en un `Pipeline` (el mismo preprocesamiento en el entrenamiento y en la API)
- [ ] Evaluación en el conjunto de prueba con las métricas de `specs/modelos_spec.md` y comparación con una línea base
- [ ] `analisis.md` con las 6 secciones redactadas y al menos una referencia científica que justifique el algoritmo

**API REST**
- [ ] `Entrada` con campos tipados y validados (sin `extra="allow"`)
- [ ] `texto` de la respuesta en lenguaje natural, listo para que JARVIS lo lea
- [ ] `test_modelo.py`: predicción válida (200) y entrada inválida (422)

**Aplicación**
- [ ] Se puede ejecutar desde la interfaz (formulario o comando de voz) y el resultado se muestra y se lee en voz alta
