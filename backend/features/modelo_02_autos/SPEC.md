# SPEC — Modelo 02 · Predicción del precio de un automóvil usado

> **Responsable:** Dev A · **Contrato:** [`specs/api_rest_spec.md`](../../../specs/api_rest_spec.md) §3 ·
> **Trazabilidad:** [`specs/alcance_spec.md`](../../../specs/alcance_spec.md) (R3, A3, A4) · **Referencia completa:** [`modelo_02_autos`](../modelo_02_autos/)
>
> Lee este archivo antes de tocar la carpeta. Al terminar, marca los criterios de aceptación y actualiza el
> estado en `CLAUDE.md` y en `specs/modelos_spec.md`.

## Objetivo

- **Pregunta:** ¿Cuál es el precio de reventa de un automóvil usado?
- **Tipo:** Regresión
- **Variable objetivo:** `Selling_Price` (lakhs INR; 1 lakh = 100 000 rupias)

## Datos

- **Fuente:** https://raw.githubusercontent.com/amankharwal/Website-data/master/car%20data.csv ✅ ya descargado
- **Archivo:** `dataset.csv` en esta carpeta. Si el original es grande, va en `data/` y aquí solo el recorte.
- 301 filas, 9 columnas, sin nulos.
- Se descarta `Car_Name` (≈100 valores distintos: no generaliza).
- Las columnas se pasan a minúsculas en `train.py`: así coinciden con los campos de `Entrada`.

## Enfoque sugerido

- ✅ Implementado: `Pipeline` (StandardScaler + OneHotEncoder) + Random Forest (200 árboles).
- ✅ Línea base: regresión lineal. Validación cruzada de 5 particiones.
- Pendiente: justificar la elección con referencias bibliográficas.

## Contrato del endpoint

- `POST /api/modelos/autos/predecir` · `GET /api/modelos/autos/info`
- **Entrada (`Entrada` en `router.py`):** `year, present_price, kms_driven, fuel_type, seller_type, transmission, owner` (ver `router.py`).
- **Salida:** `prediccion` número en lakhs INR.
- **¿Se ejecuta solo con la voz?** No: requiere 7 datos. El comando abre el formulario prellenado con `entrada_ejemplo`.

Cuando definas la entrada, quita `extra="allow"` de `Entrada` y usa `Field`/`Literal` con rangos y valores
permitidos: de ahí sale el formulario de la interfaz (`esquema_entrada`).

## Comandos de voz (`MODELO_INFO["comandos"]`)

- "precio de un auto"
- "precio de un carro"
- "cuánto vale mi carro"

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

- Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.

## Criterios de aceptación

Un modelo vale 5 pts (creación) + 1 (aplicación) + 1 (API) solo si cumple **todo** lo siguiente.

**Creación del modelo**
- [x] `dataset.csv` disponible y `python -m features.modelo_02_autos.train` corre sin errores
- [x] Entendimiento y exploración: estadísticas impresas y al menos 2 figuras en `figuras/`
- [x] Modelo en un `Pipeline` (el mismo preprocesamiento en el entrenamiento y en la API)
- [x] Evaluación en el conjunto de prueba con las métricas de `specs/modelos_spec.md` y comparación con una línea base
- [ ] `analisis.md` con las 6 secciones redactadas y al menos una referencia científica que justifique el algoritmo

**API REST**
- [x] `Entrada` con campos tipados y validados (sin `extra="allow"`)
- [x] `texto` de la respuesta en lenguaje natural, listo para que JARVIS lo lea
- [x] `test_modelo.py`: predicción válida (200) y entrada inválida (422)

**Aplicación**
- [ ] Se puede ejecutar desde la interfaz (formulario o comando de voz) y el resultado se muestra y se lee en voz alta
