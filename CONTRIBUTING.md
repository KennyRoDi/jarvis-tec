# Guía de Contribución (CONTRIBUTING.md)

Este repositorio sigue metodologías de desarrollo estructuradas: **Spec-Driven Development (SDD)**, **Feature-Driven Development (FDD)** y **Agent-Driven Development (ADD)**. 

Si eres un **Agente de IA** (como Claude) o un desarrollador humano, debes leer y cumplir estrictamente estas directrices antes de escribir, modificar o proponer código.

---

## 1. Directrices Fundamentales para Agentes de IA (ADD)
Si eres una Inteligencia Artificial operando en este repositorio, asume las siguientes restricciones de sistema:
* **Lectura de Contexto Obligatoria:** Antes de iniciar cualquier tarea, lee el archivo `CLAUDE.md` en la raíz del repositorio. Contiene el estado actual, comandos de ejecución y decisiones arquitectónicas.
* **Prohibición de Alucinación de Endpoints:** Nunca inventes rutas o cuerpos de respuesta JSON. Consulta siempre la carpeta `specs/` (ej. `specs/api_rest_spec.md`) para conocer los contratos exactos.
* **Aislamiento de Impacto:** Limita tus modificaciones a la carpeta de la característica (*feature*) asignada. Tienes prohibido modificar el enrutador principal (`backend/main.py`) para agregar dependencias; confía en el registro automático de rutas.
* **Validación Real:** No generes pruebas unitarias ficticias (*fakes* o *mocks* vacíos). Las pruebas deben ejecutar los modelos de Machine Learning reales y validar las respuestas HTTP usando el cliente de pruebas de FastAPI.

---

## 2. Flujo de Trabajo (FDD + SDD)

Para evitar cuellos de botella entre desarrolladores, trabajamos en rebanadas verticales y por contratos:

### Paso 1: Especificación (SDD)
Cualquier nueva característica (ej. un nuevo modelo de ML) comienza en la carpeta `specs/`.
1. Actualiza `specs/api_rest_spec.md` con las rutas, parámetros esperados y esquema JSON.
2. Ambos desarrolladores (Frontend y Backend) usarán este archivo como única fuente de verdad.

### Paso 2: Desarrollo por Características (FDD)
Todo el código del backend debe residir en su respectiva rebanada vertical dentro de `backend/features/`.
1. Lee el `SPEC.md` de la carpeta asignada (las 10 carpetas `backend/features/modelo_XX_<slug>/` ya existen).
2. Todo vive dentro de esa carpeta: `dataset.csv` (si es pequeño; si no, en `data/`), `train.py`, `router.py`,
   `test_modelo.py` y `analisis.md`. Referencia completa: `modelo_02_autos`.
3. Al terminar, marca los criterios de aceptación del `SPEC.md` y actualiza el estado en `CLAUDE.md`.
4. El frontend consumirá la API de forma independiente.

---

## 3. Estándares de Ramas y Commits

Seguimos el estándar de **Conventional Commits** y ramificación por características.

### Ramas
* Nuevas características: `feature/nombre-del-modelo` (ej. `feature/prediccion-bitcoin`)
* Corrección de errores: `fix/nombre-del-error` (ej. `fix/error-camara-azure`)
* Documentación: `docs/actualizacion-specs`

### Commits
El formato obligatorio para los mensajes de commit es:
`<tipo>(<alcance>): <descripción corta>`

**Tipos permitidos:**
* `feat`: Una nueva característica (nuevo modelo de ML, nueva vista web).
* `fix`: Corrección de un error.
* `docs`: Cambios en la documentación o en la carpeta `specs/`.
* `test`: Adición o corrección de pruebas unitarias.
* `refactor`: Cambio en el código que no corrige un error ni añade una característica.

*Ejemplo:* `feat(ml): agregar modelo de prediccion de precios de casas`

---

## 4. Ecosistema Tecnológico Restringido
No introduzcas tecnologías fuera de este stack sin aprobación:
* **Escritorio:** PyWebView
* **Backend API:** FastAPI + Uvicorn + Python 3.x
* **Modelos ML:** Scikit-Learn, Pandas (10 modelos aislados)
* **Servicios Cognitivos:** Azure Face API (Visión), Google Cloud Speech-to-Text (Voz)
* **Frontend:** Tecnologías Web (HTML/CSS/JS o framework acordado en `CLAUDE.md`)

---

## 5. Proceso de Pull Requests (PR)
Para fusionar código a la rama `main`:
1. El código debe incluir pruebas locales exitosas ejecutando `pytest`.
2. La funcionalidad debe estar completamente autocontenida en su carpeta de `features/`.
3. El título del PR debe seguir el formato de Conventional Commits.
4. En la descripción del PR, cita qué endpoint de `specs/api_rest_spec.md` se está cumpliendo.