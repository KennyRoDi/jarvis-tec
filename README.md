# JarvisTEC

Asistente personal de escritorio estilo "Jarvis" que reconoce la emoción del usuario por cámara, recibe
comandos por voz y responde consultando 10 modelos de aprendizaje automático expuestos por un API REST.

Proyecto del curso **Inteligencia Artificial** — I Semestre 2026, Ingeniería en Computación,
Instituto Tecnológico de Costa Rica, Campus Tecnológico Local San Carlos.

## Arquitectura

```
┌──────────────────────────── App de escritorio (PyWebView) ────────────────────────────┐
│  Interfaz web (React)  ──fetch /api/*──►  FastAPI (hilo en segundo plano, :8000)       │
│                                            ├─ features/asistente_voz  → Google Speech  │
│                                            ├─ features/vision_facial  → Azure + Vision │
│                                            └─ features/modelo_01 … modelo_10 (sklearn) │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

| Capa              | Tecnología                                                   |
|-------------------|--------------------------------------------------------------|
| Escritorio        | PyWebView                                                    |
| API REST          | FastAPI + Uvicorn                                            |
| Machine Learning  | scikit-learn, pandas, NumPy                                  |
| Voz a texto       | Google Cloud Speech-to-Text                                  |
| Emociones         | Azure Face (detección) + Google Cloud Vision (emoción)       |
| Interfaz          | React (Vite)                                                 |
| Documentación     | LaTeX (Overleaf)                                             |

El desarrollo sigue tres metodologías:

- **SDD (Spec-Driven):** [`specs/api_rest_spec.md`](specs/api_rest_spec.md) es el contrato único entre backend y frontend.
- **FDD (Feature-Driven):** cada modelo o servicio es una carpeta aislada en `backend/features/` que se
  registra sola en la API, sin editar `main.py`.
- **ADD (Agent-Driven):** [`CLAUDE.md`](CLAUDE.md) guarda reglas, comandos y estado del proyecto para los agentes de IA.

## Estructura

```
├── specs/              Contrato de la API, lista de modelos y trazabilidad del enunciado
├── backend/
│   ├── main.py         FastAPI + ventana PyWebView
│   ├── core/           Código compartido (errores, registro y carga de modelos, métricas)
│   ├── features/       Una carpeta por modelo o servicio (dataset, train.py, router.py, analisis.md)
│   ├── ui_prueba/      Página temporal mientras no existe el build de React
│   └── tests/          Pruebas del contrato (pytest)
├── frontend/           Interfaz React (cliente del API + mocks del contrato)
├── data/               Datos crudos grandes o compartidos
└── docs_latex/         Documento para Overleaf
```

## Instalación

Requisitos: **Python 3.12+** (NumPy 2.5 no corre en 3.10 ni 3.11) y Node.js 20.19+.

```bash
# Backend
python -m venv venv                      # Linux: agregar --system-site-packages (GTK/WebKit2 para PyWebView)
source venv/bin/activate                 # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp backend/.env.example backend/.env     # completar credenciales de Azure y Google Cloud

# Frontend
cd frontend && npm install
```

## Uso

| Acción                     | Comando                                                       |
|----------------------------|---------------------------------------------------------------|
| Abrir la app de escritorio | `python backend/main.py`                                      |
| Solo la API (desarrollo)   | `cd backend && uvicorn main:app --reload` → http://localhost:8000/docs |
| Frontend en desarrollo     | `cd frontend && npm run dev`                                  |
| Frontend sin backend       | `cd frontend && VITE_USAR_MOCKS=true npm run dev`             |
| Build para escritorio      | `cd frontend && npm run build`                                |
| Entrenar un modelo         | `cd backend && python -m features.modelo_02_autos.train`      |
| Pruebas                    | `cd backend && pytest`                                        |
| Regenerar datasets (Kaggle) | `pip install kaggle && bash data/descargar_datasets.sh`      |

## Modelos de Machine Learning

| #  | Modelo                                   | Tipo           | Estado        |
|----|------------------------------------------|----------------|---------------|
| 01 | Precio del Bitcoin                       | Regresión      | Pendiente     |
| 02 | Precio de un automóvil                   | Regresión      | ✅ R² = 0.962 |
| 03 | Calidad del vino                         | Clasificación  | ✅ F1 = 0.59 |
| 04 | Abandono de clientes de telefonía        | Clasificación  | ✅ AUC = 0.84 |
| 05 | Riesgo de accidente cerebrovascular      | Clasificación  | ✅ AUC = 0.84 |
| 06 | Tipo de hepatitis C                      | Clasificación  | ✅ F1 macro = 0.58 |
| 07 | Etapa de cirrosis                        | Clasificación  | ✅ F1 macro = 0.47 |
| 08 | Porcentaje de grasa corporal             | Regresión      | ✅ R² = 0.56 (CV 0.70) |
| 09 | Precio del aguacate                      | Regresión      | ✅ R² = 0.42 (temporal) |
| 10 | Precio de acciones del S&P 500           | Regresión      | Pendiente     |

Fuentes de datos, variables objetivo y comandos de voz de cada modelo: [`specs/modelos_spec.md`](specs/modelos_spec.md).

## Contribuir

Cada carpeta de trabajo tiene un `SPEC.md` con su objetivo, tareas y criterios de aceptación;
[`specs/alcance_spec.md`](specs/alcance_spec.md) relaciona cada requisito del enunciado con su carpeta.

Ver [`CONTRIBUTING.md`](CONTRIBUTING.md): flujo spec → feature, ramas, Conventional Commits y pull requests.
