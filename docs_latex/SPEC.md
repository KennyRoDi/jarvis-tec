# SPEC — Documento científico (Overleaf)

> **Responsables:** ambos devs · **Trazabilidad:** [`specs/alcance_spec.md`](../specs/alcance_spec.md) (Documentación; rubro Documento 10 %)

La documentación se desarrolla **en Overleaf sin excepción** (enunciado). Esta carpeta es la copia en el
repositorio: se sube a Overleaf (o se sincroniza con su integración de GitHub).

## Estructura obligatoria (en este orden)

| # | Sección                                   | Contenido mínimo                                                                 |
|---|-------------------------------------------|----------------------------------------------------------------------------------|
| 1 | Portada                                   | Institución, curso, profesor, integrantes, fecha                                 |
| 2 | Solución planteada                        | Arquitectura (PyWebView + FastAPI + servicios cloud + modelos), diagrama, flujo voz → comando → modelo |
| 3 | Arquitecturas de ML                       | **Diseño de los dos agentes** (A1 emoción y A2 voz) con PEAS y tipo de agente (ver los SPEC de `asistente_voz` y `vision_facial`); pipeline común de los 10 modelos |
| 4 | Justificación y explicación de los modelos | Una subsección por modelo, desde su `analisis.md` (6 etapas de la rúbrica), justificada con **artículos científicos** |
| 5 | Análisis de resultados                    | Tabla comparativa de las métricas de los 10 modelos (de cada `metricas.json`) y discusión |
| 6 | Bibliografía                              | `referencias.bib`, al menos una referencia científica por modelo, más scikit-learn y Russell & Norvig |
| 7 | Anexos                                    | Catálogo de comandos de voz, contrato de la API (resumen) y capturas de la interfaz |

## Justificación obligatoria: Azure + Google Vision

El profesor aceptó el uso de Google Vision para la emoción **siempre que se justifique**. Debe quedar en la
sección 3 (agente de emociones) con: (1) evidencia del retiro de `emotion` en Azure Face (referencia a la
documentación de Microsoft y captura del error real), (2) la cita del enunciado que permite APIs de Google para
sentimientos en esta entrega, (3) el reparto de roles (Azure detecta, Google clasifica) y sus límites
(4 de las 8 emociones del contrato, probabilidades en 5 niveles).

## Flujo de trabajo

1. Cada modelo redacta su `analisis.md` (tono académico, tercera persona).
2. Convertirlo: `pandoc backend/features/modelo_XX_<slug>/analisis.md -o docs_latex/modelos/modelo_XX_<slug>.tex`
   y agregar `\input{modelos/modelo_XX_<slug>}` en `main.tex`.
3. Copiar las figuras a `docs_latex/figuras/` con el prefijo del modelo (`modelo_02_correlacion.png`).

## Referencias base (verificadas)

- Russell, S., & Norvig, P. (2020). *Artificial Intelligence: A Modern Approach* (4.ª ed.). Pearson.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
- Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.

## Criterios de aceptación

- [ ] Las 7 secciones completas: completo vale 10 pts, incompleto 3
- [ ] 10 subsecciones de modelo con las 6 etapas cada una
- [ ] Diseño de los agentes de emoción y de voz documentado
- [ ] Justificación de Azure + Google Vision incluida (condición del profesor)
- [ ] Compila en Overleaf sin errores
