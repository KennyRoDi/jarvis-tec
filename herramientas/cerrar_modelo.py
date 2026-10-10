"""Cierra un modelo: marca criterios del SPEC y actualiza las tablas de estado de los documentos.

Uso: python3 cerrar_modelo.py <repo> <num> <slug> <siguiente_num_o_-> <resultado> <estado_corto> <readme> <fila_claude>
  resultado    texto de la sección "Resultado" del SPEC.md del modelo
  estado_corto texto de la última celda de la tabla de specs/modelos_spec.md
  readme       texto de la última celda de la tabla del README
  fila_claude  texto de la celda "Estado" de la fila del modelo en la tabla de estado de CLAUDE.md
"""
import re
import sys
from pathlib import Path

repo, num, slug, siguiente, resultado, estado_corto, readme, fila_claude = sys.argv[1:9]
R = Path(repo)
carpeta = next((R / "backend/features").glob(f"modelo_{num}_*"))


def leer(p):
    return Path(p).read_text(encoding="utf-8")


# 1. SPEC.md del modelo: marcar todo menos "Aplicación" y agregar el resultado
spec = carpeta / "SPEC.md"
s = leer(spec)
cab, resto = s.split("## Criterios de aceptación")
aplicacion = "**Aplicación**"
antes, despues = resto.split(aplicacion)
antes = antes.replace("- [ ] ", "- [x] ")
s = cab + "## Criterios de aceptación" + antes + aplicacion + despues
s = s.replace("Cuando definas la entrada, quita `extra=\"allow\"` de `Entrada` y usa `Field`/`Literal` con rangos y valores\npermitidos: de ahí sale el formulario de la interfaz (`esquema_entrada`).\n", "El formulario de la interfaz sale de `esquema_entrada` (generado desde `Entrada`).\n")
s = s.replace("- **Archivo:** `dataset.csv` en esta carpeta. Si el original es grande, va en `data/` y aquí solo el recorte.\n", "- **Archivo:** `dataset.csv` en esta carpeta.\n")
if "## Referencias sugeridas" in s:
    i, j = s.index("## Referencias sugeridas"), s.index("## Criterios de aceptación")
    s = s[:i] + "## Referencias\n\nVerificadas y listadas al final de `analisis.md`; están en `docs_latex/referencias.bib`.\n\n" + s[j:]
if "## Resultado" not in s:
    s = s.replace("## Archivos", f"## Resultado (2026-10-09)\n\n{resultado}\n\n## Archivos", 1)
spec.write_text(s, encoding="utf-8")

# 2. specs/modelos_spec.md: última celda de la fila del modelo
ms = R / "specs/modelos_spec.md"
t = leer(ms)
t, n = re.subn(rf"(\| {num} \| \[[^\n]*\|)[^|\n]*\|\n", lambda m: f"{m.group(1)} {estado_corto} |\n", t, count=1)
assert n == 1, "fila de modelos_spec no encontrada"
ms.write_text(t, encoding="utf-8")

# 3. README: última celda de la fila del modelo
rd = R / "README.md"
t = leer(rd)
t, n = re.subn(rf"(\| {num} \|[^\n]*\|)[^|\n]*\|\n", lambda m: f"{m.group(1)} {readme} |\n", t, count=1)
assert n == 1, "fila del README no encontrada"
rd.write_text(t, encoding="utf-8")

# 4. specs/alcance_spec.md: contador N/10
al = R / "specs/alcance_spec.md"
t = leer(al)
t = re.sub(r"🟡 (\d+)/10", lambda m: f"🟡 {int(m.group(1)) + 1}/10", t)
al.write_text(t, encoding="utf-8")

# 5. CLAUDE.md: tabla de orden y tabla de estado
cl = R / "CLAUDE.md"
t = leer(cl)
# 5a. orden: el modelo pasa a ✅ y el siguiente a "siguiente"
lineas = t.split("\n")
for i, l in enumerate(lineas):
    if re.match(rf"\| \d+\s+\| {num} ", l):
        lineas[i] = re.sub(r"\|[^|]*\|$", "| ✅ (2026-10-09) |", l)
    if siguiente != "-" and re.match(rf"\| \d+\s+\| {siguiente} ", l):
        lineas[i] = re.sub(r"\|[^|]*\|$", "| ⏳ **siguiente** |", l)
t = "\n".join(lineas)
# 5b. estado: quitar el modelo de la fila de pendientes y agregar su fila propia tras la última fila de modelo terminado
m = re.search(r"\| (Modelos [^|]*?)\s*\|", t)
pendientes_txt = m.group(1)
ids = [x for x in re.findall(r"\d+", pendientes_txt)]
# reconstruir con los números que quedan: leer el rango tipo "01, 03–07, 09, 10"
numeros = set()
for parte in re.findall(r"(\d+)(?:–(\d+))?", pendientes_txt.replace("Modelos", "")):
    a, b = parte
    numeros.update(range(int(a), int(b or a) + 1))
numeros.discard(int(num))
partes, resto_n = [], sorted(numeros)
i = 0
while i < len(resto_n):
    j = i
    while j + 1 < len(resto_n) and resto_n[j + 1] == resto_n[j] + 1:
        j += 1
    partes.append(f"{resto_n[i]:02d}" if i == j else f"{resto_n[i]:02d}–{resto_n[j]:02d}")
    i = j + 1
nuevo_pend = "Modelos " + ", ".join(partes)
t = t.replace(f"| {pendientes_txt}", f"| {nuevo_pend}".ljust(len(pendientes_txt) + 2), 1) if numeros else re.sub(r"\| Modelos [^\n]*\n", "", t, count=1)
# insertar fila del modelo terminado justo antes de la fila de pendientes
nombre = carpeta.name.split("_", 2)[2].replace("_", " ")
fila = f"| Modelo {num} {nombre}".ljust(31) + f"| {fila_claude} |\n"
t = re.sub(r"(\| Modelos [^\n]*\n)", lambda mm: fila + mm.group(1), t, count=1) if numeros else t
cl.write_text(t, encoding="utf-8")
print("cerrado", carpeta.name)
