"""Control de mutaciones genérico: python mutar.py <carpeta_modelo> <inicio> <fin> (lee mutaciones_<modelo>.py)."""
import importlib.util, subprocess, sys
from pathlib import Path
carpeta, inicio, fin = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
spec = importlib.util.spec_from_file_location("m", Path(__file__).with_name(f"mutaciones_{carpeta}.py")); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
base = Path(f"features/{carpeta}")
originales = {f: (base / f).read_text() for f in ("train.py", "router.py")}
try:
    for archivo, etiqueta, viejo, nuevo in mod.M[inicio:fin]:
        for f, orig in originales.items():  # restaurar AMBOS archivos: una mutación anterior no debe contaminar la siguiente
            (base / f).write_text(orig)
        o = originales[archivo]
        if viejo not in o:
            print(f"  [sin aplicar] {etiqueta}", flush=True); continue
        (base / archivo).write_text(o.replace(viejo, nuevo, 1))
        r = subprocess.run(["../venv/bin/pytest", "-q", "-x", f"features/{carpeta}", "-p", "no:cacheprovider"], capture_output=True, text=True)
        linea = next((l for l in r.stdout.splitlines() if l.startswith(("FAILED", "ERROR")) or " passed" in l or " failed" in l), "?")
        print(f"  {'detectada   ' if r.returncode else 'NO DETECTADA'} {etiqueta} -> {linea.split('::')[-1][:80]}", flush=True)
finally:
    for f, o in originales.items(): (base / f).write_text(o)
