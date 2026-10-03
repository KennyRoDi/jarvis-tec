"""Asociación de un texto libre ("JarvisTEC precio del bitcoin para mañana") con un modelo de ML.

Cada modelo declara sus frases en MODELO_INFO["comandos"]; se elige la frase más larga contenida en el texto.
"""
import re
import unicodedata

from core.modelos import REGISTRO

PALABRAS_ACTIVACION = ("jarvistec", "jarvis tec", "jarvis")


def normalizar(texto: str) -> str:
    """Minúsculas, sin tildes ni signos y con espacios simples: "¿Cuánto  vale?" -> "cuanto vale"."""
    sin_tildes = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", sin_tildes.lower()).split())


def interpretar(texto: str) -> dict:
    limpio = normalizar(texto)
    for palabra in PALABRAS_ACTIVACION:
        if limpio.startswith(palabra + " "):
            limpio = limpio[len(palabra) + 1:]
            break

    mejor, largo = None, 0
    for modelo in REGISTRO.values():
        for comando in modelo.info.get("comandos", []):
            frase = normalizar(comando)
            if frase and frase in limpio and len(frase) > largo:
                mejor, largo = modelo, len(frase)

    if mejor is None:
        return {
            "reconocido": False,
            "modelo": None,
            "parametros": {},
            "respuesta_texto": "No entendí el comando. Puede pedirme, por ejemplo, el precio del bitcoin.",
        }
    return {
        "reconocido": True,
        "modelo": mejor.info["slug"],
        "parametros": {},  # TODO(Dev B): extraer parámetros del texto cuando el modelo lo requiera
        "respuesta_texto": f"Consultando el modelo: {mejor.info['nombre']}.",
    }
