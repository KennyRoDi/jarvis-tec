"""Ajustes de la ventana PyWebView que dependen de la plataforma."""
import logging
import sys

logger = logging.getLogger("jarvis")


def permitir_camara_y_microfono(ventana) -> None:
    """Concede getUserMedia (cámara y micrófono) a la interfaz.

    WebKitGTK (Linux) rechaza la solicitud con NotAllowedError si nadie atiende la señal
    'permission-request', y PyWebView no la atiende. En Windows (WebView2) aún no se ha
    verificado el comportamiento: ver frontend/SPEC.md.
    """
    if not sys.platform.startswith("linux"):
        return
    try:
        import gi

        gi.require_version("WebKit2", "4.1")
        from gi.repository import GLib, Gtk, WebKit2
    except (ImportError, ValueError):
        logger.warning("WebKit2GTK no disponible: la cámara y el micrófono quedarán bloqueados.")
        return

    def buscar_webview(widget):
        if isinstance(widget, WebKit2.WebView):
            return widget
        if isinstance(widget, Gtk.Container):
            for hijo in widget.get_children():
                encontrado = buscar_webview(hijo)
                if encontrado is not None:
                    return encontrado
        return None

    def conceder(_, solicitud) -> bool:
        if isinstance(solicitud, WebKit2.UserMediaPermissionRequest):
            solicitud.allow()
            return True
        return False  # otros permisos: comportamiento por defecto

    def conectar() -> bool:
        webview = buscar_webview(ventana.native)
        if webview is None:
            logger.warning("No se encontró el WebView de GTK: la cámara y el micrófono quedarán bloqueados.")
        else:
            webview.connect("permission-request", conceder)
        return False  # GLib.idle_add: ejecutar una sola vez

    GLib.idle_add(conectar)
