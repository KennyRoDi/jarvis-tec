"""Fixtures compartidas por las pruebas de tests/ y de cada features/<feature>/test_*.py."""
import pytest
from fastapi.testclient import TestClient

from main import app


@pytest.fixture(scope="session")
def cliente() -> TestClient:
    return TestClient(app)


@pytest.fixture(scope="session")
def assert_error():
    """Verifica el formato de error del contrato (specs/api_rest_spec.md §1.1)."""
    def verificar(respuesta, status: int, codigo: str):
        assert respuesta.status_code == status
        error = respuesta.json()["error"]
        assert error["codigo"] == codigo
        assert error["mensaje"]
    return verificar
