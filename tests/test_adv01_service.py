"""Тесты микросервиса. Зависимости подменяются через dependency_overrides:
каждый тест получает свой чистый калькулятор и предсказуемый «ИИ»."""

import pytest
from fastapi.testclient import TestClient

from adv01_service import app, get_assistant, get_calculator, get_provider
from adv08_ai_client import AIAssistant
from task04_calculator import Calculator


class StubProvider:
    name = "stub"

    def __init__(self, reply):
        self.reply = reply

    def complete(self, system, user):
        return self.reply


@pytest.fixture
def calculator():
    return Calculator()


@pytest.fixture
def client(calculator):
    app.dependency_overrides[get_calculator] = lambda: calculator
    app.dependency_overrides[get_assistant] = (
        lambda: AIAssistant(StubProvider("350 * 20 / 100 + 15"), calculator))
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": "1.0.0"}


def test_calculate(client):
    response = client.post("/calculate", json={"expression": " 2 + 3 * (4 - 1) "})
    assert response.status_code == 200
    assert response.json() == {"expression": "2 + 3 * (4 - 1)", "result": 11}


def test_calculate_float(client):
    assert client.post("/calculate", json={"expression": "10 / 4"}).json()["result"] == 2.5


@pytest.mark.parametrize("expression, message", [
    ("1 / 0", "ноль"),
    ("__import__('os')", "Недопустимый"),
    ("2 +", "Некорректное"),
])
def test_calculate_errors_are_400(client, expression, message):
    response = client.post("/calculate", json={"expression": expression})
    assert response.status_code == 400
    assert message in response.json()["detail"]


@pytest.mark.parametrize("body", [
    {},
    {"expression": ""},
    {"expression": "1+" * 150},
    {"expr": "2 + 2"},
])
def test_calculate_validation_is_422(client, body):
    assert client.post("/calculate", json=body).status_code == 422


def test_history_and_clear(client):
    client.post("/calculate", json={"expression": "1 + 1"})
    client.post("/calculate", json={"expression": "2 * 3"})
    client.post("/calculate", json={"expression": "1 / 0"})       # ошибка — не в истории
    assert client.get("/history").json() == [
        {"expression": "1 + 1", "result": 2},
        {"expression": "2 * 3", "result": 6},
    ]
    response = client.delete("/history")
    assert response.status_code == 204 and response.content == b""
    assert client.get("/history").json() == []


def test_ask(client):
    response = client.post("/ask", json={"question": "20 процентов от 350 плюс 15"})
    assert response.status_code == 200
    assert response.json() == {
        "question": "20 процентов от 350 плюс 15",
        "expression": "350 * 20 / 100 + 15",
        "result": 85,
        "provider": "stub",
    }


def test_ask_bad_model_answer_is_502(client, calculator):
    app.dependency_overrides[get_assistant] = (
        lambda: AIAssistant(StubProvider("rm -rf /"), calculator))
    response = client.post("/ask", json={"question": "что-нибудь"})
    assert response.status_code == 502
    assert "не похож" in response.json()["detail"]


def test_ask_not_configured_is_503(calculator, monkeypatch):
    """Выбран anthropic, но ключа нет — сервис честно отвечает 503."""
    monkeypatch.setenv("AI_PROVIDER", "anthropic")
    monkeypatch.delenv("AI_API_KEY", raising=False)
    get_provider.cache_clear()
    app.dependency_overrides[get_calculator] = lambda: calculator
    try:
        with TestClient(app) as test_client:
            response = test_client.post("/ask", json={"question": "2 плюс 2"})
    finally:
        app.dependency_overrides.clear()
        get_provider.cache_clear()
    assert response.status_code == 503
    assert "AI_API_KEY" in response.json()["detail"]


def test_ask_offline_by_default(calculator, monkeypatch):
    """Без переменных окружения сервис работает в режиме offline."""
    monkeypatch.delenv("AI_PROVIDER", raising=False)
    get_provider.cache_clear()
    app.dependency_overrides[get_calculator] = lambda: calculator
    try:
        with TestClient(app) as test_client:
            response = test_client.post("/ask", json={"question": "корень из 144"})
    finally:
        app.dependency_overrides.clear()
        get_provider.cache_clear()
    assert response.status_code == 200
    assert response.json()["result"] == 12
    assert response.json()["provider"] == "offline"


def test_openapi_schema_lists_all_routes(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert set(paths) == {"/health", "/calculate", "/history", "/ask"}
