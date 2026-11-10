"""Тесты интеграции с ИИ API.

Настоящая сеть в тестах не используется: httpx.MockTransport подменяет
сервер и позволяет проверить, КАКОЙ запрос мы отправили и как разобрали ответ.
"""

import json

import httpx
import pytest

from adv08_ai_client import (
    AIAssistant, AIConfigError, AIError, AnthropicProvider, OfflineProvider,
    OpenAICompatibleProvider, extract_expression, provider_from_env,
)


def mock_client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


class StubProvider:
    """Поставщик, который всегда отвечает заданной строкой."""

    name = "stub"

    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def complete(self, system, user):
        self.calls.append((system, user))
        return self.reply


# ======================= разбор ответа модели =======================

@pytest.mark.parametrize("reply, expected", [
    ("2 + 2", "2 + 2"),
    ("```\n350 * 20 / 100 + 15\n```", "350 * 20 / 100 + 15"),
    ("`10 / 4`", "10 / 4"),
    ("Выражение: 2 ^ 10", "2 ^ 10"),
    ("Ответ: sqrt(144)", "sqrt(144)"),
    ("5 * 5 = ?", "5 * 5"),
    ("12 - 3.", "12 - 3"),
    ("  (1 + 2) * 3  \nПояснение: сначала скобки", "(1 + 2) * 3"),
])
def test_extract_expression(reply, expected):
    assert extract_expression(reply) == expected


@pytest.mark.parametrize("reply", [
    "",
    "Я не знаю",
    "__import__('os').system('rm -rf /')",
    "rm -rf /",
    "open('secret.txt').read()",
    "2 + x",
])
def test_extract_expression_rejects_non_arithmetic(reply):
    with pytest.raises(AIError):
        extract_expression(reply)


# ============================ Anthropic ============================

def test_anthropic_request_and_response():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["headers"] = request.headers
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "content": [{"type": "text", "text": "350 * 20 / 100"}],
        })

    provider = AnthropicProvider("test-key", "claude-test", client=mock_client(handler))
    assert provider.complete("системный промпт", "20% от 350") == "350 * 20 / 100"

    assert seen["url"] == "https://api.anthropic.com/v1/messages"
    assert seen["headers"]["x-api-key"] == "test-key"
    assert seen["headers"]["anthropic-version"] == "2023-06-01"
    assert seen["body"]["model"] == "claude-test"
    assert seen["body"]["system"] == "системный промпт"
    assert seen["body"]["messages"] == [{"role": "user", "content": "20% от 350"}]
    assert seen["body"]["max_tokens"] > 0


def test_anthropic_http_error_becomes_ai_error():
    client = mock_client(lambda request: httpx.Response(401, text="invalid x-api-key"))
    with pytest.raises(AIError, match="401"):
        AnthropicProvider("bad", "m", client=client).complete("s", "u")


def test_anthropic_unexpected_format():
    client = mock_client(lambda request: httpx.Response(200, json={"oops": True}))
    with pytest.raises(AIError, match="формат"):
        AnthropicProvider("k", "m", client=client).complete("s", "u")


def test_timeout_becomes_ai_error():
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(AIError, match="не ответил"):
        AnthropicProvider("k", "m", client=mock_client(handler)).complete("s", "u")


def test_not_json_response():
    client = mock_client(lambda request: httpx.Response(200, text="<html>"))
    with pytest.raises(AIError, match="JSON"):
        AnthropicProvider("k", "m", client=client).complete("s", "u")


# ======================= OpenAI-совместимый API =======================

def test_openai_compatible_request_and_response():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "choices": [{"message": {"role": "assistant", "content": "2 ^ 10"}}],
        })

    provider = OpenAICompatibleProvider("sk-test", "model-x",
                                        base_url="http://localhost:11434/v1/",
                                        client=mock_client(handler))
    assert provider.complete("sys", "два в десятой") == "2 ^ 10"

    assert seen["url"] == "http://localhost:11434/v1/chat/completions"
    assert seen["auth"] == "Bearer sk-test"
    assert seen["body"]["temperature"] == 0
    assert [m["role"] for m in seen["body"]["messages"]] == ["system", "user"]


def test_openai_compatible_without_key_sends_no_auth_header():
    seen = {}

    def handler(request):
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"choices": [{"message": {"content": "1"}}]})

    OpenAICompatibleProvider("", "llama", base_url="http://localhost:11434/v1",
                             client=mock_client(handler)).complete("s", "u")
    assert seen["auth"] is None


# ============================ Ассистент ============================

def test_assistant_full_cycle():
    stub = StubProvider("350 * 20 / 100 + 15")
    answer = AIAssistant(stub).ask("сколько будет двадцать процентов от 350 плюс 15")
    assert answer.expression == "350 * 20 / 100 + 15"
    assert answer.result == 85
    assert answer.provider == "stub"
    system, user = stub.calls[0]
    assert "ТОЛЬКО выражением" in system
    assert user == "сколько будет двадцать процентов от 350 плюс 15"


def test_assistant_never_executes_model_output(tmp_path):
    marker = tmp_path / "hacked"
    stub = StubProvider(f"__import__('pathlib').Path('{marker}').touch()")
    with pytest.raises(AIError):
        AIAssistant(stub).ask("вопрос")
    assert not marker.exists()


def test_assistant_reports_bad_expression_from_model():
    with pytest.raises(AIError, match="ноль"):
        AIAssistant(StubProvider("1 / 0")).ask("раздели на ноль")


@pytest.mark.parametrize("question", ["", "   ", "я" * 501])
def test_assistant_rejects_bad_question(question):
    with pytest.raises(AIError):
        AIAssistant(StubProvider("1")).ask(question)


# ======================= Режим offline =======================

@pytest.mark.parametrize("question, expected", [
    ("сколько будет 2 плюс 3 умножить на 4?", 14),
    ("20 процентов от 350 плюс 15", 85),
    ("Корень из 144", 12),
    ("5 в квадрате минус 1", 24),
    ("Чему равно 10 разделить на 4", 2.5),
    ("2 в степени 10", 1024),
    ("7,5% от 200", 15),
])
def test_offline_provider(question, expected):
    assert AIAssistant(OfflineProvider()).ask(question).result == expected


def test_offline_provider_does_not_understand_chat():
    with pytest.raises(AIError):
        AIAssistant(OfflineProvider()).ask("как дела?")


# ======================= Настройка через окружение =======================

def test_default_provider_is_offline():
    assert isinstance(provider_from_env({}), OfflineProvider)


def test_anthropic_from_env():
    provider = provider_from_env({"AI_PROVIDER": "anthropic",
                                  "AI_API_KEY": "k", "AI_MODEL": "m"})
    assert isinstance(provider, AnthropicProvider)
    assert provider.model == "m"


def test_openai_from_env_with_custom_url():
    provider = provider_from_env({"AI_PROVIDER": "openai", "AI_MODEL": "llama3",
                                  "AI_BASE_URL": "http://localhost:11434/v1"})
    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider.base_url == "http://localhost:11434/v1"


@pytest.mark.parametrize("env", [
    {"AI_PROVIDER": "anthropic", "AI_MODEL": "m"},          # нет ключа
    {"AI_PROVIDER": "anthropic", "AI_API_KEY": "k"},        # нет модели
    {"AI_PROVIDER": "openai", "AI_MODEL": "m"},             # OpenAI без ключа
    {"AI_PROVIDER": "skynet"},
])
def test_bad_configuration(env):
    with pytest.raises(AIConfigError):
        provider_from_env(env)
