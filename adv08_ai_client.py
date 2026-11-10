"""Задание повышенной сложности №8: интеграция внешнего ИИ API.

ИИ переводит вопрос на естественном языке в арифметическое выражение,
а считает его наш калькулятор. Ответ модели НИКОГДА не выполняется как код:
он проверяется по белому списку символов и вычисляется безопасным разбором.

Поставщик выбирается переменными окружения:

    AI_PROVIDER   offline (по умолчанию) | anthropic | openai
    AI_API_KEY    ключ доступа (для anthropic обязателен)
    AI_MODEL      название модели у выбранного поставщика
    AI_BASE_URL   адрес OpenAI-совместимого сервера
                  (по умолчанию https://api.openai.com/v1; подходит и
                  для локального Ollama: http://localhost:11434/v1)

Режим offline работает без интернета и ключей: простые правила вместо модели.
Он нужен, чтобы программа запускалась у проверяющего, и для тестов.
"""

import os
import re
from dataclasses import dataclass

import httpx

from task04_calculator import Calculator, CalculatorError

SYSTEM_PROMPT = (
    "Ты переводишь вопросы на русском языке в арифметические выражения. "
    "Отвечай ТОЛЬКО выражением в одну строку, без слов и пояснений. "
    "Разрешены числа, операции + - * / ^ %, скобки и функция sqrt(). "
    "Пример: вопрос «сколько будет двадцать процентов от 350 плюс 15» — "
    "ответ: 350 * 20 / 100 + 15"
)

MAX_QUESTION_LENGTH = 500
TIMEOUT_SECONDS = 20

# В ответе модели допустимы только цифры, знаки операций, скобки, пробелы
# и слово sqrt. Всё остальное — повод отказаться от ответа.
ALLOWED_EXPRESSION = re.compile(r"^(?:[0-9+\-*/^%().,\s]|sqrt)+$")


class AIError(RuntimeError):
    """Не удалось получить или использовать ответ ИИ."""


class AIConfigError(AIError):
    """ИИ не настроен: нет ключа, модели или неизвестный поставщик."""


@dataclass
class AIAnswer:
    question: str
    expression: str
    result: float
    provider: str


# ============================ Разбор ответа ============================

def extract_expression(text):
    """Достаёт выражение из ответа модели и проверяет его.

    Модели, несмотря на инструкцию, иногда добавляют обёртку:
    ```2+2```, `2+2`, «Выражение: 2+2», «2+2 = ?». Её срезаем.
    """
    cleaned = text.strip().replace("```", "\n").replace("`", "")
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if not lines:
        raise AIError("ИИ вернул пустой ответ")

    expression = lines[0]
    expression = re.sub(r"^[A-Za-zА-Яа-яЁё ]+:\s*", "", expression)   # «Ответ: …»
    expression = re.sub(r"\s*=\s*\??\s*$", "", expression)            # «… = ?»
    expression = expression.rstrip(".")

    if not expression or not ALLOWED_EXPRESSION.match(expression):
        raise AIError(f"Ответ ИИ не похож на арифметическое выражение: {text.strip()!r}")
    return expression


# ============================== Поставщики ==============================

def _post_json(client, url, headers, payload):
    """POST с переводом всех сетевых и HTTP-ошибок в AIError."""
    try:
        response = client.post(url, headers=headers, json=payload)
    except httpx.TimeoutException:
        raise AIError(f"ИИ не ответил за {TIMEOUT_SECONDS} с") from None
    except httpx.HTTPError as error:
        raise AIError(f"Не удалось связаться с ИИ: {error}") from None

    if response.status_code >= 400:
        raise AIError(f"ИИ вернул ошибку HTTP {response.status_code}: "
                      f"{response.text[:200]}")
    try:
        return response.json()
    except ValueError:
        raise AIError("ИИ вернул ответ не в формате JSON") from None


class AnthropicProvider:
    """Claude через Anthropic Messages API."""

    name = "anthropic"
    URL = "https://api.anthropic.com/v1/messages"
    API_VERSION = "2023-06-01"

    def __init__(self, api_key, model, client=None):
        self.api_key = api_key
        self.model = model
        self.client = client or httpx.Client(timeout=TIMEOUT_SECONDS)

    def complete(self, system, user):
        data = _post_json(
            self.client, self.URL,
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": self.API_VERSION,
                "content-type": "application/json",
            },
            payload={
                "model": self.model,
                "max_tokens": 100,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            })
        try:
            return "".join(block["text"] for block in data["content"]
                           if block.get("type") == "text")
        except (KeyError, TypeError):
            raise AIError("Неожиданный формат ответа Anthropic") from None


class OpenAICompatibleProvider:
    """Любой сервер с API в формате OpenAI Chat Completions.

    Это сама OpenAI, а также многие другие сервисы и локальный Ollama.
    """

    name = "openai"

    def __init__(self, api_key, model, base_url="https://api.openai.com/v1",
                 client=None):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.client = client or httpx.Client(timeout=TIMEOUT_SECONDS)

    def complete(self, system, user):
        headers = {"content-type": "application/json"}
        if self.api_key:   # локальным серверам ключ не нужен
            headers["authorization"] = f"Bearer {self.api_key}"

        data = _post_json(
            self.client, f"{self.base_url}/chat/completions",
            headers=headers,
            payload={
                "model": self.model,
                "temperature": 0,
                "max_tokens": 100,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            })
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise AIError("Неожиданный формат ответа OpenAI-совместимого API") from None


class OfflineProvider:
    """Замена ИИ без сети: переводит простые фразы правилами.

    Понимает цифры и слова «плюс», «минус», «умножить на», «разделить на»,
    «N процентов от M», «корень из N», «в квадрате», «в кубе», «в степени».
    """

    name = "offline"

    NUMBER = r"\d+(?:[.,]\d+)?"
    REPLACEMENTS = [
        (rf"({NUMBER})\s*(?:%|процент\w*)\s+от\s+({NUMBER})", r"(\2 * \1 / 100)"),
        (rf"(?:квадратный\s+)?корень\s+из\s+({NUMBER})", r"sqrt(\1)"),
        (r"\s+в\s+квадрате", " ^ 2"),
        (r"\s+в\s+кубе", " ^ 3"),
        (r"\s+в\s+степени\s+", " ^ "),
        (r"\s+(?:умножить|умноженное|помножить)\s+на\s+", " * "),
        (r"\s+(?:разделить|разделённое|поделить|делить)\s+на\s+", " / "),
        (r"\s+плюс\s+", " + "),
        (r"\s+минус\s+", " - "),
    ]
    FILLERS = r"(?:сколько\s+будет|чему\s+равн[оа]|посчитай|вычисли|пожалуйста)"

    def complete(self, system, user):
        text = " " + user.lower().replace("ё", "е") + " "
        text = re.sub(self.FILLERS, " ", text)
        text = text.replace("?", " ")
        for pattern, replacement in self.REPLACEMENTS:
            text = re.sub(pattern, replacement, text)
        return " ".join(text.split())


def provider_from_env(env=None):
    """Создаёт поставщика по переменным окружения."""
    env = os.environ if env is None else env
    kind = env.get("AI_PROVIDER", "offline").strip().lower()
    key = env.get("AI_API_KEY", "")
    model = env.get("AI_MODEL", "")

    if kind == "offline":
        return OfflineProvider()

    if kind == "anthropic":
        if not key:
            raise AIConfigError("Для AI_PROVIDER=anthropic задайте AI_API_KEY")
        if not model:
            raise AIConfigError("Задайте AI_MODEL — название модели Claude")
        return AnthropicProvider(key, model)

    if kind == "openai":
        if not model:
            raise AIConfigError("Задайте AI_MODEL — название модели")
        base_url = env.get("AI_BASE_URL", "https://api.openai.com/v1")
        if "api.openai.com" in base_url and not key:
            raise AIConfigError("Для OpenAI задайте AI_API_KEY")
        return OpenAICompatibleProvider(key, model, base_url)

    raise AIConfigError(
        f"Неизвестный AI_PROVIDER={kind!r}. Допустимо: offline, anthropic, openai")


# ============================== Ассистент ==============================

class AIAssistant:
    """Связывает ИИ (перевод вопроса) и калькулятор (вычисление)."""

    def __init__(self, provider, calculator=None):
        self.provider = provider
        self.calculator = calculator or Calculator()

    def ask(self, question):
        question = question.strip()
        if not question:
            raise AIError("Пустой вопрос")
        if len(question) > MAX_QUESTION_LENGTH:
            raise AIError(f"Вопрос длиннее {MAX_QUESTION_LENGTH} символов")

        raw = self.provider.complete(SYSTEM_PROMPT, question)
        expression = extract_expression(raw)
        try:
            result = self.calculator.evaluate(expression)
        except CalculatorError as error:
            raise AIError(f"ИИ предложил выражение {expression!r}, "
                          f"но его нельзя вычислить: {error}") from None

        return AIAnswer(question, expression, result, self.provider.name)


if __name__ == "__main__":
    import sys

    assistant = AIAssistant(provider_from_env())
    question = " ".join(sys.argv[1:]) or "сколько будет 20 процентов от 350 плюс 15"
    answer = assistant.ask(question)
    print(f"{answer.question}\n  -> {answer.expression} = {answer.result} "
          f"[{answer.provider}]")
