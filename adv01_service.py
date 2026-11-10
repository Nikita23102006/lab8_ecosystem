"""Задание повышенной сложности №1: мини-микросервис на FastAPI.

Запуск:   python3 -m uvicorn adv01_service:app --reload
Swagger:  http://127.0.0.1:8000/docs

Микросервис — небольшое самостоятельное приложение с одной задачей
и сетевым API. Этот считает выражения, хранит историю и отвечает
на вопросы через ИИ. Внутри — та же модель (task04_calculator),
что и у командной строки; меняется только контроллер.
"""

from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException, Response, status
from pydantic import BaseModel, Field

import views
from adv08_ai_client import AIAssistant, AIConfigError, AIError, provider_from_env
from task04_calculator import MAX_EXPRESSION_LENGTH, Calculator, CalculatorError

__version__ = "1.0.0"

app = FastAPI(
    title="Калькулятор — микросервис",
    description="Вычисление выражений, история и вопросы на естественном языке.",
    version=__version__,
)


# ================================ Схемы ================================

class ExpressionIn(BaseModel):
    expression: str = Field(min_length=1, max_length=MAX_EXPRESSION_LENGTH,
                            examples=["2 + 3 * (4 - 1)"])


class ResultOut(BaseModel):
    expression: str
    result: int | float


class QuestionIn(BaseModel):
    question: str = Field(min_length=1, max_length=500,
                          examples=["сколько будет 20 процентов от 350 плюс 15"])


class AnswerOut(BaseModel):
    question: str
    expression: str
    result: int | float
    provider: str


# ============================ Зависимости ============================
# FastAPI сам вызывает эти функции и передаёт результат в обработчики.
# В тестах их подменяют через app.dependency_overrides — без правки кода.

_calculator = Calculator(history_size=50)


def get_calculator() -> Calculator:
    return _calculator


@lru_cache
def get_provider():
    """Поставщик ИИ создаётся один раз — при первом вопросе."""
    return provider_from_env()


def get_assistant(calculator: Calculator = Depends(get_calculator)) -> AIAssistant:
    try:
        provider = get_provider()
    except AIConfigError as error:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(error)) from None
    return AIAssistant(provider, calculator)


# ============================== Маршруты ==============================

@app.get("/health")
def health():
    """Проверка живости — её опрашивают балансировщики и системы мониторинга."""
    return {"status": "ok", "version": __version__}


@app.post("/calculate", response_model=ResultOut)
def calculate(data: ExpressionIn, calculator: Calculator = Depends(get_calculator)):
    try:
        result = calculator.evaluate(data.expression)
    except CalculatorError as error:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(error)) from None
    return views.result_dict(data.expression.strip(), result)


@app.get("/history", response_model=list[ResultOut])
def history(calculator: Calculator = Depends(get_calculator)):
    return views.history_dicts(calculator.history)


@app.delete("/history", status_code=status.HTTP_204_NO_CONTENT)
def clear_history(calculator: Calculator = Depends(get_calculator)):
    calculator.clear_history()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.post("/ask", response_model=AnswerOut)
def ask(data: QuestionIn, assistant: AIAssistant = Depends(get_assistant)):
    """502 Bad Gateway — ошибка не у нас, а у внешнего сервиса ИИ."""
    try:
        answer = assistant.ask(data.question)
    except AIError as error:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(error)) from None
    return views.ai_answer_dict(answer)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("adv01_service:app", host="127.0.0.1", port=8000, reload=True)
