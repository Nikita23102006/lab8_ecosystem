"""Представление (View) в архитектуре MVC приложения.

Здесь только оформление: как показать результат модели человеку (текст)
или программе (JSON). Никаких вычислений и никакого разбора аргументов.
Подробнее — task10_mvc_architecture.md.
"""

import json


def format_number(value):
    """Целые — без «.0», дробные — как есть."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


# ------------------------------- результат -------------------------------

def result_text(expression, result):
    return f"{expression} = {format_number(result)}"


def result_dict(expression, result):
    return {"expression": expression, "result": result}


# -------------------------------- ошибки --------------------------------

def error_text(message):
    return f"Ошибка: {message}"


def error_dict(message):
    return {"error": str(message)}


# -------------------------------- история --------------------------------

def history_text(history):
    if not history:
        return "История пуста"
    return "\n".join(f"{number}. {result_text(expression, result)}"
                     for number, (expression, result) in enumerate(history, 1))


def history_dicts(history):
    return [result_dict(expression, result) for expression, result in history]


# ------------------------------- ответ ИИ -------------------------------

def ai_answer_text(answer):
    return (f"Вопрос:     {answer.question}\n"
            f"Выражение:  {answer.expression}\n"
            f"Результат:  {format_number(answer.result)}\n"
            f"Перевёл:    {answer.provider}")


def ai_answer_dict(answer):
    return {
        "question": answer.question,
        "expression": answer.expression,
        "result": answer.result,
        "provider": answer.provider,
    }


def to_json(data):
    """ensure_ascii=False — чтобы кириллица осталась кириллицей, а не \\u0410."""
    return json.dumps(data, ensure_ascii=False)
