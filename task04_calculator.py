"""Задание средней сложности №4: TDD для калькулятора.

Модуль написан по циклу TDD: сначала тесты (tests/test_task04_calculator.py),
затем минимальная реализация, затем рефакторинг при зелёных тестах.

Калькулятор — модель (Model) в архитектуре MVC приложения: он ничего
не знает ни о командной строке, ни о HTTP и используется обоими интерфейсами.
"""

import ast
import math
import operator
from collections import deque

MAX_EXPRESSION_LENGTH = 200
MAX_EXPONENT = 1000          # защита от «бомб» вроде 9^9^9
INT_LIMIT = 1e15             # до этого предела целый float показываем как int


class CalculatorError(ValueError):
    """Ошибка вычисления, понятная пользователю."""


def _safe_divide(a, b):
    if b == 0:
        raise CalculatorError("Деление на ноль")
    return a / b


def _safe_mod(a, b):
    if b == 0:
        raise CalculatorError("Деление на ноль")
    return a % b


def _safe_power(a, b):
    if abs(b) > MAX_EXPONENT:
        raise CalculatorError(f"Слишком большая степень (больше {MAX_EXPONENT})")
    return float(a) ** b


def _safe_sqrt(x):
    if x < 0:
        raise CalculatorError("Корень из отрицательного числа")
    return math.sqrt(x)


# Таблицы допустимых операций. Всё, чего здесь нет, в выражении запрещено.
BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: _safe_divide,
    ast.Mod: _safe_mod,
    ast.Pow: _safe_power,
}

UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

FUNCTIONS = {
    "sqrt": _safe_sqrt,
}


class Calculator:
    """Калькулятор с безопасным разбором выражений и историей вычислений."""

    def __init__(self, precision=10, history_size=10):
        self.precision = precision
        self._history = deque(maxlen=history_size)   # старые записи уходят сами

    # ---------------------------- история ----------------------------

    @property
    def history(self):
        """Копия истории: снаружи её нельзя испортить."""
        return list(self._history)

    def clear_history(self):
        self._history.clear()

    # ---------------------------- операции ----------------------------

    def add(self, a, b):
        return self._record(f"{a} + {b}", a + b)

    def subtract(self, a, b):
        return self._record(f"{a} - {b}", a - b)

    def multiply(self, a, b):
        return self._record(f"{a} * {b}", a * b)

    def divide(self, a, b):
        return self._record(f"{a} / {b}", _safe_divide(a, b))

    def power(self, a, b):
        return self._record(f"{a} ^ {b}", self._guard(_safe_power, a, b))

    def sqrt(self, x):
        return self._record(f"sqrt({x})", _safe_sqrt(x))

    def percent(self, value, pct):
        return self._record(f"{pct}% от {value}", value * pct / 100)

    def evaluate(self, expression):
        """Вычисляет арифметическое выражение из строки.

        Строка разбирается модулем ast в дерево, и дерево обходится вручную.
        eval() не используется: через него можно выполнить любой код.
        """
        text = expression.strip()
        if not text:
            raise CalculatorError("Пустое выражение")
        if len(text) > MAX_EXPRESSION_LENGTH:
            raise CalculatorError(
                f"Слишком длинное выражение (больше {MAX_EXPRESSION_LENGTH} символов)")

        # ^ — степень, как в обычных калькуляторах; в Python это **.
        # Замена до разбора сохраняет правильный приоритет: 2+3^2 = 11.
        source = text.replace("^", "**").replace(",", ".")
        try:
            tree = ast.parse(source, mode="eval")
        except SyntaxError:
            raise CalculatorError("Некорректное выражение") from None

        return self._record(text, self._guard(self._eval_node, tree.body))

    # ---------------------------- внутреннее ----------------------------

    def _eval_node(self, node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value

        if isinstance(node, ast.BinOp) and type(node.op) in BINARY_OPERATORS:
            function = BINARY_OPERATORS[type(node.op)]
            return function(self._eval_node(node.left), self._eval_node(node.right))

        if isinstance(node, ast.UnaryOp) and type(node.op) in UNARY_OPERATORS:
            return UNARY_OPERATORS[type(node.op)](self._eval_node(node.operand))

        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in FUNCTIONS
                and len(node.args) == 1 and not node.keywords):
            return FUNCTIONS[node.func.id](self._eval_node(node.args[0]))

        raise CalculatorError(f"Недопустимый элемент выражения: {type(node).__name__}")

    @staticmethod
    def _guard(function, *args):
        """Переводит переполнение float в понятную ошибку."""
        try:
            return function(*args)
        except OverflowError:
            raise CalculatorError("Результат слишком большой") from None

    def _record(self, text, value):
        """Проверяет и приводит результат, затем записывает его в историю."""
        if isinstance(value, complex):
            raise CalculatorError("Результат не является действительным числом")
        if isinstance(value, float):
            if math.isinf(value) or math.isnan(value):
                raise CalculatorError("Результат слишком большой")
            value = round(value, self.precision)
            if value.is_integer() and abs(value) < INT_LIMIT:
                value = int(value)

        self._history.append((text, value))
        return value
