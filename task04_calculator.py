"""Задание средней сложности №4: TDD для калькулятора.

Первая версия — ровно столько кода, чтобы тесты прошли.
"""

import ast
import math


class CalculatorError(ValueError):
    pass


class Calculator:

    def __init__(self, precision=10, history_size=10):
        self.precision = precision
        self.history_size = history_size
        self._history = []

    @property
    def history(self):
        return list(self._history)

    def clear_history(self):
        self._history = []

    def _done(self, text, value):
        if isinstance(value, complex):
            raise CalculatorError("Результат не является действительным числом")
        if isinstance(value, float) and (math.isinf(value) or math.isnan(value)):
            raise CalculatorError("Результат слишком большой")
        if isinstance(value, float):
            value = round(value, self.precision)
            if value.is_integer() and abs(value) < 1e15:
                value = int(value)
        self._history.append((text, value))
        if len(self._history) > self.history_size:
            self._history.pop(0)
        return value

    def add(self, a, b):
        return self._done(f"{a} + {b}", a + b)

    def subtract(self, a, b):
        return self._done(f"{a} - {b}", a - b)

    def multiply(self, a, b):
        return self._done(f"{a} * {b}", a * b)

    def divide(self, a, b):
        if b == 0:
            raise CalculatorError("Деление на ноль")
        return self._done(f"{a} / {b}", a / b)

    def power(self, a, b):
        if abs(b) > 1000:
            raise CalculatorError("Слишком большая степень")
        try:
            value = float(a) ** b
        except OverflowError:
            raise CalculatorError("Результат слишком большой")
        return self._done(f"{a} ^ {b}", value)

    def sqrt(self, x):
        if x < 0:
            raise CalculatorError("Корень из отрицательного числа")
        return self._done(f"sqrt({x})", math.sqrt(x))

    def percent(self, value, pct):
        return self._done(f"{pct}% от {value}", value * pct / 100)

    def evaluate(self, expression):
        text = expression.strip()
        if not text:
            raise CalculatorError("Пустое выражение")
        if len(text) > 200:
            raise CalculatorError("Слишком длинное выражение")
        source = text.replace("^", "**").replace(",", ".")
        try:
            tree = ast.parse(source, mode="eval")
        except SyntaxError:
            raise CalculatorError("Некорректное выражение")
        try:
            value = self._calc(tree.body)
        except OverflowError:
            raise CalculatorError("Результат слишком большой")
        return self._done(text, value)

    def _calc(self, node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -self._calc(node.operand)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
            return self._calc(node.operand)
        if isinstance(node, ast.BinOp):
            a = self._calc(node.left)
            b = self._calc(node.right)
            if isinstance(node.op, ast.Add):
                return a + b
            if isinstance(node.op, ast.Sub):
                return a - b
            if isinstance(node.op, ast.Mult):
                return a * b
            if isinstance(node.op, ast.Div):
                if b == 0:
                    raise CalculatorError("Деление на ноль")
                return a / b
            if isinstance(node.op, ast.Mod):
                if b == 0:
                    raise CalculatorError("Деление на ноль")
                return a % b
            if isinstance(node.op, ast.Pow):
                if abs(b) > 1000:
                    raise CalculatorError("Слишком большая степень")
                return float(a) ** b
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "sqrt" and len(node.args) == 1):
            x = self._calc(node.args[0])
            if x < 0:
                raise CalculatorError("Корень из отрицательного числа")
            return math.sqrt(x)
        raise CalculatorError("Недопустимый элемент выражения")
