"""Тесты калькулятора. Написаны ДО реализации — по TDD.

Цикл TDD:
  1. красный  — пишем тест на ещё не существующее поведение, он падает;
  2. зелёный  — пишем минимальный код, чтобы тест прошёл;
  3. рефакторинг — улучшаем код, тесты при этом остаются зелёными.
"""

import pytest

from task04_calculator import Calculator, CalculatorError


@pytest.fixture
def calc():
    """Свежий калькулятор для каждого теста — тесты не влияют друг на друга."""
    return Calculator()


# ============================ Базовые операции ============================

@pytest.mark.parametrize("a, b, expected", [
    (2, 3, 5),
    (-2, 3, 1),
    (0.1, 0.2, 0.3),
    (1e10, 1, 10000000001),
])
def test_add(calc, a, b, expected):
    assert calc.add(a, b) == expected


def test_subtract(calc):
    assert calc.subtract(10, 4) == 6
    assert calc.subtract(4, 10) == -6


def test_multiply(calc):
    assert calc.multiply(6, 7) == 42
    assert calc.multiply(-2, 0.5) == -1


def test_divide(calc):
    assert calc.divide(10, 4) == 2.5


def test_divide_by_zero_raises(calc):
    with pytest.raises(CalculatorError, match="ноль"):
        calc.divide(1, 0)


def test_power(calc):
    assert calc.power(2, 10) == 1024
    assert calc.power(4, 0.5) == 2


def test_too_large_power_raises(calc):
    with pytest.raises(CalculatorError):
        calc.power(10, 100_000)


def test_sqrt(calc):
    assert calc.sqrt(16) == 4
    assert calc.sqrt(2) == pytest.approx(1.41421356)


def test_sqrt_of_negative_raises(calc):
    with pytest.raises(CalculatorError, match="отрицательн"):
        calc.sqrt(-1)


def test_percent(calc):
    assert calc.percent(350, 20) == 70


# ============================ Формат результата ============================

def test_whole_result_is_int(calc):
    result = calc.divide(10, 2)
    assert result == 5
    assert isinstance(result, int)


def test_precision():
    assert Calculator(precision=3).divide(1, 3) == 0.333


def test_float_noise_is_removed(calc):
    # Без округления 0.1 + 0.2 == 0.30000000000000004
    assert calc.add(0.1, 0.2) == 0.3


# ============================ Выражения ============================

@pytest.mark.parametrize("expression, expected", [
    ("2 + 3 * 4", 14),            # приоритет операций
    ("(2 + 3) * 4", 20),          # скобки
    ("10 / 4", 2.5),
    ("2 ^ 10", 1024),             # ^ — степень, как в калькуляторах
    ("2 ** 10", 1024),            # ** — степень, как в Python
    ("2 + 3 ^ 2", 11),            # степень приоритетнее сложения
    ("2 ^ 3 ^ 2", 512),           # степень правоассоциативна: 2^(3^2)
    ("-3 + 5", 2),                # унарный минус
    ("-(2 + 3)", -5),
    ("2,5 * 2", 5),               # десятичная запятая
    ("sqrt(16) + 1", 5),          # функция корня
    ("7 % 3", 1),                 # остаток от деления
    ("  42  ", 42),
])
def test_evaluate(calc, expression, expected):
    assert calc.evaluate(expression) == expected


@pytest.mark.parametrize("expression", [
    "",
    "   ",
    "2 +",
    "2 + * 3",
    "(2 + 3",
    "hello",
])
def test_evaluate_invalid_syntax_raises(calc, expression):
    with pytest.raises(CalculatorError):
        calc.evaluate(expression)


@pytest.mark.parametrize("expression", [
    "__import__('os').system('ls')",   # попытка выполнить код
    "open('secret.txt')",
    "x + 1",
    "abs(-5)",                         # разрешена только sqrt
    "[1, 2, 3]",
    "'строка'",
    "2 if 1 else 3",
])
def test_evaluate_rejects_anything_but_arithmetic(calc, expression):
    with pytest.raises(CalculatorError):
        calc.evaluate(expression)


def test_evaluate_division_by_zero_raises(calc):
    with pytest.raises(CalculatorError, match="ноль"):
        calc.evaluate("1 / (2 - 2)")


def test_evaluate_power_bomb_raises(calc):
    # 9^9^9 — число с сотнями миллионов цифр: без защиты процесс «зависнет».
    with pytest.raises(CalculatorError):
        calc.evaluate("9 ^ 9 ^ 9")


@pytest.mark.parametrize("expression", [
    "10 ^ 999",        # переполнение float — OverflowError в Python
    "1e308 * 10",      # переполнение без исключения — получилась бы бесконечность
    "(-8) ^ 0.5",      # в Python это комплексное число
])
def test_evaluate_result_out_of_range_raises(calc, expression):
    with pytest.raises(CalculatorError):
        calc.evaluate(expression)


def test_power_with_complex_result_raises(calc):
    with pytest.raises(CalculatorError):
        calc.power(-8, 0.5)


def test_evaluate_too_long_expression_raises(calc):
    with pytest.raises(CalculatorError, match="длинн"):
        calc.evaluate("1+" * 200 + "1")


# ============================ История ============================

def test_history_records_operations(calc):
    calc.add(2, 3)
    calc.evaluate("2 * 4")
    assert calc.history == [("2 + 3", 5), ("2 * 4", 8)]


def test_history_does_not_record_errors(calc):
    with pytest.raises(CalculatorError):
        calc.divide(1, 0)
    assert calc.history == []


def test_history_is_limited():
    calc = Calculator(history_size=3)
    for i in range(5):
        calc.add(i, 0)
    assert [result for _, result in calc.history] == [2, 3, 4]


def test_clear_history(calc):
    calc.add(1, 1)
    calc.clear_history()
    assert calc.history == []


def test_history_is_a_copy(calc):
    calc.add(1, 1)
    calc.history.clear()          # меняем полученный список...
    assert len(calc.history) == 1  # ...а история калькулятора цела
