"""Задание средней сложности №8: простой модуль с CLI на argparse.

Примеры:
    python3 task08_cli.py add 2 3
    python3 task08_cli.py div 10 3 --precision 2
    python3 task08_cli.py eval "2 + 3 * (4 - 1)"
    python3 task08_cli.py --json eval "2 ^ 10"
    python3 task08_cli.py ask "сколько будет 20 процентов от 350 плюс 15"
    python3 task08_cli.py --help

Коды выхода: 0 — успех, 1 — ошибка вычисления, 2 — неверные аргументы.

В архитектуре MVC этот модуль — контроллер: разбирает ввод, вызывает
модель (калькулятор) и передаёт результат представлению (views.py).
"""

import argparse
import logging
import sys

import views
from adv08_ai_client import AIAssistant, AIError, provider_from_env
from task04_calculator import Calculator, CalculatorError

__version__ = "1.0.0"

log = logging.getLogger("calc")

# команда -> (метод калькулятора, имена аргументов, справка)
OPERATIONS = {
    "add": ("add", ("a", "b"), "сложение: a + b"),
    "sub": ("subtract", ("a", "b"), "вычитание: a - b"),
    "mul": ("multiply", ("a", "b"), "умножение: a * b"),
    "div": ("divide", ("a", "b"), "деление: a / b"),
    "pow": ("power", ("a", "b"), "степень: a ^ b"),
    "sqrt": ("sqrt", ("x",), "квадратный корень"),
    "percent": ("percent", ("value", "pct"), "pct процентов от value"),
}


def number(text):
    """Тип аргумента: число с точкой или запятой. 3 -> int, 2,5 -> float."""
    try:
        value = float(text.replace(",", "."))
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} — не число") from None
    return int(value) if value.is_integer() else value


def positive_int(text):
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("значение должно быть не меньше нуля")
    return value


def add_common_options(parser, in_subcommand=False):
    """Общие флаги. Их можно писать и до команды, и после неё.

    У подкоманд значение по умолчанию — SUPPRESS: если флаг после
    команды не указан, он не затрёт значение, заданное до команды.
    """
    def default(value):
        return argparse.SUPPRESS if in_subcommand else value

    parser.add_argument("--json", action="store_true", default=default(False),
                        help="вывод в формате JSON (для других программ)")
    parser.add_argument("-p", "--precision", type=positive_int, default=default(10),
                        metavar="N", help="знаков после запятой (по умолчанию 10)")
    parser.add_argument("-v", "--verbose", action="store_true", default=default(False),
                        help="подробный журнал работы")


def build_parser():
    parser = argparse.ArgumentParser(
        prog="calc",
        description="Калькулятор: операции, выражения и вопросы к ИИ.",
        epilog="Выражения берите в кавычки: символ * командная оболочка "
               "может заменить на имена файлов.",
    )
    parser.add_argument("--version", action="version",
                        version=f"%(prog)s {__version__}")
    add_common_options(parser)

    commands = parser.add_subparsers(dest="command", required=True,
                                     metavar="команда")

    def add_command(name, help_text):
        sub = commands.add_parser(name, help=help_text, description=help_text)
        add_common_options(sub, in_subcommand=True)
        return sub

    for name, (_, arg_names, help_text) in OPERATIONS.items():
        sub = add_command(name, help_text)
        for arg_name in arg_names:
            sub.add_argument(arg_name, type=number)

    evaluate = add_command("eval", "вычислить выражение")
    evaluate.add_argument("expression", nargs="+",
                          help='например "2 + 3 * (4 - 1)" или "sqrt(16) ^ 2"')

    ask = add_command("ask", "задать вопрос ИИ: он переведёт вопрос в выражение, "
                             "калькулятор его вычислит")
    ask.add_argument("question", nargs="+", help="вопрос на русском языке")

    return parser


def run(args, provider=None):
    """Выполняет команду и возвращает (код выхода, текст для вывода)."""
    calculator = Calculator(precision=args.precision)

    if args.command == "ask":
        assistant = AIAssistant(provider or provider_from_env(), calculator)
        answer = assistant.ask(" ".join(args.question))
        log.debug("ИИ (%s) перевёл вопрос в %r", answer.provider, answer.expression)
        if args.json:
            return views.to_json(views.ai_answer_dict(answer))
        return views.ai_answer_text(answer)

    if args.command == "eval":
        calculator.evaluate(" ".join(args.expression))
    else:
        method, arg_names, _ = OPERATIONS[args.command]
        values = [getattr(args, name) for name in arg_names]
        log.debug("Вызов %s%s", method, tuple(values))
        getattr(calculator, method)(*values)

    expression, result = calculator.history[-1]
    if args.json:
        return views.to_json(views.result_dict(expression, result))
    return views.result_text(expression, result)


def main(argv=None, provider=None):
    """Точка входа. argv=None — взять аргументы из командной строки."""
    parser = build_parser()
    args = parser.parse_args(argv)     # при ошибке сам печатает справку и выходит с кодом 2

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    try:
        output = run(args, provider)
    except (CalculatorError, AIError) as error:
        log.debug("Ошибка: %r", error)
        if args.json:
            print(views.to_json(views.error_dict(error)))
        else:
            print(views.error_text(error), file=sys.stderr)
        return 1

    print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
