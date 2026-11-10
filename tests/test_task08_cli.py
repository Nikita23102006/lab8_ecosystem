"""Тесты командной строки. main() принимает список аргументов,
поэтому CLI проверяется без запуска отдельного процесса."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from task08_cli import build_parser, main

PROJECT = Path(__file__).resolve().parent.parent


class StubProvider:
    name = "stub"

    def __init__(self, reply):
        self.reply = reply

    def complete(self, system, user):
        return self.reply


@pytest.mark.parametrize("argv, expected", [
    (["add", "2", "3"], "2 + 3 = 5"),
    (["sub", "10", "4"], "10 - 4 = 6"),
    (["mul", "6", "7"], "6 * 7 = 42"),
    (["div", "10", "4"], "10 / 4 = 2.5"),
    (["pow", "2", "10"], "2 ^ 10 = 1024"),
    (["sqrt", "16"], "sqrt(16) = 4"),
    (["percent", "350", "20"], "20% от 350 = 70"),
    (["add", "2,5", "0,5"], "2.5 + 0.5 = 3"),
    (["sub", "--", "-5", "3"], "-5 - 3 = -8"),
])
def test_operations(capsys, argv, expected):
    assert main(argv) == 0
    assert capsys.readouterr().out.strip() == expected


def test_eval_quoted(capsys):
    assert main(["eval", "2 + 3 * (4 - 1)"]) == 0
    assert capsys.readouterr().out.strip() == "2 + 3 * (4 - 1) = 11"


def test_eval_unquoted_parts_are_joined(capsys):
    assert main(["eval", "2", "+", "2"]) == 0
    assert capsys.readouterr().out.strip() == "2 + 2 = 4"


def test_precision(capsys):
    main(["--precision", "2", "div", "10", "3"])
    assert capsys.readouterr().out.strip() == "10 / 3 = 3.33"


@pytest.mark.parametrize("argv", [
    ["--json", "-p", "2", "div", "10", "3"],      # флаги до команды
    ["div", "10", "3", "--json", "-p", "2"],      # флаги после команды
    ["--json", "div", "10", "3", "-p", "2"],      # вперемешку
])
def test_options_before_and_after_command(capsys, argv):
    assert main(argv) == 0
    assert json.loads(capsys.readouterr().out) == {"expression": "10 / 3", "result": 3.33}


def test_negative_numbers_without_double_dash(capsys):
    assert main(["sub", "-5", "3"]) == 0
    assert capsys.readouterr().out.strip() == "-5 - 3 = -8"


def test_json_output(capsys):
    assert main(["--json", "eval", "2 ^ 10"]) == 0
    assert json.loads(capsys.readouterr().out) == {"expression": "2 ^ 10", "result": 1024}


def test_calculation_error_exit_code_and_stderr(capsys):
    assert main(["div", "1", "0"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Ошибка: Деление на ноль" in captured.err


def test_calculation_error_in_json_mode(capsys):
    assert main(["--json", "eval", "sqrt(-1)"]) == 1
    assert "error" in json.loads(capsys.readouterr().out)


@pytest.mark.parametrize("argv", [
    [],                         # нет команды
    ["unknown"],                # неизвестная команда
    ["add", "2"],               # не хватает аргумента
    ["add", "два", "3"],        # не число
    ["--precision", "-1", "add", "1", "1"],
])
def test_bad_arguments_exit_with_code_2(capsys, argv):
    with pytest.raises(SystemExit) as error:
        main(argv)
    assert error.value.code == 2
    assert "usage" in capsys.readouterr().err


def test_version(capsys):
    with pytest.raises(SystemExit) as error:
        main(["--version"])
    assert error.value.code == 0
    assert "calc 1.0.0" in capsys.readouterr().out


def test_help_lists_all_commands():
    help_text = build_parser().format_help()
    for command in ("add", "sub", "mul", "div", "pow", "sqrt", "percent", "eval", "ask"):
        assert command in help_text


def test_ask_with_stub_provider(capsys):
    assert main(["ask", "двадцать", "процентов", "от", "350"],
                provider=StubProvider("350 * 20 / 100")) == 0
    out = capsys.readouterr().out
    assert "Выражение:  350 * 20 / 100" in out
    assert "Результат:  70" in out


def test_ask_rejects_dangerous_model_answer(capsys):
    assert main(["ask", "вопрос"], provider=StubProvider("__import__('os')")) == 1
    assert "не похож на арифметическое выражение" in capsys.readouterr().err


def test_real_process_run():
    """Запуск как настоящей программы — проверяет и блок if __name__ == '__main__'."""
    done = subprocess.run([sys.executable, "task08_cli.py", "eval", "6 * 7"],
                          cwd=PROJECT, capture_output=True, text=True)
    assert done.returncode == 0
    assert done.stdout.strip() == "6 * 7 = 42"
