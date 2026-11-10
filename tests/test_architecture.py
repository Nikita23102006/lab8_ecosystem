"""Проверка правил MVC из task10_mvc_architecture.md.

Тест читает исходный код модулей, собирает их импорты через ast
и проверяет, что слои не зависят друг от друга в запрещённую сторону.
"""

import ast
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parent.parent

CONTROLLERS = {"task08_cli", "adv01_service"}
VIEWS = {"views"}
MODELS = {"task04_calculator", "adv08_ai_client"}
UI_FRAMEWORKS = {"argparse", "fastapi", "uvicorn", "starlette", "pydantic"}

RULES = {
    # модуль: что ему импортировать нельзя
    "task04_calculator": CONTROLLERS | VIEWS | UI_FRAMEWORKS,
    "adv08_ai_client": CONTROLLERS | VIEWS | UI_FRAMEWORKS,
    "views": CONTROLLERS | MODELS | UI_FRAMEWORKS,
}


def imported_modules(module_name):
    tree = ast.parse((PROJECT / f"{module_name}.py").read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


@pytest.mark.parametrize("module, forbidden", RULES.items())
def test_layer_does_not_depend_on_forbidden_layers(module, forbidden):
    violations = imported_modules(module) & forbidden
    assert not violations, f"{module} не должен импортировать {sorted(violations)}"


@pytest.mark.parametrize("controller", sorted(CONTROLLERS))
def test_controllers_use_model_and_view(controller):
    imports = imported_modules(controller)
    assert "task04_calculator" in imports
    assert "views" in imports
