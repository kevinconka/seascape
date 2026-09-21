"""Options shared by the test suite."""

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--render",
        action="store_true",
        help="run the render checks: they need Cycles and take seconds each",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if config.getoption("--render"):
        return
    skip = pytest.mark.skip(reason="needs --render")
    for item in items:
        if "render" in item.keywords:
            item.add_marker(skip)
