from __future__ import annotations

import logging


def test_configure_logging_suppresses_non_actionable_dependency_noise():
    from app.logging_config import configure_logging

    root = logging.getLogger()
    ge_logger = logging.getLogger("great_expectations._docs_decorators")
    ge_registry_logger = logging.getLogger("great_expectations.expectations.registry")
    matplotlib_logger = logging.getLogger("matplotlib.style.core")
    visions_logger = logging.getLogger("visions.backends")

    original_root_handlers = list(root.handlers)
    original_root_level = root.level
    original_ge_level = ge_logger.level
    original_ge_registry_level = ge_registry_logger.level
    original_matplotlib_level = matplotlib_logger.level
    original_visions_level = visions_logger.level

    try:
        root.handlers = []
        ge_logger.setLevel(logging.NOTSET)
        ge_registry_logger.setLevel(logging.NOTSET)
        matplotlib_logger.setLevel(logging.NOTSET)
        visions_logger.setLevel(logging.NOTSET)

        configure_logging()

        assert ge_logger.level == logging.WARNING
        assert ge_registry_logger.level == logging.WARNING
        assert matplotlib_logger.level == logging.ERROR
        assert visions_logger.level == logging.WARNING
    finally:
        for handler in root.handlers:
            if handler not in original_root_handlers:
                handler.close()
        root.handlers = original_root_handlers
        root.setLevel(original_root_level)
        ge_logger.setLevel(original_ge_level)
        ge_registry_logger.setLevel(original_ge_registry_level)
        matplotlib_logger.setLevel(original_matplotlib_level)
        visions_logger.setLevel(original_visions_level)