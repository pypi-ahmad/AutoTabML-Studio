"""Tests for :mod:`app.errors` structured logging utilities."""

from __future__ import annotations

import logging

import pytest

from app.errors import AutoTabMLError, log_and_wrap, log_exception


class _DomainError(Exception):
    """Test-local domain error used to validate wrapping behavior."""


def test_log_exception_emits_structured_payload(caplog: pytest.LogCaptureFixture) -> None:
    logger = logging.getLogger("test.errors.exception")
    caplog.set_level(logging.WARNING, logger=logger.name)

    try:
        raise ValueError("boom")
    except ValueError as exc:
        log_exception(
            logger,
            exc,
            operation="probe.thing",
            context={"resource_id": "abc-123"},
        )

    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert "operation='probe.thing'" in record.getMessage()
    assert "error_type='builtins.ValueError'" in record.getMessage()
    assert "error_message='boom'" in record.getMessage()
    assert "resource_id='abc-123'" in record.getMessage()
    assert record.operation == "probe.thing"
    assert record.error_type == "builtins.ValueError"
    assert record.error_message == "boom"
    assert record.resource_id == "abc-123"
    assert record.exc_info is not None


def test_log_exception_debug_level_skips_traceback(
    caplog: pytest.LogCaptureFixture,
) -> None:
    logger = logging.getLogger("test.errors.debug")
    caplog.set_level(logging.DEBUG, logger=logger.name)

    try:
        raise RuntimeError("optional missing")
    except RuntimeError as exc:
        log_exception(logger, exc, operation="probe.optional", level=logging.DEBUG)

    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.levelno == logging.DEBUG
    assert not record.exc_info


def test_log_and_wrap_chains_cause_and_preserves_message(
    caplog: pytest.LogCaptureFixture,
) -> None:
    logger = logging.getLogger("test.errors.wrap")
    caplog.set_level(logging.WARNING, logger=logger.name)

    with pytest.raises(_DomainError, match="wrapped failure") as exc_info:
        try:
            raise OSError("disk gone")
        except OSError as exc:
            log_and_wrap(
                logger,
                exc,
                operation="storage.write",
                wrap_with=_DomainError,
                message="wrapped failure",
            )

    assert isinstance(exc_info.value.__cause__, OSError)
    assert "operation='storage.write'" in caplog.records[-1].getMessage()


def test_autotabml_error_is_exception_subclass() -> None:
    assert issubclass(AutoTabMLError, Exception)
