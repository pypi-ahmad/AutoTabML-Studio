"""Tests for security / masking helpers."""

from __future__ import annotations

import logging

from app.logging_config import _RedactingFormatter
from app.security.masking import mask_secret, redact_key_in_text, safe_error_message


class TestMaskSecret:
    def test_none_returns_not_set(self):
        assert mask_secret(None) == "<not set>"

    def test_empty_returns_not_set(self):
        assert mask_secret("") == "<not set>"

    def test_short_string_fully_masked(self):
        assert mask_secret("ab") == "****"

    def test_normal_key_masked_middle(self):
        result = mask_secret("sk-abcdef123456xyz")
        assert result.startswith("sk-a")
        assert result.endswith("6xyz")
        assert "*" in result

    def test_reveal_chars_at_edges(self):
        key = "0123456789abcdef"
        result = mask_secret(key, reveal=4)
        assert result[:4] == "0123"
        assert result[-4:] == "cdef"
        assert "****" in result


class TestRedactKeyInText:
    def test_redacts_openai_style_key(self):
        text = "Using key sk-abcdef123456789xyz for requests"
        result = redact_key_in_text(text)
        assert "sk-abcdef123456789xyz" not in result
        assert "sk-a" in result

    def test_redacts_gemini_style_key(self):
        text = "Using key AIabcdefghijklmnop123456 for requests"
        result = redact_key_in_text(text)
        assert "AIabcdefghijklmnop123456" not in result
        assert "AIab" in result

    def test_leaves_non_key_text_alone(self):
        text = "Hello world, no secrets here"
        assert redact_key_in_text(text) == text

    def test_redacts_bearer_token(self):
        text = "Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGciOi.long_token"
        result = redact_key_in_text(text)
        assert "eyJ0eXAi" not in result
        assert "Bearer ****" in result

    def test_redacts_password_in_connection_uri(self):
        text = "connecting to mysql://admin:s3cretP@ss@db.example.com:3306/mydb"
        result = redact_key_in_text(text)
        assert "s3cretP@ss" not in result
        assert "://admin:****@" in result


class TestSafeErrorMessage:
    def test_wraps_exception_with_redaction(self):
        exc = ValueError("bad key sk-abcdef123456789xyz")
        result = safe_error_message(exc)
        assert "sk-abcdef123456789xyz" not in result
        assert "sk-a" in result


class TestLoggingRedaction:
    def test_redacting_formatter_masks_secret_like_substrings(self):
        formatter = _RedactingFormatter("%(message)s")
        record = logging.LogRecord(
            name="autotabml.tests",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="Using key sk-abcdef123456789xyz for requests",
            args=(),
            exc_info=None,
        )

        formatted = formatter.format(record)

        assert "sk-abcdef123456789xyz" not in formatted
        assert "sk-a" in formatted
