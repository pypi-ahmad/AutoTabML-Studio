"""Enumerations for AutoTabML Studio configuration."""

from enum import Enum


class WorkspaceMode(str, Enum):
    """First-class workspace modes."""
    DASHBOARD = "dashboard"
    NOTEBOOK = "notebook"


class ExecutionBackend(str, Enum):
    """Execution backends – where ML jobs actually run."""
    COLAB_MCP = "colab_mcp"
    LOCAL = "local"


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    OLLAMA = "ollama"


# Providers available per execution backend
PROVIDERS_BY_BACKEND: dict[ExecutionBackend, list[LLMProvider]] = {
    ExecutionBackend.LOCAL: [
        LLMProvider.OPENAI,
        LLMProvider.ANTHROPIC,
        LLMProvider.GEMINI,
        LLMProvider.OLLAMA,
    ],
    ExecutionBackend.COLAB_MCP: [
        LLMProvider.OPENAI,
        LLMProvider.ANTHROPIC,
        LLMProvider.GEMINI,
        # Ollama intentionally excluded – local-only provider
    ],
}

# Stable fallback default model IDs per provider (used when live list is unavailable)
DEFAULT_MODELS: dict[LLMProvider, str | None] = {
    LLMProvider.OPENAI: "gpt-5.4-mini",
    LLMProvider.ANTHROPIC: "claude-sonnet-4-6",
    LLMProvider.GEMINI: "gemini-2.5-flash",
    LLMProvider.OLLAMA: None,
}
