"""Pydantic schemas for the model registry layer."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class PromotionAction(str, Enum):
    """Supported promotion actions."""

    CHAMPION = "champion"
    CANDIDATE = "candidate"
    ARCHIVED = "archived"


class RegistryModelSummary(BaseModel):
    """Normalized MLflow registered model metadata."""

    name: str
    creation_timestamp: datetime | None = None
    last_updated_timestamp: datetime | None = None
    description: str = ""
    tags: dict[str, str] = Field(default_factory=dict)
    aliases: dict[str, str] = Field(default_factory=dict)
    version_count: int = 0
    latest_version: str | None = None


class RegistryVersionSummary(BaseModel):
    """Normalized MLflow model version metadata."""

    model_name: str
    version: str
    creation_timestamp: datetime | None = None
    last_updated_timestamp: datetime | None = None
    description: str = ""
    source: str | None = None
    run_id: str | None = None
    run_link: str | None = None
    status: str = "UNKNOWN"
    tags: dict[str, str] = Field(default_factory=dict)
    aliases: list[str] = Field(default_factory=list)
    app_status: str | None = None


class PromotionRequest(BaseModel):
    """Request to promote a model version."""

    model_name: str
    version: str
    action: PromotionAction
    description: str | None = None


class PromotionResult(BaseModel):
    """Outcome of a promotion action."""

    model_name: str
    version: str
    action: PromotionAction
    success: bool = True
    alias_changes: list[str] = Field(default_factory=list)
    tag_changes: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
