"""Model registry service – list, inspect, register, and tag models."""

from __future__ import annotations

import logging

from app.errors import log_exception
from app.registry.errors import ModelNotFoundError, PromotionError, RegistryError
from app.registry.schemas import (
    PromotionAction,
    PromotionRequest,
    PromotionResult,
    RegistryModelSummary,
    RegistryVersionSummary,
)
from app.tracking import mlflow_query
from app.tracking.errors import TrackingError

logger = logging.getLogger(__name__)


class RegistryService:
    """High-level service for MLflow model registry operations."""

    def __init__(
        self,
        *,
        tracking_uri: str | None = None,
        registry_uri: str | None = None,
        champion_alias: str = "champion",
        candidate_alias: str = "candidate",
        archived_tag_key: str = "app.status",
    ) -> None:
        self._tracking_uri = tracking_uri
        self._registry_uri = registry_uri
        self._champion_alias = champion_alias
        self._candidate_alias = candidate_alias
        self._archived_tag_key = archived_tag_key

    def list_models(self) -> list[RegistryModelSummary]:
        """List all registered models."""

        return mlflow_query.list_registered_models(
            tracking_uri=self._tracking_uri,
            registry_uri=self._registry_uri,
        )

    def get_model(self, name: str) -> RegistryModelSummary:
        """Get a single registered model by name."""

        return mlflow_query.get_registered_model(
            name,
            tracking_uri=self._tracking_uri,
            registry_uri=self._registry_uri,
        )

    def list_versions(self, name: str) -> list[RegistryVersionSummary]:
        """List all versions for a registered model."""

        return mlflow_query.list_model_versions(
            name,
            tracking_uri=self._tracking_uri,
            registry_uri=self._registry_uri,
        )

    def get_version(self, name: str, version: str) -> RegistryVersionSummary:
        """Get a specific model version."""

        return mlflow_query.get_model_version(
            name,
            version,
            tracking_uri=self._tracking_uri,
            registry_uri=self._registry_uri,
        )

    def get_version_by_alias(self, name: str, alias: str) -> RegistryVersionSummary:
        """Get the model version currently assigned to an alias."""

        return mlflow_query.get_model_version_by_alias(
            name,
            alias,
            tracking_uri=self._tracking_uri,
            registry_uri=self._registry_uri,
        )

    def register_model(
        self,
        name: str,
        *,
        source: str,
        run_id: str | None = None,
        description: str = "",
        tags: dict[str, str] | None = None,
    ) -> RegistryVersionSummary:
        """Register a model and create its first version.

        Creates the registered model if it does not exist yet.
        """

        try:
            mlflow_query.get_registered_model(
                name,
                tracking_uri=self._tracking_uri,
                registry_uri=self._registry_uri,
            )
        except ModelNotFoundError:
            mlflow_query.create_registered_model(
                name,
                description=description,
                tracking_uri=self._tracking_uri,
                registry_uri=self._registry_uri,
            )

        version = mlflow_query.create_model_version(
            name,
            source=source,
            run_id=run_id,
            description=description,
            tags=tags,
            tracking_uri=self._tracking_uri,
            registry_uri=self._registry_uri,
        )
        return version

    def promote(self, request: PromotionRequest) -> PromotionResult:
        """Execute a promotion action on a model version."""

        alias_changes: list[str] = []
        tag_changes: list[str] = []
        warnings: list[str] = []

        try:
            mlflow_query.get_model_version(
                request.model_name,
                request.version,
                tracking_uri=self._tracking_uri,
                registry_uri=self._registry_uri,
            )
        except (RegistryError, TrackingError) as exc:
            log_exception(
                logger,
                exc,
                operation="registry.promote.lookup_version",
                context={"model": request.model_name, "version": request.version},
            )
            raise PromotionError(
                f"Cannot promote: version '{request.version}' of model "
                f"'{request.model_name}' not found: {exc}"
            ) from exc

        try:
            model = self.get_model(request.model_name)
        except (RegistryError, TrackingError) as exc:
            log_exception(
                logger,
                exc,
                operation="registry.promote.get_model",
                context={"model": request.model_name},
            )
            model = RegistryModelSummary(name=request.model_name)
            warnings.append(
                "Could not inspect existing alias state; status tags may need manual cleanup."
            )

        try:
            existing_versions = {
                version.version: version
                for version in self.list_versions(request.model_name)
            }
        except (RegistryError, TrackingError) as exc:
            log_exception(
                logger,
                exc,
                operation="registry.promote.list_versions",
                context={"model": request.model_name},
            )
            existing_versions = {}
            warnings.append(
                "Could not inspect existing version tags; status tags may need manual cleanup."
            )

        current_aliases = dict(model.aliases)
        desired_aliases = dict(current_aliases)

        if request.action == PromotionAction.CHAMPION:
            if current_aliases.get(self._champion_alias) != request.version:
                mlflow_query.set_model_alias(
                    request.model_name,
                    self._champion_alias,
                    request.version,
                    tracking_uri=self._tracking_uri,
                    registry_uri=self._registry_uri,
                )
                alias_changes.append(f"Set alias '{self._champion_alias}' -> v{request.version}")
            desired_aliases[self._champion_alias] = request.version

        elif request.action == PromotionAction.CANDIDATE:
            if current_aliases.get(self._candidate_alias) != request.version:
                mlflow_query.set_model_alias(
                    request.model_name,
                    self._candidate_alias,
                    request.version,
                    tracking_uri=self._tracking_uri,
                    registry_uri=self._registry_uri,
                )
                alias_changes.append(f"Set alias '{self._candidate_alias}' -> v{request.version}")
            desired_aliases[self._candidate_alias] = request.version

        elif request.action == PromotionAction.ARCHIVED:
            for alias, alias_version in current_aliases.items():
                if alias_version == request.version:
                    mlflow_query.delete_model_alias(
                        request.model_name,
                        alias,
                        tracking_uri=self._tracking_uri,
                        registry_uri=self._registry_uri,
                    )
                    desired_aliases.pop(alias, None)
                    alias_changes.append(f"Removed alias '{alias}' from v{request.version}")

        self._sync_status_tags(
            model_name=request.model_name,
            request=request,
            current_aliases=current_aliases,
            desired_aliases=desired_aliases,
            existing_versions=existing_versions,
            tag_changes=tag_changes,
        )

        return PromotionResult(
            model_name=request.model_name,
            version=request.version,
            action=request.action,
            success=True,
            alias_changes=alias_changes,
            tag_changes=tag_changes,
            warnings=warnings,
        )

    def _sync_status_tags(
        self,
        *,
        model_name: str,
        request: PromotionRequest,
        current_aliases: dict[str, str],
        desired_aliases: dict[str, str],
        existing_versions: dict[str, RegistryVersionSummary],
        tag_changes: list[str],
    ) -> None:
        affected_versions = {
            request.version,
            *current_aliases.values(),
            *desired_aliases.values(),
        }

        for version in sorted(affected_versions, key=_safe_version_sort_key):
            desired_status = self._desired_status_for_version(
                version,
                desired_aliases,
                archived_version=request.version if request.action == PromotionAction.ARCHIVED else None,
            )
            existing_status = existing_versions.get(version)
            current_status = (
                existing_status.tags.get(self._archived_tag_key)
                if existing_status is not None
                else None
            )

            if desired_status == current_status:
                continue

            if desired_status is None:
                if current_status is not None:
                    mlflow_query.delete_model_version_tag(
                        model_name,
                        version,
                        self._archived_tag_key,
                        tracking_uri=self._tracking_uri,
                        registry_uri=self._registry_uri,
                    )
                    tag_changes.append(f"Cleared tag '{self._archived_tag_key}' from v{version}")
                continue

            mlflow_query.set_model_version_tag(
                model_name,
                version,
                self._archived_tag_key,
                desired_status,
                tracking_uri=self._tracking_uri,
                registry_uri=self._registry_uri,
            )
            tag_changes.append(f"Set tag '{self._archived_tag_key}' = '{desired_status}' on v{version}")

    def _desired_status_for_version(
        self,
        version: str,
        desired_aliases: dict[str, str],
        *,
        archived_version: str | None,
    ) -> str | None:
        if archived_version == version:
            return "archived"
        if desired_aliases.get(self._champion_alias) == version:
            return "champion"
        if desired_aliases.get(self._candidate_alias) == version:
            return "candidate"
        return None


def _safe_version_sort_key(version: str) -> int:
    try:
        return int(version)
    except ValueError:
        return 0
