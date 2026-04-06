"""Run history service – list, filter, and inspect MLflow runs."""

from __future__ import annotations

from app.tracking import mlflow_query
from app.tracking.filters import (
    RunHistoryFilter,
    RunHistorySort,
    RunSortField,
    SortDirection,
    build_mlflow_filter_string,
)
from app.tracking.schemas import RunDetailView, RunHistoryItem, RunType


class HistoryService:
    """High-level service for querying and inspecting run history."""

    def __init__(
        self,
        *,
        tracking_uri: str | None = None,
        default_experiment_names: list[str] | None = None,
        default_limit: int = 50,
    ) -> None:
        self._tracking_uri = tracking_uri
        self._default_experiment_names = default_experiment_names or []
        self._default_limit = default_limit

    def list_runs(
        self,
        *,
        history_filter: RunHistoryFilter | None = None,
        sort: RunHistorySort | None = None,
        limit: int | None = None,
    ) -> list[RunHistoryItem]:
        """List runs with optional filtering and sorting."""

        experiment_ids, name_map = self._resolve_experiment_ids(history_filter)
        filter_string = build_mlflow_filter_string(history_filter)
        order_by = _build_order_by(sort)

        runs = mlflow_query.search_runs(
            experiment_ids=experiment_ids or None,
            filter_string=filter_string,
            order_by=order_by,
            max_results=limit or self._default_limit,
            tracking_uri=self._tracking_uri,
            experiment_name_map=name_map,
        )

        runs = self._apply_client_side_filters(runs, history_filter)

        effective_sort = sort or RunHistorySort()
        runs = _sort_runs(runs, effective_sort)

        return runs

    def resolve_run_id(self, run_id_prefix: str) -> str:
        """Resolve a possibly truncated run ID prefix to a full 32-char ID.

        If *run_id_prefix* already looks like a complete hex ID (32 chars)
        it is returned as-is.  Otherwise we search recent runs for a
        unique prefix match.

        Raises ``RunNotFoundError`` when zero or multiple runs match.
        """
        from app.tracking.errors import RunNotFoundError

        cleaned = run_id_prefix.strip()
        if len(cleaned) >= 32:
            return cleaned

        runs = self.list_runs(limit=500)
        matches = [r for r in runs if r.run_id.startswith(cleaned)]
        if len(matches) == 1:
            return matches[0].run_id
        if len(matches) == 0:
            raise RunNotFoundError(f"No run found matching prefix '{cleaned}'.")
        raise RunNotFoundError(
            f"Ambiguous prefix '{cleaned}': matches {len(matches)} runs. "
            "Provide more characters to disambiguate."
        )

    def get_run_detail(self, run_id: str) -> RunDetailView:
        """Fetch extended detail for a single run."""

        name_map = self._build_full_name_map()
        return mlflow_query.get_run(
            run_id,
            tracking_uri=self._tracking_uri,
            experiment_name_map=name_map,
        )

    def _resolve_experiment_ids(
        self,
        history_filter: RunHistoryFilter | None,
    ) -> tuple[list[str] | None, dict[str, str]]:
        """Resolve experiment name(s) to ids, building a name lookup map."""

        names = set(self._default_experiment_names)
        if history_filter and history_filter.experiment_names:
            names = set(history_filter.experiment_names)
        if not names:
            return None, self._build_full_name_map()

        name_map: dict[str, str] = {}
        ids: list[str] = []
        for name in names:
            try:
                info = mlflow_query.get_experiment_by_name(
                    name, tracking_uri=self._tracking_uri,
                )
                ids.append(info.experiment_id)
                name_map[info.experiment_id] = info.name
            except Exception:
                continue
        return ids if ids else None, name_map

    def _build_full_name_map(self) -> dict[str, str]:
        try:
            experiments = mlflow_query.list_experiments(tracking_uri=self._tracking_uri)
            return {exp.experiment_id: exp.name for exp in experiments}
        except Exception:
            return {}

    def _apply_client_side_filters(
        self,
        runs: list[RunHistoryItem],
        history_filter: RunHistoryFilter | None,
    ) -> list[RunHistoryItem]:
        """Apply filters that cannot be expressed in MLflow filter syntax."""

        if history_filter is None:
            return runs

        if history_filter.run_type is not None:
            runs = [r for r in runs if r.run_type == history_filter.run_type]

        if history_filter.model_name:
            target = history_filter.model_name.lower()
            runs = [r for r in runs if r.model_name and target in r.model_name.lower()]

        if history_filter.date_from is not None:
            runs = [r for r in runs if r.start_time and r.start_time >= history_filter.date_from]

        if history_filter.date_to is not None:
            runs = [r for r in runs if r.start_time and r.start_time <= history_filter.date_to]

        return runs


def _build_order_by(sort: RunHistorySort | None) -> list[str]:
    if sort is None:
        return ["attributes.start_time DESC"]
    direction = "ASC" if sort.direction == SortDirection.ASCENDING else "DESC"
    if sort.field == RunSortField.START_TIME:
        return [f"attributes.start_time {direction}"]
    if sort.field == RunSortField.DURATION:
        return [f"attributes.start_time {direction}"]
    return [f"attributes.start_time {direction}"]


def _sort_runs(runs: list[RunHistoryItem], sort: RunHistorySort) -> list[RunHistoryItem]:
    reverse = sort.direction == SortDirection.DESCENDING
    if sort.field == RunSortField.START_TIME:
        return sorted(runs, key=lambda r: r.start_time or 0, reverse=reverse)
    if sort.field == RunSortField.DURATION:
        return sorted(runs, key=lambda r: r.duration_seconds or 0.0, reverse=reverse)
    if sort.field == RunSortField.MODEL_NAME:
        return sorted(runs, key=lambda r: (r.model_name or "").lower(), reverse=reverse)
    if sort.field == RunSortField.PRIMARY_SCORE:
        return sorted(
            runs,
            key=lambda r: r.primary_metric_value if r.primary_metric_value is not None else float("-inf"),
            reverse=reverse,
        )
    return runs
