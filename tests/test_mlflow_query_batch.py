"""Regression tests for the batched + cached registry query layer.

Verifies that ``mlflow_query.list_registered_models`` collapses what used to
be an O(N) sequence of ``search_model_versions(name='...')`` calls into a
single batched ``search_model_versions`` call, and that repeated callers
within the TTL window do not hit MLflow at all.
"""

from __future__ import annotations

import time

import pytest

from app.tracking import mlflow_query


class _FakeRegisteredModel:
    def __init__(self, name: str) -> None:
        self.name = name
        self.description = ""
        self.creation_timestamp = 0
        self.last_updated_timestamp = 0
        self.tags = {}
        self.aliases = {}
        self.latest_versions = []


class _FakeVersion:
    def __init__(self, name: str, version: str) -> None:
        self.name = name
        self.version = version
        self.creation_timestamp = 0
        self.last_updated_timestamp = 0
        self.description = ""
        self.source = f"file:///models/{name}/{version}"
        self.run_id = None
        self.run_link = ""
        self.status = "READY"
        self.tags = {}
        self.aliases = []


class _CountingClient:
    """Minimal MlflowClient stub that records every call it receives."""

    def __init__(self, models, versions, *, per_call_latency: float = 0.0) -> None:
        self._models = list(models)
        self._versions = list(versions)
        self._latency = per_call_latency
        self.calls: list[tuple[str, tuple, dict]] = []

    def _record(self, name: str, args, kwargs) -> None:
        if self._latency:
            time.sleep(self._latency)
        self.calls.append((name, args, kwargs))

    def search_registered_models(self, *args, **kwargs):
        self._record("search_registered_models", args, kwargs)
        return list(self._models)

    def search_model_versions(self, *args, **kwargs):
        self._record("search_model_versions", args, kwargs)
        # Honour name='X' filters when invoked the legacy per-model way so we
        # can compare against the unbatched baseline below.
        if args and isinstance(args[0], str) and args[0].startswith("name="):
            wanted = args[0].split("=", 1)[1].strip().strip("'\"")
            return [v for v in self._versions if v.name == wanted]
        return list(self._versions)

    def count(self, name: str) -> int:
        return sum(1 for call in self.calls if call[0] == name)


@pytest.fixture(autouse=True)
def _clear_cache():
    mlflow_query.invalidate_registry_cache()
    yield
    mlflow_query.invalidate_registry_cache()


def _patch_client(monkeypatch, client):
    monkeypatch.setattr(mlflow_query, "is_mlflow_available", lambda: True)
    monkeypatch.setattr(mlflow_query, "_require_mlflow", lambda: None)
    monkeypatch.setattr(
        mlflow_query,
        "_get_client",
        lambda tracking_uri=None, registry_uri=None: client,
    )


def test_list_registered_models_uses_single_batched_version_call(monkeypatch):
    models = [_FakeRegisteredModel(f"model-{i}") for i in range(20)]
    versions = [
        _FakeVersion(f"model-{i}", str(v))
        for i in range(20)
        for v in range(1, 4)
    ]
    client = _CountingClient(models, versions)
    _patch_client(monkeypatch, client)

    summaries = mlflow_query.list_registered_models()

    assert len(summaries) == 20
    assert client.count("search_registered_models") == 1
    # The whole point of the batching: one search_model_versions call total,
    # not one per registered model.
    assert client.count("search_model_versions") == 1
    # Each model still receives its real version count and latest version.
    assert summaries[0].version_count == 3
    assert summaries[0].latest_version == "3"


def test_list_registered_models_pages_through_full_registry(monkeypatch):
    monkeypatch.setattr(mlflow_query, "_REGISTRY_LIST_PAGE_SIZE", 100)
    models = [_FakeRegisteredModel(f"model-{i}") for i in range(205)]
    versions = [_FakeVersion(f"model-{i}", "1") for i in range(205)]

    class _PagedList(list):
        def __init__(self, items, token=None):
            super().__init__(items)
            self.token = token

    class _PagingClient(_CountingClient):
        def search_registered_models(self, *args, **kwargs):
            self._record("search_registered_models", args, kwargs)
            max_results = kwargs.get("max_results", len(self._models))
            page_token = kwargs.get("page_token")
            start = int(page_token or 0)
            end = start + max_results
            next_token = str(end) if end < len(self._models) else None
            return _PagedList(self._models[start:end], token=next_token)

    client = _PagingClient(models, versions)
    _patch_client(monkeypatch, client)

    summaries = mlflow_query.list_registered_models(use_cache=False)

    assert len(summaries) == 205
    assert client.count("search_registered_models") == 3
    assert summaries[-1].name == "model-204"


def test_list_registered_models_falls_back_to_legacy_client_signature(monkeypatch):
    models = [_FakeRegisteredModel(f"model-{i}") for i in range(3)]
    versions = [_FakeVersion(f"model-{i}", "1") for i in range(3)]

    class _LegacyClient(_CountingClient):
        def search_registered_models(self, *args, **kwargs):
            self._record("search_registered_models", args, kwargs)
            if kwargs:
                raise TypeError("unexpected keyword argument 'page_token'")
            return list(self._models)

    client = _LegacyClient(models, versions)
    _patch_client(monkeypatch, client)

    summaries = mlflow_query.list_registered_models(use_cache=False)

    assert [summary.name for summary in summaries] == ["model-0", "model-1", "model-2"]
    assert client.count("search_registered_models") == 2


def test_list_registered_models_caches_results_within_ttl(monkeypatch):
    models = [_FakeRegisteredModel("alpha"), _FakeRegisteredModel("beta")]
    versions = [_FakeVersion("alpha", "1"), _FakeVersion("beta", "1")]
    client = _CountingClient(models, versions)
    _patch_client(monkeypatch, client)

    first = mlflow_query.list_registered_models()
    second = mlflow_query.list_registered_models()
    third = mlflow_query.list_registered_models()

    assert [m.name for m in first] == ["alpha", "beta"]
    assert [m.name for m in second] == ["alpha", "beta"]
    assert [m.name for m in third] == ["alpha", "beta"]
    # Only the first call should reach the client.
    assert client.count("search_registered_models") == 1
    assert client.count("search_model_versions") == 1


def test_list_registered_models_use_cache_false_bypasses_cache(monkeypatch):
    models = [_FakeRegisteredModel("alpha")]
    versions = [_FakeVersion("alpha", "1")]
    client = _CountingClient(models, versions)
    _patch_client(monkeypatch, client)

    mlflow_query.list_registered_models()
    mlflow_query.list_registered_models(use_cache=False)

    assert client.count("search_registered_models") == 2
    assert client.count("search_model_versions") == 2


def test_invalidate_registry_cache_drops_entries(monkeypatch):
    models = [_FakeRegisteredModel("alpha")]
    versions = [_FakeVersion("alpha", "1")]
    client = _CountingClient(models, versions)
    _patch_client(monkeypatch, client)

    mlflow_query.list_registered_models()
    mlflow_query.invalidate_registry_cache()
    mlflow_query.list_registered_models()

    assert client.count("search_registered_models") == 2


def test_create_model_version_invalidates_cache(monkeypatch):
    models = [_FakeRegisteredModel("alpha")]
    versions = [_FakeVersion("alpha", "1")]

    class _MutatingClient(_CountingClient):
        def create_model_version(self, name, *, source, run_id=None, description="", tags=None):
            self._record("create_model_version", (name,), {})
            new_version = _FakeVersion(name, "2")
            self._versions.append(new_version)
            return new_version

    client = _MutatingClient(models, versions)
    _patch_client(monkeypatch, client)

    mlflow_query.list_registered_models()
    mlflow_query.create_model_version("alpha", source="file:///x")
    mlflow_query.list_registered_models()

    # First list, then post-mutation re-list ⇒ the registry list call is made
    # twice. If the cache had not been invalidated, the second list_registered
    # _models call would have served stale data without touching the client.
    assert client.count("search_registered_models") == 2


def test_list_registered_models_batched_is_faster_than_unbatched_baseline(monkeypatch):
    """Measurable latency improvement, simulated with per-call latency.

    With 20 models and 5 ms of artificial per-call latency, the historic
    pattern (1 + N calls) should take ~21 * 5 ms ≈ 105 ms, while the batched
    path makes exactly 2 calls ≈ 10 ms. We assert the batched run is at
    least 3× faster — a conservative bound that survives jitter.
    """

    n_models = 20
    per_call_latency = 0.005  # 5 ms

    models = [_FakeRegisteredModel(f"model-{i}") for i in range(n_models)]
    versions = [_FakeVersion(f"model-{i}", "1") for i in range(n_models)]

    # Batched (current implementation).
    batched_client = _CountingClient(models, versions, per_call_latency=per_call_latency)
    _patch_client(monkeypatch, batched_client)
    mlflow_query.invalidate_registry_cache()
    t0 = time.perf_counter()
    mlflow_query.list_registered_models(use_cache=False)
    batched_elapsed = time.perf_counter() - t0

    # Simulated unbatched baseline: 1 search_registered_models + N filtered
    # search_model_versions calls.
    baseline_client = _CountingClient(models, versions, per_call_latency=per_call_latency)
    t0 = time.perf_counter()
    baseline_client.search_registered_models()
    for m in models:
        baseline_client.search_model_versions(f"name='{m.name}'")
    baseline_elapsed = time.perf_counter() - t0

    assert batched_client.count("search_model_versions") == 1
    assert baseline_client.count("search_model_versions") == n_models
    assert baseline_elapsed > batched_elapsed * 3, (
        f"Expected batched path to be >3x faster. "
        f"batched={batched_elapsed * 1000:.1f}ms, "
        f"baseline={baseline_elapsed * 1000:.1f}ms"
    )



