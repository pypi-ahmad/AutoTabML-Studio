# Graph Report - AutoTabML-Studio  (2026-08-14)

## Corpus Check
- 557 files · ~1,606,411 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 4621 nodes · 13137 edges · 208 communities (182 shown, 26 thin omitted)
- Extraction: 91% EXTRACTED · 9% INFERRED · 0% AMBIGUOUS · INFERRED: 1193 edges (avg confidence: 0.51)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `347ebb24`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- ExecutionBackend
- BenchmarkConfig
- main
- test_flaml_automl.py
- ProfilingMode
- flaml/service.py
- test_prediction.py
- LLMProvider
- test_benchmark.py
- test_colab_mcp_backend.py
- DatasetInputSpec
- SavedModelMetadata
- dataset_workspace.py
- benchmark_page.py
- url_loader.py
- PredictionWorkflowService
- mlflow_query.py
- go_to_page
- ValidationRuleConfig
- generate_job_notebook
- test_uci_real_datasets.py
- JobRecord
- PredictionHistoryStore
- IngestionSourceType
- pyrightconfig.json
- AppSettings
- gx_runner.py
- RegistryService
- ColabMCPExecutionBackend
- ExperimentResultBundle
- PredictionService
- loader.py
- AppMetadataStore
- storage/models.py
- test_modeling_exception_handling.py
- batch_uci_runner.py
- gather_with_concurrency
- models_page.py
- ExperimentTaskType
- pycaret/summary.py
- ui_cache.py
- Architecture Guide
- test_cli.py
- app/errors.py
- pycaret/errors.py
- cuda_summary
- UCIRepoLoader
- safe_http.py
- SQLiteConnector
- test_registry.py
- AppJobType
- validate_dataset
- safe_error_message
- ExperimentWorkflowService
- test_hardening_smoke.py
- run_app_rules
- BaseTracker
- FlamlConfig
- list_registered_models
- CheckpointResolver
- log_exception
- TabFMConfig
- observability/__init__.py
- InMemoryMetricsBackend
- registry.py
- safe_fetch_async
- SafeFetchPolicy
- RunHistoryItem
- test_tracking.py
- HistoryService
- test_ingestion_utils.py
- ProfilingResultSummary
- validate_public_release_metadata
- history_page.py
- ExcelLoader
- experiment_page.py
- metrics.py
- update-graph.cjs
- compare_page.py
- BatchRunRecord
- ComparisonService
- capture_screenshots.py
- verify_optional_deps.py
- test_observability.py
- PredictionRequest
- LocalArtifactManager
- autorun.py
- ._make_args
- Path
- Path
- RunHistoryFilter
- test_product_improvements.py
- _FakeExperimentBase
- _cli_error
- drift.py
- tracking/errors.py
- CSVLoader
- redact_key_in_text
- TestFlamlRunType
- explain_global
- TimesFMConfig
- load_dataset_async
- foundation_models_page.py
- SavedLocalModelRecord
- test_e2e_local_smoke.py
- migrations.py
- ProjectRecord
- configure_logging
- cli.py
- test_list_cached_registered_models_uses_cache_until_invalidated
- build_provenance
- _FakeMLflow
- model_testing_page.py
- ArtifactKind
- RunHistorySort
- AutoTabML Studio Social Preview
- profile_dataset
- build-batches-10-18.cjs
- log_ui_exception
- test_dataset_workspace.py
- Code of Conduct
- AutoML Modeling
- build-batches-6-9.cjs
- manager.py
- start_span
- Namespace
- ua-assemble.cjs
- export_deployment_bundle
- fetch_url_bytes
- run_auto_run
- mask_secret
- Benchmark
- History view
- EDA / Profiling Dashboard
- Settings view
- calculate_model_cost
- _DummyTracker
- Benchmark configuration form
- Data Validation
- TestCmdRegistryGates
- TestCliOutputEncoding
- TestLabelMaps
- store.py
- cmd_experiment_run
- MLflow run comparison view
- Dashboard
- Loaded dataset 'diabetes'
- Prediction Center Interface
- e2e-demo-classifier
- IngestionError
- extract-ua-batches-10-13.cjs
- process-ua-batches-19-26.cjs
- notebook_page.py
- recorders.py
- ua-fingerprint-input.cjs
- _is_blocked_ip
- BatchRunItemRecord
- Load Dataset Sources
- Recent Prediction Jobs
- AutoTabML Studio v0.2.0
- ProfilingSettings
- FlamlAutoMLService
- TestRuntimeState
- .run
- .trash-1786689789/tmp/ua-arch-analyze.js
- app/__init__.py
- modeling/__init__.py
- Highlights
- ua-arch-write-layers.js
- profiling/__init__.py
- Leaderboard results
- .trash-1786689789/tmp/ua-inline-validate.cjs
- ua-tour-analyze.js
- Demo assets documentation
- AutoTabML social preview
- AutoTabML Studio Architecture Diagram
- Dependabot dependency update policy
- Bug Report issue template
- Feature Request issue template
- autotabml-studio
- Pre-commit quality hooks
- Security Policy
- Changed Working Tree Record
- TestFlamlSchemas
- BackgroundJobService
- FlamlSearchConfig
- ExperimentInfo
- evaluate_model
- .load_raw_dataframe
- get_page_registry
- .prepare_session
- TestMetricSortDirection
- TestFlamlPageHelpers
- TestRunSummary
- TestDefaultModels
- DataFrame
- .trash-1786706008/tmp/ua-arch-analyze.js
- .preview
- _clear_registry_cache
- .trash-1786706008/tmp/ua-inline-validate.cjs

## God Nodes (most connected - your core abstractions)
1. `ExecutionBackend` - 210 edges
2. `AppSettings` - 202 edges
3. `DatasetInputSpec` - 183 edges
4. `WorkspaceMode` - 161 edges
5. `IngestionSourceType` - 91 edges
6. `AppMetadataStore` - 83 edges
7. `log_exception()` - 80 edges
8. `ProfilingMode` - 66 edges
9. `LLMProvider` - 65 edges
10. `RegistryService` - 58 edges

## Surprising Connections (you probably didn't know these)
- `Coverage Gate` --semantically_similar_to--> `Hermetic Test Strategy`  [INFERRED] [semantically similar]
  .github/workflows/ci.yml → docs/developer-guide.md
- `Local-First Architecture` --semantically_similar_to--> `Local-First Automated Machine Learning Workbench`  [INFERRED] [semantically similar]
  docs/architecture.md → README.md
- `Shared Service Layer Architecture` --semantically_similar_to--> `Shared UI and CLI Service Layer`  [INFERRED] [semantically similar]
  docs/architecture.md → README.md
- `main()` --calls--> `Counter`  [INFERRED]
  .ua/.trash-1786706008/tmp/ua-tour-analyze.py → app/observability/metrics.py
- `TestBuildBackend` --uses--> `ColabMCPExecutionBackend`  [INFERRED]
  tests/test_colab_mcp_backend.py → app/backends/colab_mcp_backend.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **** —  [EXTRACTED 1.00]
- **docs_assets_screenshots_dashboard_overview_workspace_overview** — docs_assets_screenshots_dashboard_overview_dashboard, docs_assets_screenshots_dashboard_overview_active_dataset, docs_assets_screenshots_dashboard_overview_workspace_metrics, docs_assets_screenshots_dashboard_overview_recent_local_jobs [EXTRACTED 0.98]
- **docs_assets_screenshots_dataset_intake_active_dataset_workflows** — docs_assets_screenshots_dataset_intake_diabetes_dataset, docs_assets_screenshots_dataset_intake_validation, docs_assets_screenshots_dataset_intake_profiling, docs_assets_screenshots_dataset_intake_benchmark, docs_assets_screenshots_dataset_intake_experiment [EXTRACTED 1.00]
- **Quality Assurance Delivery Surface** — github_workflows_ci_workflow, github_workflows_security_workflow, github_workflows_release_readiness_workflow, docs_developer_guide_test_strategy [INFERRED 0.85]
- **Local-First Workbench Documentation** — readme_local_first_workbench, docs_architecture_local_first, docs_operations_local_runtime_layout [INFERRED 0.85]
- **Primary AutoTabML Journey** — docs_autotabml_studio_architecture_studio_workbench, docs_autotabml_studio_architecture_dataset_ingestion, docs_autotabml_studio_architecture_automl_modeling, docs_autotabml_studio_architecture_prediction_service, docs_autotabml_studio_architecture_trusted_artifacts [EXTRACTED 1.00]

## Communities (208 total, 26 thin omitted)

### Community 0 - "ExecutionBackend"
Cohesion: 0.03
Nodes (113): ExecutionBackend, str, Execution backends – where ML jobs actually run., First-class workspace modes., WorkspaceMode, ArtifactSettings, DatabaseSettings, MLflowSettings (+105 more)

### Community 1 - "BenchmarkConfig"
Cohesion: 0.04
Nodes (85): BaseService, Common modeling-service configuration and helpers., Return a configured tracker instance when MLflow tracking is enabled., BenchmarkConfigurationError, BenchmarkDependencyError, BenchmarkError, BenchmarkExecutionError, BenchmarkTargetError (+77 more)

### Community 2 - "main"
Cohesion: 0.13
Nodes (25): _add_prediction_model_source_args(), cmd_compare_runs(), cmd_doctor(), cmd_history_show(), cmd_info(), cmd_init_local_storage(), cmd_registry_list(), cmd_registry_promote() (+17 more)

### Community 3 - "test_flaml_automl.py"
Cohesion: 0.12
Nodes (21): FlamlSettings, Configuration for FLAML AutoML workflows., LocalFlamlModelLoader, Load local saved FLAML model artifacts., PredictionScorer, Normalize scoring across local PyCaret and MLflow / sklearn-like models., Tests for the FLAML AutoML integration., Verify the CLI history-list parser accepts 'flaml' as a run type. (+13 more)

### Community 4 - "ProfilingMode"
Cohesion: 0.10
Nodes (31): ProfilingMode, str, Profiling report modes., ProfilingConfig, Pydantic schemas for profiling inputs and outputs., User-facing configuration for a profiling run., maybe_sample(), DataFrame (+23 more)

### Community 5 - "flaml/service.py"
Cohesion: 0.04
Nodes (70): FlamlArtifactsWriter, Path, Artifact generation for FLAML AutoML runs., Shared-path artifact writer for FLAML result bundles., Write FLAML artifacts to disk and return their paths., write_flaml_artifacts(), FlamlAutoMLError, FlamlConfigurationError (+62 more)

### Community 6 - "test_prediction.py"
Cohesion: 0.06
Nodes (67): Artifact generation for prediction jobs., Prediction service facade., Run one-row prediction., Run batch prediction., build_batch_prediction_result(), infer_input_source_type(), DataFrame, Batch prediction helpers. (+59 more)

### Community 7 - "LLMProvider"
Cohesion: 0.03
Nodes (86): BaseExecutionBackend, Abstract execution backend interface., Common interface for all execution backends., Check that the backend is reachable / properly configured., _find_uvx(), Colab MCP execution backend – connects to Google Colab via MCP. This backend…, Return the path to ``uvx`` if it is installed, else *None*., Execution backend package – factory for backend instances. (+78 more)

### Community 8 - "test_benchmark.py"
Cohesion: 0.10
Nodes (32): BenchmarkSettings, Configuration for baseline model benchmarking., LazyPredictBenchmarkService, Path, Benchmark service backed by LazyClassifier/LazyRegressor., MLflowBenchmarkTracker, Lightweight MLflow tracking wrapper for benchmark runs., classification_df() (+24 more)

### Community 9 - "test_colab_mcp_backend.py"
Cohesion: 0.11
Nodes (14): build_backend(), Create the execution backend instance for the given enum value., LocalExecutionBackend, Any, ExecutionSettings, Settings related to the execution backend., Path, Tests for Colab MCP backend and notebook infrastructure. (+6 more)

### Community 10 - "DatasetInputSpec"
Cohesion: 0.08
Nodes (26): load_dataset(), Public entry point for full dataset loading., DatasetInputSpec, model_validator, Canonical input contract for all supported ingestion paths., Return the human-readable locator for lineage and logging., Validate source-specific input requirements., _dataset_cache_signature() (+18 more)

### Community 11 - "SavedModelMetadata"
Cohesion: 0.05
Nodes (66): discover_saved_benchmark_models(), load_saved_benchmark_model(), load_saved_benchmark_model_metadata_file(), Path, Trusted benchmark model discovery and loading helpers., Parse a benchmark saved-model metadata file, returning None when invalid., Discover trusted benchmark models from checksum-backed metadata sidecars., Load a trusted benchmark model from a skops artifact. (+58 more)

### Community 12 - "dataset_workspace.py"
Cohesion: 0.06
Nodes (59): Guided Auto Run page., _render_active_job(), render_autorun_page(), _detect_completed_steps(), _friendly_job_name(), _load_example_dataset(), Dashboard page – professional workspace home., Return the highest *completed* step number (0 = nothing done yet). Heuristic… (+51 more)

### Community 13 - "benchmark_page.py"
Cohesion: 0.13
Nodes (18): build_benchmark_run_key(), default_ranking_metric_for_task(), Streamlit page for LazyPredict benchmark execution and results., Return a stable session-state key for benchmark results., Resolve a LazyPredict model name to its sklearn-compatible estimator class., Render controls to retrain and save the best model from benchmark results., Return the default ranking metric to prefill in the UI., render_benchmark_page() (+10 more)

### Community 14 - "url_loader.py"
Cohesion: 0.10
Nodes (33): Raised when a remote resource cannot be reached or inspected safely., RemoteAccessError, async_fetch_url_bytes(), async_fetch_url_to_temp_file(), DownloadedURLFile, _fetch_sniff_sample(), _fetch_sniff_sample_async(), fetch_url_text() (+25 more)

### Community 15 - "PredictionWorkflowService"
Cohesion: 0.08
Nodes (20): Page-layer services that keep Streamlit pages focused on rendering., ModelTestingEvaluation, ModelTestingRunResult, ModelTestingSelection, PredictionExecutionConfig, PredictionWorkflowService, Any, DataFrame (+12 more)

### Community 16 - "mlflow_query.py"
Cohesion: 0.07
Nodes (54): Raised when the MLflow model registry backend is not available., Raised when a model version cannot be found., RegistryUnavailableError, VersionNotFoundError, create_model_version(), create_registered_model(), delete_model_alias(), delete_model_version_tag() (+46 more)

### Community 17 - "go_to_page"
Cohesion: 0.10
Nodes (35): _build_schema_frame(), DataFrame, Dedicated Streamlit page for dataset intake and active selection., render_dataset_intake_page(), _render_uci_source_details(), go_to_page(), Navigate to another registered page within the Streamlit app., Streamlit profiling / EDA page – run and view dataset profiling results. (+27 more)

### Community 18 - "ValidationRuleConfig"
Cohesion: 0.08
Nodes (28): Configuration for the data validation layer., ValidationSettings, User-facing configuration for a validation run., ValidationRuleConfig, constant_df(), duplicate_rows_df(), empty_df(), good_df() (+20 more)

### Community 19 - "generate_job_notebook"
Cohesion: 0.09
Nodes (38): _benchmark_cells(), _experiment_cells(), _flaml_cells(), generate_job_notebook(), _json_literal(), _json_string(), _markdown_cell(), NotebookGenerationError (+30 more)

### Community 20 - "test_uci_real_datasets.py"
Cohesion: 0.11
Nodes (26): auto_mpg_dataset(), _fetch_uci(), heart_disease_dataset(), iris_dataset(), DataFrame, fixture, Path, Real-dataset integration tests using UCI ML Repository. These tests fetch… (+18 more)

### Community 21 - "JobRecord"
Cohesion: 0.20
Nodes (6): JobRecord, Local app job record used for dashboard/history convenience., JobRepository, Connection, Row, CRUD operations for job rows.

### Community 22 - "PredictionHistoryStore"
Cohesion: 0.12
Nodes (12): PredictionHistoryStore, Path, Persist and query recent prediction jobs via newline-delimited JSON., Append one prediction-history record to disk., Return recent prediction jobs ordered newest-first., _FakeClassifier, _FakeRegressor, _make_loaded_model() (+4 more)

### Community 23 - "IngestionSourceType"
Cohesion: 0.08
Nodes (42): BaseLoader, Base abstractions shared by all ingestion loaders., Common load lifecycle for all dataset sources., Load, normalize, and enrich a dataset from the supplied input spec., Async counterpart of :meth:`load` for I/O-heavy loader implementations., Load only a preview slice where the underlying loader supports it., Async counterpart of :meth:`preview`., Ensure the loader is being used for a compatible source type. (+34 more)

### Community 24 - "pyrightconfig.json"
Cohesion: 0.04
Nodes (45): exclude, ignore, include, pythonPlatform, pythonVersion, reportArgumentType, reportAttributeAccessIssue, reportCallIssue (+37 more)

### Community 25 - "AppSettings"
Cohesion: 0.06
Nodes (19): AppSettings, model_validator, setter, Top-level application configuration with safe defaults., Backward-compatible alias for the canonical MLflow settings section., Return the verified stable fallback model id for the given (or current)…, default_settings(), fixture (+11 more)

### Community 26 - "gx_runner.py"
Cohesion: 0.08
Nodes (35): Exception, Custom exceptions for the validation layer., Raised when the validation infrastructure cannot be initialized., Raised when a validation rule configuration is invalid., Base exception for validation failures., RuleConfigError, ValidationError, ValidationSetupError (+27 more)

### Community 27 - "RegistryService"
Cohesion: 0.12
Nodes (16): Execute a promotion action on a model version., High-level service for MLflow model registry operations., Get a single registered model by name., List all versions for a registered model., RegistryService, PromotionRequest, Request to promote a model version., _FakeModelVersion (+8 more)

### Community 28 - "ColabMCPExecutionBackend"
Cohesion: 0.09
Nodes (15): ColabMCPExecutionBackend, Any, Execute a job by calling Colab MCP tools. ``job_payload`` must include a…, Call ``open_colab_browser_connection`` to link to a Colab notebook., Return the currently available MCP tool names., Tear down the MCP session and server subprocess., Execution backend that delegates work to a Google Colab runtime via MCP., Check that ``uvx`` is installed and the MCP SDK is importable. (+7 more)

### Community 29 - "ExperimentResultBundle"
Cohesion: 0.13
Nodes (18): PyCaretExecutionError, Raised when a PyCaret operation fails., list_available_metrics(), Return normalized metric rows from the active experiment., _build_metrics(), _build_run_name(), Backward-compatible experiment tracker entrypoint., ExperimentResultBundle (+10 more)

### Community 30 - "PredictionService"
Cohesion: 0.09
Nodes (33): DriftBaseline, DataFrame, Path, Persist prediction artifacts and return their paths., _render_markdown(), write_prediction_artifacts(), PredictionService, Path (+25 more)

### Community 31 - "loader.py"
Cohesion: 0.07
Nodes (43): load_model_artifact(), Load a saved model artifact without rebuilding a full setup run., ModelDiscoveryError, ModelLoadError, Raised when a model source cannot be discovered or resolved cleanly., Raised when a prediction model cannot be loaded., Raised when a model artifact fails trust-boundary validation., TrustedArtifactError (+35 more)

### Community 32 - "AppMetadataStore"
Cohesion: 0.11
Nodes (18): AppMetadataStore, Path, Repository facade for local workspace metadata. The store composes the per-…, _dataset(), _job(), _now(), datetime, fixture (+10 more)

### Community 33 - "storage/models.py"
Cohesion: 0.10
Nodes (20): Local metadata storage exports., AppJobStatus, BatchItemStatus, BatchRunStatus, Enum, str, Typed records for the local app metadata database., Overall batch run status. (+12 more)

### Community 34 - "test_modeling_exception_handling.py"
Cohesion: 0.07
Nodes (37): get_mlflow_module(), is_mlflow_available(), mlflow_exception_types(), BaseException, Shared base classes for modeling services, trackers, and artifact writers., Return True when mlflow is importable., Import and return the mlflow module., Return common MLflow exception types plus generic boundary failures. (+29 more)

### Community 35 - "batch_uci_runner.py"
Cohesion: 0.11
Nodes (25): build_200_dataset_list(), main(), Batch UCI dataset runner – runs validate → profile → benchmark for 200 NEW UCI…, Combine the new UCI datasets + extra re-framed datasets to get 200., _build_resume_state(), _declared_target_from_item(), _detect_target_and_task(), main() (+17 more)

### Community 36 - "gather_with_concurrency"
Cohesion: 0.10
Nodes (25): gather_with_concurrency(), Any, BaseException, T, Small async concurrency helpers used across the app. These helpers exist so…, Run awaitables concurrently with at most ``limit`` running at a time. ``limit``…, Run a blocking ``func`` over many argument batches concurrently in threads.…, to_thread_many() (+17 more)

### Community 37 - "models_page.py"
Cohesion: 0.16
Nodes (20): Path, Streamlit page for browsing all saved models with details., Render an expander card for a PyCaret experiment model., Render an expander card for a benchmark-saved model., Render an expander card for an MLflow registry model., Render an expander card for a FLAML-saved model., Render a clearly restricted saved TabFM context., _render_benchmark_model_card() (+12 more)

### Community 38 - "ExperimentTaskType"
Cohesion: 0.11
Nodes (18): generate_evaluation_plots(), pushd(), Path, Evaluation helpers for plots and optional interactive evaluation., Temporarily change the working directory., Generate evaluation plot artifacts and continue past unsupported plots., _resolve_plot_path(), ExperimentTaskType (+10 more)

### Community 39 - "pycaret/summary.py"
Cohesion: 0.13
Nodes (28): ExperimentSortDirection, Metric ordering direction., metric_sort_direction(), Return the expected ordering direction for a metric name., _coerce_bool(), coerce_float(), extract_mean_metrics(), _first_existing_column() (+20 more)

### Community 40 - "ui_cache.py"
Cohesion: 0.11
Nodes (32): Streamlit page for the model registry., Render the manual model registration form., _render_register_section(), render_registry_page(), _section_cache_controls(), _get_experiment_workflow_service_resource(), _get_flaml_automl_service_resource(), get_history_service() (+24 more)

### Community 41 - "Architecture Guide"
Cohesion: 0.06
Nodes (35): Changelog, AutoTabML Container Service, Docker Compose Stack, Architecture Guide, Architecture Diagram HTML, Local-First Architecture, Reproducible Runs, Security Model (+27 more)

### Community 42 - "test_cli.py"
Cohesion: 0.07
Nodes (23): BaseModel, One startup diagnostic item., Initialization result for local app runtime resources., StartupIssue, StartupStatus, Tests for CLI helpers and default wiring., Direct functional tests for cmd_history_show., Direct functional tests for cmd_registry_register. (+15 more)

### Community 43 - "app/errors.py"
Cohesion: 0.08
Nodes (27): Cross-cutting error-handling utilities. This module provides: *…, AutoTabML Studio – Streamlit entry point., glossary_definition(), metric_explanation(), Shared glossary, metric explainers, and inline term helpers. Provides plain-…, Return a plain-English explanation for a metric, or None if unknown., Return a plain-English definition for a glossary term, or None if unknown., Render a collapsible glossary in the sidebar. (+19 more)

### Community 44 - "pycaret/errors.py"
Cohesion: 0.21
Nodes (12): Exception, PyCaretDependencyError, PyCaretExperimentError, PyCaretTargetError, PyCaretTrackingError, Custom exceptions for the PyCaret experiment layer., Raised when the optional PyCaret dependency is unavailable., Raised when the selected target column cannot be used. (+4 more)

### Community 45 - "cuda_summary"
Cohesion: 0.11
Nodes (20): cuda_device_name(), cuda_summary(), _driver_probe(), is_cuda_available(), CUDA / GPU detection utilities for AutoTabML Studio., Lightweight probe for the NVIDIA driver library without importing torch. Loads…, Return True when a CUDA-capable GPU is reachable from the current Python…, Return the name of the first CUDA device, or None. (+12 more)

### Community 46 - "UCIRepoLoader"
Cohesion: 0.10
Nodes (22): _import_ucimlrepo(), list_available_uci_datasets(), list_available_uci_datasets_async(), _parse_catalog_output(), Any, DataFrame, UCI ML Repository dataset loader via the ``ucimlrepo`` package., Fetch a dataset from the UCI ML Repository by ID or name. (+14 more)

### Community 47 - "safe_http.py"
Cohesion: 0.15
Nodes (30): _attempt_download_to_path(), _attempt_download_to_path_async(), _attempt_fetch(), _attempt_fetch_async(), _check_advertised_size(), _check_response_headers(), _normalize_content_type(), Path (+22 more)

### Community 48 - "SQLiteConnector"
Cohesion: 0.10
Nodes (16): Connection, Path, T, Reusable SQLite connector with safe defaults for local metadata storage., Open SQLite connections with consistent PRAGMAs and atomic write helpers., Yield a configured connection for read operations., Run a read-only callback using a configured connection., Run a write callback in an atomic transaction with lock retries. (+8 more)

### Community 49 - "test_registry.py"
Cohesion: 0.08
Nodes (29): ModelNotFoundError, PromotionError, Exception, Custom exceptions for the model registry layer., Raised when a registered model cannot be found., Raised when a promotion action cannot be completed., Base exception for registry failures., RegistryError (+21 more)

### Community 50 - "AppJobType"
Cohesion: 0.10
Nodes (18): AppJobType, Local app job categories stored outside MLflow., _build_footer(), _build_llm_prompt(), generate_llm_description(), generate_template_description(), _generic_template(), _job_icon() (+10 more)

### Community 51 - "validate_dataset"
Cohesion: 0.10
Nodes (28): record_validation_job(), Path, Produce validation artifacts: JSON summary, Markdown report., Write validation artifacts to disk and return the bundle., _render_markdown(), write_artifacts(), Data validation layer for AutoTabML Studio., CheckSeverity (+20 more)

### Community 52 - "safe_error_message"
Cohesion: 0.14
Nodes (23): _prediction_task_type_input(), Streamlit page for local-first prediction workflows., Render the default saved-model browser. Advanced sources (manual path, MLflow)…, Render manual-path, MLflow run, and MLflow registry options inside a collapsed…, Render per-column inputs and return the row payload dict., _render_advanced_model_sources(), _render_artifacts(), _render_batch_panel() (+15 more)

### Community 53 - "ExperimentWorkflowService"
Cohesion: 0.16
Nodes (14): ExperimentFormValues, ExperimentWorkflowService, Page-facing workflow service for the Train & Tune (PyCaret) page., User-entered form values from the experiment page., Verdict for a tuned-vs-baseline metric comparison., Encapsulate Train & Tune page orchestration., TuningInterpretation, _form_values() (+6 more)

### Community 54 - "test_hardening_smoke.py"
Cohesion: 0.09
Nodes (29): Benchmarking foundation for AutoTabML Studio., _build_metrics(), _build_run_name(), _log_artifacts(), Path, Backward-compatible benchmark tracker entrypoint., BenchmarkArtifactBundle, BenchmarkResultBundle (+21 more)

### Community 55 - "run_app_rules"
Cohesion: 0.28
Nodes (27): _check_allowed_categories(), _check_constant_columns(), _check_dtype_summary(), _check_duplicate_column_names(), _check_duplicate_rows(), _check_fully_null_columns(), _check_id_columns(), _check_leakage_heuristics() (+19 more)

### Community 56 - "BaseTracker"
Cohesion: 0.09
Nodes (19): BaseArtifacts, BaseTracker, Any, Logger, Path, Shared MLflow tracking lifecycle for modeling bundles., Log params, metrics, and artifacts for one modeling bundle., Return True if the subclass-specific MLflow boundary is available. (+11 more)

### Community 57 - "FlamlConfig"
Cohesion: 0.15
Nodes (6): FlamlConfig, Top-level FLAML AutoML configuration., _FakeAutoML, _make_service(), Minimal mock of flaml.AutoML for unit tests., TestFlamlService

### Community 58 - "list_registered_models"
Cohesion: 0.19
Nodes (21): list_registered_models(), Return all registered models from the MLflow model registry. Uses paginated…, _read_registry_cache(), _write_registry_cache(), _clear_cache(), _CountingClient, _FakeRegisteredModel, _FakeVersion (+13 more)

### Community 59 - "CheckpointResolver"
Cohesion: 0.13
Nodes (20): CheckpointResolver, CheckpointSpec, ModelDownloadRequiredError, Any, Path, RuntimeError, Pinned Hugging Face checkpoint resolution with explicit download consent., Immutable model repository identity. (+12 more)

### Community 60 - "log_exception"
Cohesion: 0.10
Nodes (35): AutoTabMLError, log_and_wrap(), log_exception(), Any, BaseException, Exception, Logger, Opt-in umbrella base class for application-level domain errors. (+27 more)

### Community 61 - "TabFMConfig"
Cohesion: 0.15
Nodes (17): BaseModel, Configuration for a local TabFM holdout evaluation., TabFMConfig, Forecast one or more independent regular time series with TimesFM 2.5., TimesFMService, ModelFactory, _cached_resolver(), _Classifier (+9 more)

### Community 62 - "observability/__init__.py"
Cohesion: 0.14
Nodes (22): bind_context(), clear_context(), correlation_scope(), current_context(), _empty_context_default(), new_correlation_id(), Any, Correlation context for log/metric/trace records. A small wrapper around… (+14 more)

### Community 63 - "InMemoryMetricsBackend"
Cohesion: 0.10
Nodes (17): get_metrics_backend(), InMemoryMetricsBackend, MetricsBackend, NoopMetricsBackend, Protocol, Install ``backend`` as the active sink and return the previous one., Return the currently installed metrics backend., Strategy interface implemented by metric exporters. (+9 more)

### Community 64 - "registry.py"
Cohesion: 0.15
Nodes (15): build_streamlit_navigation(), default_page_label(), get_nav_sections(), get_page_by_label(), PageSpec, Central registry for Streamlit page navigation and rendering., Declarative Streamlit page registration., Return pages grouped by section, preserving section order. Returns a list of… (+7 more)

### Community 65 - "safe_fetch_async"
Cohesion: 0.14
Nodes (19): BaseException, RuntimeError, Async counterpart of :func:`safe_fetch` with identical retry semantics., Fetch many URLs concurrently with bounded parallelism. Per-URL guards (SSRF,…, Async counterpart of :func:`safe_download_to_path`., safe_download_to_path_async(), safe_fetch_async(), safe_fetch_many_async() (+11 more)

### Community 66 - "SafeFetchPolicy"
Cohesion: 0.11
Nodes (24): Fetch ``url`` using the SSRF-resistant, bounded HTTP client. Retries are…, Convenience wrapper returning decoded text., Raised when a response declares a disallowed Content-Type., Async convenience wrapper returning decoded text., Tunable knobs for ``safe_fetch``., safe_fetch(), safe_fetch_text(), safe_fetch_text_async() (+16 more)

### Community 67 - "RunHistoryItem"
Cohesion: 0.13
Nodes (24): Path, Artifact generation for comparison bundles., Write comparison artifacts and return their paths., _render_markdown(), write_comparison_artifacts(), _check_comparability(), _compute_config_differences(), _compute_metric_deltas() (+16 more)

### Community 68 - "test_tracking.py"
Cohesion: 0.12
Nodes (23): cmd_history_list(), List MLflow runs from the history center., Enum, str, Filtering and sorting helpers for run history queries., Fields available for sorting run history., Ascending or descending sort., RunSortField (+15 more)

### Community 69 - "HistoryService"
Cohesion: 0.25
Nodes (9): HistoryService, High-level service for querying and inspecting run history., _FakeMLflowExperiment, _FakeMLflowRun, _patch_mlflow_query(), Patch mlflow_query module functions for testing., Regression tests for run-ID prefix resolution (HistoryService.resolve_run_id)., TestHistoryService (+1 more)

### Community 70 - "test_ingestion_utils.py"
Cohesion: 0.13
Nodes (18): compute_content_hash(), compute_schema_hash(), detect_file_extension(), extract_dataset_metadata(), Any, DataFrame, Metadata extraction and deterministic hashing for ingested datasets., Convert values to a stable, JSON-serializable representation. (+10 more)

### Community 71 - "ProfilingResultSummary"
Cohesion: 0.14
Nodes (13): ProfilingResultSummary, Quick-access summary extracted from a profiling run., Any, DataFrame, Path, Suppress known non-actionable third-party warnings during profiling., Profiling service backed by ydata-profiling., _suppress_profiling_runtime_noise() (+5 more)

### Community 72 - "validate_public_release_metadata"
Cohesion: 0.15
Nodes (18): check_public_release_metadata(), _collect_contacts(), _has_license_metadata(), load_project_metadata(), main(), _normalized(), Any, Path (+10 more)

### Community 73 - "history_page.py"
Cohesion: 0.08
Nodes (33): Streamlit page for job and dataset run history., Render the MLflow run description panel for a job., Persist the generated description into job metadata., Optional collapsible section to browse raw MLflow runs., render_history_page(), _render_job_description(), _render_mlflow_section(), _save_description_to_job() (+25 more)

### Community 74 - "ExcelLoader"
Cohesion: 0.19
Nodes (10): ExcelLoader, Any, DataFrame, Path, Load local or remote Excel workbooks., KaggleLoader, Any, DataFrame (+2 more)

### Community 75 - "experiment_page.py"
Cohesion: 0.07
Nodes (44): MLflowTrackingMode, How MLflow tracking should be handled for experiments., _format_pycaret_import_error(), is_pycaret_available(), _probe_pycaret_import_error(), Exception, pycaret_install_guidance(), PyCaret setup helpers and experiment construction. (+36 more)

### Community 76 - "metrics.py"
Cohesion: 0.10
Nodes (20): Counter, Gauge, Histogram, _merge_labels(), _Metric, Any, Pluggable metrics façade. Three primitive types are exposed – :class:`Counter`,…, Shared base – holds a metric name and forwards to the active backend. (+12 more)

### Community 77 - "update-graph.cjs"
Cohesion: 0.08
Nodes (18): assignedIds, changed, configLayer, dataLayer, docsLayer, fs, graph, graphPath (+10 more)

### Community 78 - "compare_page.py"
Cohesion: 0.23
Nodes (14): _find_model_column(), _find_score_column(), _leaderboard_rows_to_df(), _load_leaderboard(), DataFrame, Streamlit page for comparing algorithm performance on a dataset., Try to load a leaderboard from the job's artifact paths., Convert a list of leaderboard row dicts to a clean DataFrame. (+6 more)

### Community 79 - "BatchRunRecord"
Cohesion: 0.13
Nodes (8): BatchRunRecord, Tracks an overall batch execution run., Connection, T, BatchRunRepository, Connection, Row, CRUD operations for batch runs and their items.

### Community 80 - "ComparisonService"
Cohesion: 0.26
Nodes (6): ComparisonService, Produce structured comparisons between two runs., _make_run(), Path, TestComparisonArtifacts, TestComparisonService

### Community 81 - "capture_screenshots.py"
Cohesion: 0.18
Nodes (22): Page, capture(), choose_streamlit_option(), main(), navigate_to_page(), Path, Automated screenshot capture for AutoTabML Studio using Playwright. Launches a…, Navigate to Profiling and run it. (+14 more)

### Community 82 - "verify_optional_deps.py"
Cohesion: 0.37
Nodes (13): is_lazypredict_available(), Return True if lazypredict is importable., main(), _make_iris_df(), DataFrame, _result(), _section(), verify_gx() (+5 more)

### Community 83 - "test_observability.py"
Cohesion: 0.13
Nodes (22): configure_observability_logging(), CorrelationFilter, install_correlation_filter(), JsonFormatter, Logger, Structured (JSON) logging with correlation injection. The implementation is…, Attach :class:`CorrelationFilter` to ``logger`` (root by default). Idempotent:…, Configure the root logger for structured/observable output. Parameters… (+14 more)

### Community 84 - "PredictionRequest"
Cohesion: 0.05
Nodes (29): BasePredictionService, ABC, Abstract prediction service contract., Return discoverable local saved models., Load a normalized model for prediction., LocalTabFMContextLoader, MLflowModelLoader, ModelLoader (+21 more)

### Community 85 - "LocalArtifactManager"
Cohesion: 0.16
Nodes (8): LocalArtifactManager, DataFrame, Path, Create, write, and conservatively clean local workspace artifacts., _artifact_settings_for(), Path, Tests for centralized local artifact lifecycle management., TestLocalArtifactManager

### Community 86 - "autorun.py"
Cohesion: 0.20
Nodes (15): AutoRunConfig, AutoRunMode, AutoRunPlan, AutoRunResult, BaseModel, Enum, str, Guided, engine-aware Auto Run planning and execution. (+7 more)

### Community 87 - "._make_args"
Cohesion: 0.10
Nodes (9): Regression: experiment-tune/evaluate/save must catch setup errors cleanly., Direct functional tests for cmd_validate., Direct functional tests for cmd_history_list., Direct functional tests for cmd_profile., TestCmdBenchmark, TestCmdHistoryList, TestCmdProfile, TestCmdValidate (+1 more)

### Community 88 - "Path"
Cohesion: 0.17
Nodes (7): _build_input_spec(), Build an ingestion input spec from a CLI dataset locator., Path, Happy-path coverage for experiment-tune, experiment-evaluate, experiment-save., TestBuildInputSpec, TestExperimentCommandsHappyPath, TestCLIUCILocator

### Community 89 - "Path"
Cohesion: 0.16
Nodes (7): Persist settings to ~/.autotabml/settings.json (secrets excluded)., save_settings(), Save non-secret settings to disk., Path, As of v0.2.0 the project builds with hatchling; the profiling extra no longer…, TestPackagingMetadata, TestSettingsPersistence

### Community 90 - "RunHistoryFilter"
Cohesion: 0.15
Nodes (11): build_mlflow_filter_string(), Declarative filter for run history queries., Build an MLflow-compatible filter string from a RunHistoryFilter., RunHistoryFilter, Fetch extended detail for a single run., Resolve experiment name(s) to ids, building a name lookup map., Apply filters that cannot be expressed in MLflow filter syntax., List runs with optional filtering and sorting. (+3 more)

### Community 91 - "test_product_improvements.py"
Cohesion: 0.33
Nodes (8): plan_auto_run(), DataFrame, suggest_targets(), integration, Path, test_auto_run_plan_requires_confirmation_ready_target(), test_auto_run_saves_reloads_and_predicts(), test_background_job_cancel_is_persisted()

### Community 92 - "_FakeExperimentBase"
Cohesion: 0.10
Nodes (5): classification_df(), _FakeExperimentBase, DataFrame, fixture, regression_df()

### Community 93 - "_cli_error"
Cohesion: 0.17
Nodes (21): _cli_error(), cmd_benchmark(), cmd_deploy_export(), cmd_drift_check(), cmd_explain(), cmd_profile(), cmd_tabfm_run(), cmd_timesfm_forecast() (+13 more)

### Community 94 - "drift.py"
Cohesion: 0.22
Nodes (17): build_drift_baseline(), _categorical_proportions(), compare_drift(), DriftLevel, DriftReport, FeatureBaseline, FeatureDrift, _level() (+9 more)

### Community 95 - "tracking/errors.py"
Cohesion: 0.21
Nodes (12): ComparisonError, ExperimentNotFoundError, Exception, Custom exceptions for the tracking and history layer., Raised when MLflow tracking is not available or configured., Raised when a requested run cannot be found., Raised when a requested MLflow experiment cannot be found., Raised when a comparison cannot be completed. (+4 more)

### Community 96 - "CSVLoader"
Cohesion: 0.13
Nodes (17): CSVLoader, Any, DataFrame, Path, Load local or remote delimited tabular files into pandas., ParseFailureError, Raised when a source is reachable but could not be parsed., get_loader() (+9 more)

### Community 97 - "redact_key_in_text"
Cohesion: 0.17
Nodes (7): LogRecord, Replace anything that looks like an API key or credential in free text., redact_key_in_text(), Tests for security / masking helpers., TestLoggingRedaction, TestRedactKeyInText, TestSafeErrorMessage

### Community 99 - "explain_global"
Cohesion: 0.24
Nodes (15): explain_global(), explain_prediction(), FeatureContribution, ModelExplanation, Any, BaseModel, DataFrame, ndarray (+7 more)

### Community 100 - "TimesFMConfig"
Cohesion: 0.21
Nodes (18): _backtest_group(), _backtest_metrics(), _build_model(), _forecast_group(), _output_frame(), _prepare_series(), Any, BaseModel (+10 more)

### Community 101 - "load_dataset_async"
Cohesion: 0.26
Nodes (8): load_dataset_async(), Async public entry point for full dataset loading., asyncio, mock, MonkeyPatch, Path, TestAsyncIngestionFactory, TestAsyncUCIHelpers

### Community 102 - "foundation_models_page.py"
Cohesion: 0.18
Nodes (18): log_foundation_run(), Any, Path, Minimal MLflow summary logging for foundation-model runs., Log aggregate configuration, metrics, and the non-sensitive summary artifact., _dependency_status(), _download_artifacts(), cache_resource (+10 more)

### Community 103 - "SavedLocalModelRecord"
Cohesion: 0.20
Nodes (8): BaseModel, Local saved-model metadata kept outside MLflow registry state., SavedLocalModelRecord, Connection, Path, Row, CRUD operations for saved local model rows., SavedModelRepository

### Community 104 - "test_e2e_local_smoke.py"
Cohesion: 0.12
Nodes (12): is_ydata_available(), Return True if ydata-profiling is importable., Real end-to-end local smoke test — no mocks for core paths. Unlike…, Verify the startup MLflow URI validation catches bad config., Verify each optional dep has a clean availability check., Verify the real startup path creates all expected resources., Exercise the real pipeline — no mocks for any service code., _sqlite_uri() (+4 more)

### Community 105 - "migrations.py"
Cohesion: 0.31
Nodes (13): apply_migrations(), _detect_legacy_version(), _ensure_version_table(), _get_applied_versions(), Migration, Connection, Incremental SQLite schema migrations for the local metadata store., Create the version table and apply any pending schema migrations. (+5 more)

### Community 106 - "ProjectRecord"
Cohesion: 0.21
Nodes (7): ProjectRecord, Local project metadata for workspace navigation convenience., ProjectRepository, Connection, Row, CRUD operations for project rows., Upsert using an existing connection (used during initial migration).

### Community 107 - "configure_logging"
Cohesion: 0.24
Nodes (8): configure_logging(), _configure_noisy_dependency_loggers(), Logging configuration for AutoTabML Studio. Thin compatibility shim over…, Reduce non-actionable third-party log noise in normal app and batch runs., Configure structured logging to stderr (12-factor compliant). Honors the…, Plain-text formatter that scrubs obvious secrets from the message., _RedactingTextFormatter, test_configure_logging_suppresses_non_actionable_dependency_noise()

### Community 108 - "cli.py"
Cohesion: 0.18
Nodes (18): cmd_auto_run(), cmd_batch_history(), cmd_batch_show(), cmd_job_cancel(), cmd_job_list(), cmd_job_status(), CLI entry points for AutoTabML Studio. Usage: python -m app.cli --version…, List batch run history from the local database. (+10 more)

### Community 109 - "test_list_cached_registered_models_uses_cache_until_invalidated"
Cohesion: 0.53
Nodes (6): MonkeyPatch, Path, _settings_for(), test_get_prediction_service_reuses_resource_instance(), test_list_cached_registered_models_uses_cache_until_invalidated(), test_load_dataset_for_ui_uses_cache_until_invalidated()

### Community 110 - "build_provenance"
Cohesion: 0.28
Nodes (14): build_provenance(), _engine_packages(), _git_dirty(), _git_value(), _json_safe(), _package_version(), Any, Path (+6 more)

### Community 112 - "model_testing_page.py"
Cohesion: 0.16
Nodes (16): Streamlit page for model testing – evaluate a trained model on real-world data., Show evaluation metrics when ground-truth labels are available., _render_evaluation_metrics(), render_model_testing_page(), Unified Predictions page — combines Score New Data and Test & Evaluate., Delegate to the existing prediction page, skipping the outer title., Delegate to the existing Model Testing page, skipping the outer title., _render_evaluate_tab() (+8 more)

### Community 113 - "ArtifactKind"
Cohesion: 0.18
Nodes (15): ArtifactKind, str, Supported local artifact directory kinds., BenchmarkArtifactsWriter, Path, Artifact generation for benchmark runs., Shared-path artifact writer for benchmark result bundles., Write benchmark artifacts to disk and return their paths. (+7 more)

### Community 114 - "RunHistorySort"
Cohesion: 0.29
Nodes (7): BaseModel, Sort specification for run history queries., RunHistorySort, _build_order_by(), Translate a :class:`RunHistorySort` into MLflow ``order_by`` clauses. Only…, Validates that history sorting is correct and not partial-dataset., TestHistoryServiceSorting

### Community 115 - "AutoTabML Studio Social Preview"
Cohesion: 0.17
Nodes (12): AutoTabML Studio Social Preview, Benchmarking, CLI workflows, Dashboard, Local-first tabular ML workbench, MLflow-backed history, Model Registry, Portfolio-ready local ML tooling (+4 more)

### Community 116 - "profile_dataset"
Cohesion: 0.19
Nodes (14): profile_dataset(), DataFrame, Path, Convenience function: profile a DataFrame and optionally write artifacts., DataFrame, parametrize, Path, Optional integration checks for heavy dependency imports. These tests are… (+6 more)

### Community 117 - "build-batches-10-18.cjs"
Cohesion: 0.26
Nodes (11): complexity(), fileSummary(), fs, make(), memberSummary(), path, special, tagsFor() (+3 more)

### Community 118 - "log_ui_exception"
Cohesion: 0.21
Nodes (15): flaml_install_guidance(), Return a user-facing installation hint for environments without FLAML., _build_flaml_run_key(), Streamlit page for the FLAML AutoML workflow., render_flaml_automl_page(), _render_flaml_results(), get_flaml_automl_service(), Return a cached FLAML AutoML service for the current UI settings. (+7 more)

### Community 119 - "test_dataset_workspace.py"
Cohesion: 0.21
Nodes (9): Return a stable, unique session dataset label., resolve_session_dataset_name(), _make_loaded_dataset(), Path, Tests for shared dataset workspace helpers., Tests for the unified render_dataset_header helper., TestActiveDatasetSelection, TestRenderDatasetHeader (+1 more)

### Community 120 - "Code of Conduct"
Cohesion: 0.18
Nodes (11): Community interest, Code of Conduct, Enforcement actions, Giving and accepting feedback gracefully, Harassment-free participation, Maintainer enforcement responsibility, Positive environment, Project spaces (+3 more)

### Community 121 - "AutoML Modeling"
Cohesion: 0.18
Nodes (11): AutoML Modeling, Background Jobs, Data Quality, Dataset Ingestion, Deployment Bundle, External Providers, Metadata Store, ML Practitioner (+3 more)

### Community 122 - "build-batches-6-9.cjs"
Cohesion: 0.27
Nodes (10): comp(), emit(), fileSummary(), fs, make(), memberSummary(), path, purpose (+2 more)

### Community 123 - "manager.py"
Cohesion: 0.33
Nodes (4): Local artifact management utilities., datetime, Enum, Centralized local artifact path and lifecycle management. This manager only…

### Community 124 - "start_span"
Cohesion: 0.15
Nodes (13): _NoopSpan, Any, BaseException, Protocol, Optional OpenTelemetry tracing with a stdlib-only fallback. We deliberately do…, The minimal span surface used by AutoTabML Studio call sites., Open a span named ``name`` and yield a :class:`SpanLike`. When OpenTelemetry is…, SpanLike (+5 more)

### Community 125 - "Namespace"
Cohesion: 0.20
Nodes (16): _build_prediction_request_kwargs(), _build_prediction_service(), cmd_predict_batch(), cmd_predict_history(), cmd_predict_single(), cmd_uci_list(), _load_prediction_row_payload(), _print_local_model_trust_notice() (+8 more)

### Community 126 - "ua-assemble.cjs"
Cohesion: 0.20
Nodes (9): dir, fs, graph, ids, layers, output, path, scan (+1 more)

### Community 127 - "export_deployment_bundle"
Cohesion: 0.36
Nodes (8): _build_current_wheel(), DeploymentBundle, export_deployment_bundle(), BaseModel, Path, Portable prediction bundle generation., Create API, Docker, and command-line deployment assets for one trusted model., _sha256()

### Community 128 - "fetch_url_bytes"
Cohesion: 0.33
Nodes (4): fetch_url_bytes(), Fetch a remote resource and return bytes, content-type, and final URL., Confirm ingestion-facing wrappers translate guard errors into RemoteAccessError., TestIngestionIntegration

### Community 129 - "run_auto_run"
Cohesion: 0.16
Nodes (14): Any, Path, Run FLAML on a training split and evaluate on an untouched holdout., run_auto_run(), main(), Path, Subprocess entry point for persistent Auto Run jobs., _update() (+6 more)

### Community 130 - "mask_secret"
Cohesion: 0.39
Nodes (3): mask_secret(), Mask a secret string, keeping only *reveal* chars at each end visible.…, TestMaskSecret

### Community 131 - "Benchmark"
Cohesion: 0.25
Nodes (8): Benchmark, Benchmark configuration screen, Random seed: 42, Sample rows: 0 (full dataset), Target column: target, Task type: regression, Test size: 0.20, Top-k shortlist: 5

### Community 132 - "History view"
Cohesion: 0.29
Nodes (8): History navigation item, History view, Inspect run selector, Navigation sidebar, Run Detail, Run history table, Run metadata columns, Selected benchmark classification run

### Community 133 - "EDA / Profiling Dashboard"
Cohesion: 0.36
Nodes (8): Column Types: 11 Numeric, 0 Categorical, Data Quality Summary: 0.0% Missing, 0 Duplicates, Dataset Dimensions: 442 Rows, 11 Columns, Active Diabetes Dataset, Profiling Complete, Profiling Artifacts, EDA / Profiling Dashboard, Standard Report Mode

### Community 134 - "Settings view"
Cohesion: 0.29
Nodes (8): Accelerators, colab_mcp backend, Execution backend, Local backend, Application navigation, Promote version, Register model, Settings view

### Community 135 - "calculate_model_cost"
Cohesion: 0.19
Nodes (14): _section_model_cost_calculator(), calculate_model_cost(), get_model_pricing(), ModelCostEstimate, ModelPricing, Reference prices and token-cost estimates for hosted AI models., Standard per-million-token prices for a hosted model., Estimated input, output, and combined cost in US dollars. (+6 more)

### Community 137 - "Benchmark configuration form"
Cohesion: 0.29
Nodes (7): Benchmark configuration form, Experiment Lab, Experiment navigation item, Navigation sidebar, Run Benchmark button, Target column selector, Task type selector (regression)

### Community 138 - "Data Validation"
Cohesion: 0.29
Nodes (7): Active Dataset: diabetes, Data Validation, Dataset Dimensions: 442 Rows, 11 Columns, Data Validation Screen, Target Column: target, Validation Complete, Validation Results: 5 Passed, 0 Warnings, 0 Failed

### Community 140 - "TestCliOutputEncoding"
Cohesion: 0.29
Nodes (3): Regression tests for CLI output encoding on Windows cp1252., Ensure compare CLI output uses ASCII-safe arrows (->), not Unicode → (\u2192)., TestCliOutputEncoding

### Community 142 - "store.py"
Cohesion: 0.20
Nodes (7): DatasetRecord, Locally persisted dataset lineage record., DatasetRepository, Connection, Row, CRUD operations for dataset rows., SQLite-backed local app metadata store. This store is a thin facade over the…

### Community 143 - "cmd_experiment_run"
Cohesion: 0.27
Nodes (14): _build_pycaret_service(), cmd_experiment_evaluate(), cmd_experiment_run(), cmd_experiment_save(), cmd_experiment_tune(), _default_experiment_fold_strategy(), _default_experiment_tracking_mode(), _pycaret_native_tracking_mode() (+6 more)

### Community 144 - "MLflow run comparison view"
Cohesion: 0.33
Nodes (6): Artifact Availability, Saved comparison artifacts, Comparison status: not comparable, MLflow run comparison view, Compare view screenshot, Metadata verification warnings

### Community 145 - "Dashboard"
Cohesion: 0.33
Nodes (6): dashboard-overview, Active dataset diabetes, Dashboard, Application navigation, Recent Local Jobs, Workspace metrics

### Community 146 - "Loaded dataset 'diabetes'"
Cohesion: 0.33
Nodes (6): Benchmark, Loaded dataset 'diabetes', Experiment, Load Local Path, Profiling, Validation

### Community 147 - "Prediction Center Interface"
Cohesion: 0.33
Nodes (6): Discovered Local Model, Load Model Action, Model Loading Guidance, Prediction Center Interface, Prediction Navigation, Task Type Hint

### Community 148 - "e2e-demo-classifier"
Cohesion: 0.33
Nodes (6): e2e-demo-classifier, Model Version Promotion, Model Registry, Model Versions, Registered Models, Registry View Screenshot

### Community 149 - "IngestionError"
Cohesion: 0.19
Nodes (12): EmptyDatasetError, IngestionError, Exception, Raised when a loaded dataset has no usable rows or columns., Base exception for user-facing ingestion failures., normalize_duplicate_column_names(), normalize_to_pandas(), DataFrame (+4 more)

### Community 150 - "extract-ua-batches-10-13.cjs"
Cohesion: 0.33
Nodes (5): cp, fs, path, projectRoot, uaDir

### Community 151 - "process-ua-batches-19-26.cjs"
Cohesion: 0.33
Nodes (5): cp, fs, path, projectRoot, uaDir

### Community 152 - "notebook_page.py"
Cohesion: 0.22
Nodes (13): _generate_notebook_for_job(), Path, Notebook page – browse auto-generated notebooks per dataset / job. Each…, Generate and return notebook path for a job record., Render a simple preview of notebook cells., Execute *coro* from synchronous Streamlit code., Show notebooks grouped by dataset, one per job., _render_colab_mcp_notebook() (+5 more)

### Community 153 - "recorders.py"
Cohesion: 0.22
Nodes (12): Profiling artifact generation helpers., ProfilingArtifactBundle, BaseModel, Paths and metadata for profiling artifacts produced., _enrich_metadata_with_description(), ensure_dataset_record(), Any, Helpers that map app workflows into local metadata-store records. (+4 more)

### Community 154 - "ua-fingerprint-input.cjs"
Cohesion: 0.40
Nodes (4): fs, output, path, scan

### Community 155 - "_is_blocked_ip"
Cohesion: 0.50
Nodes (4): _is_blocked_ip(), Return a human-readable reason if ``addr`` is in a blocked range., IPv4Address, IPv6Address

### Community 156 - "BatchRunItemRecord"
Cohesion: 0.21
Nodes (11): BatchRunItemRecord, Tracks a single dataset within a batch run through validate/profile/benchmark., Path, When ALL datasets are already completed, batch record counts must still be set., Regression: targets that previously had wrong casing must match their UCI…, test_declared_target_from_item_prefers_item_id_suffix_for_resume_keys(), test_full_resume_updates_batch_record_counts(), test_known_case_sensitive_target_mappings() (+3 more)

### Community 157 - "Load Dataset Sources"
Cohesion: 0.50
Nodes (4): Dataset Intake page, Load Dataset Sources, Local dataset path: E:\Github\AutoTabML-Studio\datasets\sklearn\Diabetes\diabetes.csv, Local Path tab

### Community 158 - "Recent Prediction Jobs"
Cohesion: 0.50
Nodes (4): Model Source, Prediction Job Mode, Prediction Job Status, Recent Prediction Jobs

### Community 159 - "AutoTabML Studio v0.2.0"
Cohesion: 0.50
Nodes (4): 0.1.x to 0.2.0 migration guide, v0.2.0 release notes, AutoTabML Studio v0.2.0, 0.1.x to 0.2.0 upgrade summary

### Community 160 - "ProfilingSettings"
Cohesion: 0.19
Nodes (7): ProfilingSettings, Configuration for the automated profiling layer., profiling_install_guidance(), Return a user-facing installation hint for profiling dependencies., ImportError, TestArtifactPaths, TestProfilingConfig

### Community 161 - "FlamlAutoMLService"
Cohesion: 0.21
Nodes (7): _build_flaml_service(), FlamlAutoMLService, Any, Path, Build a leaderboard from FLAML's best_loss_per_estimator., FLAML-backed AutoML service., TestFlamlCLI

### Community 162 - "TestRuntimeState"
Cohesion: 0.17
Nodes (5): Switching to colab_mcp while provider=ollama should auto-reset provider., Switching backend when provider is valid for both should keep it., Setting the same backend value should not clear anything., backend_valid was removed — make sure it's not in to_dict., TestRuntimeState

### Community 163 - ".run"
Cohesion: 0.31
Nodes (8): _build_estimator(), Any, DataFrame, ndarray, Path, Series, _resolve_task_type(), _score_predictions()

### Community 167 - "Highlights"
Cohesion: 0.20
Nodes (9): Desktop and AI-provider experience, Evaluation and delivery, Guided Auto Run, Highlights, Performance and maintainability, Release Notes — v0.3.0, Security, Upgrade (+1 more)

### Community 168 - "ua-arch-write-layers.js"
Cohesion: 0.20
Nodes (9): assigned, definitions, duplicates, fs, known, layers, missing, results (+1 more)

### Community 169 - "profiling/__init__.py"
Cohesion: 0.31
Nodes (7): ProfilingError, ProfilingSetupError, Exception, Custom exceptions for the profiling layer., Raised when the profiling library cannot be initialized., Base exception for profiling failures., Automated profiling layer for AutoTabML Studio.

### Community 191 - "BackgroundJobService"
Cohesion: 0.32
Nodes (4): BackgroundJobService, DataFrame, Path, Submit and control one local training process at a time.

### Community 192 - "FlamlSearchConfig"
Cohesion: 0.33
Nodes (6): cmd_flaml_run(), cmd_flaml_save(), Run FLAML AutoML search from CLI., Run FLAML search and save the best model from CLI., FlamlSearchConfig, Configuration for the FLAML AutoML search.

### Community 193 - "ExperimentInfo"
Cohesion: 0.38
Nodes (7): get_experiment_by_name(), list_experiments(), _normalize_experiment(), Return all active MLflow experiments., Return one experiment by name or raise ExperimentNotFoundError., ExperimentInfo, Normalized MLflow experiment metadata.

### Community 194 - "evaluate_model"
Cohesion: 0.33
Nodes (6): evaluate_model(), Any, DataFrame, Series, Evaluate a fitted sklearn-compatible model on untouched data., test_evaluation_covers_regression()

### Community 195 - ".load_raw_dataframe"
Cohesion: 0.40
Nodes (4): Any, DataFrame, Return a raw DataFrame and source details before normalization., Async counterpart of :meth:`load_raw_dataframe`. Loaders with native async I/O…

### Community 196 - "get_page_registry"
Cohesion: 0.40
Nodes (3): get_page_registry(), Return the registered Streamlit pages in display order., TestFlamlPageRegistry

### Community 197 - ".prepare_session"
Cohesion: 0.40
Nodes (3): Any, Prepare any runtime context needed before a job runs., Execute a job payload and return results. Concrete implementations will be…

### Community 198 - "TestMetricSortDirection"
Cohesion: 0.50
Nodes (3): metric_sort_direction(), Return the expected ordering direction for a metric name., TestMetricSortDirection

### Community 200 - "TestRunSummary"
Cohesion: 0.50
Nodes (3): Return a compact one-line summary of a run., run_summary_line(), TestRunSummary

### Community 202 - "DataFrame"
Cohesion: 0.67
Nodes (4): classification_df(), DataFrame, fixture, regression_df()

### Community 203 - ".trash-1786706008/tmp/ua-arch-analyze.js"
Cohesion: 0.50
Nodes (3): fs, [inputPath, outputPath], path

### Community 205 - "_clear_registry_cache"
Cohesion: 0.67
Nodes (3): _clear_registry_cache(), fixture, Drop the in-process registry cache before and after every test.

## Knowledge Gaps
- **218 isolated node(s):** `fs`, `path`, `ua`, `purpose`, `fs` (+213 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **26 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `ExecutionBackend` connect `ExecutionBackend` to `BenchmarkConfig`, `test_flaml_automl.py`, `ProfilingMode`, `flaml/service.py`, `LLMProvider`, `test_benchmark.py`, `test_colab_mcp_backend.py`, `_DummyTracker`, `SavedModelMetadata`, `dataset_workspace.py`, `TestCliOutputEncoding`, `TestCmdRegistryGates`, `ValidationRuleConfig`, `notebook_page.py`, `AppSettings`, `ColabMCPExecutionBackend`, `ExperimentResultBundle`, `ProfilingSettings`, `FlamlAutoMLService`, `test_modeling_exception_handling.py`, `TestRuntimeState`, `ExperimentTaskType`, `pycaret/summary.py`, `test_cli.py`, `app/errors.py`, `test_hardening_smoke.py`, `FlamlConfig`, `TestFlamlSchemas`, `FlamlSearchConfig`, `get_page_registry`, `TestMetricSortDirection`, `TestFlamlPageHelpers`, `TestDefaultModels`, `experiment_page.py`, `._make_args`, `Path`, `Path`, `_FakeExperimentBase`, `TestFlamlRunType`, `_FakeMLflow`?**
  _High betweenness centrality (0.094) - this node is a cross-community bridge._
- **Why does `DatasetInputSpec` connect `DatasetInputSpec` to `ExecutionBackend`, `test_prediction.py`, `dataset_workspace.py`, `url_loader.py`, `test_uci_real_datasets.py`, `IngestionError`, `IngestionSourceType`, `PredictionService`, `loader.py`, `batch_uci_runner.py`, `ui_cache.py`, `app/errors.py`, `UCIRepoLoader`, `test_hardening_smoke.py`, `.load_raw_dataframe`, `test_ingestion_utils.py`, `ExcelLoader`, `PredictionRequest`, `Path`, `CSVLoader`, `load_dataset_async`, `test_e2e_local_smoke.py`, `cli.py`, `test_list_cached_registered_models_uses_cache_until_invalidated`, `test_dataset_workspace.py`?**
  _High betweenness centrality (0.083) - this node is a cross-community bridge._
- **Why does `AppSettings` connect `AppSettings` to `ExecutionBackend`, `test_flaml_automl.py`, `test_prediction.py`, `LLMProvider`, `test_colab_mcp_backend.py`, `TestCmdRegistryGates`, `dataset_workspace.py`, `TestCliOutputEncoding`, `SavedModelMetadata`, `PredictionHistoryStore`, `IngestionSourceType`, `BatchRunItemRecord`, `ColabMCPExecutionBackend`, `loader.py`, `storage/models.py`, `FlamlAutoMLService`, `ui_cache.py`, `test_cli.py`, `app/errors.py`, `SQLiteConnector`, `test_hardening_smoke.py`, `FlamlConfig`, `TestFlamlSchemas`, `get_page_registry`, `TestMetricSortDirection`, `TestFlamlPageHelpers`, `TestDefaultModels`, `experiment_page.py`, `compare_page.py`, `PredictionRequest`, `LocalArtifactManager`, `._make_args`, `Path`, `Path`, `test_product_improvements.py`, `TestFlamlRunType`, `test_e2e_local_smoke.py`, `cli.py`, `test_list_cached_registered_models_uses_cache_until_invalidated`, `model_testing_page.py`, `log_ui_exception`?**
  _High betweenness centrality (0.062) - this node is a cross-community bridge._
- **Are the 167 inferred relationships involving `ExecutionBackend` (e.g. with `BaseExecutionBackend` and `ColabMCPExecutionBackend`) actually correct?**
  _`ExecutionBackend` has 167 INFERRED edges - model-reasoned connections that need verification._
- **Are the 81 inferred relationships involving `AppSettings` (e.g. with `ExecutionBackend` and `LLMProvider`) actually correct?**
  _`AppSettings` has 81 INFERRED edges - model-reasoned connections that need verification._
- **Are the 53 inferred relationships involving `DatasetInputSpec` (e.g. with `BaseLoader` and `CSVLoader`) actually correct?**
  _`DatasetInputSpec` has 53 INFERRED edges - model-reasoned connections that need verification._
- **Are the 129 inferred relationships involving `WorkspaceMode` (e.g. with `AppSettings` and `ArtifactSettings`) actually correct?**
  _`WorkspaceMode` has 129 INFERRED edges - model-reasoned connections that need verification._