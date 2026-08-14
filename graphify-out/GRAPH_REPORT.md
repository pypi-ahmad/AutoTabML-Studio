# Graph Report - AutoTabML-Studio  (2026-08-14)

## Corpus Check
- 611 files · ~2,004,689 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 4641 nodes · 13154 edges · 197 communities (175 shown, 22 thin omitted)
- Extraction: 91% EXTRACTED · 9% INFERRED · 0% AMBIGUOUS · INFERRED: 1194 edges (avg confidence: 0.51)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `1aa12011`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- test_pycaret_experiment.py
- lazypredict_runner.py
- cli.py
- test_flaml_automl.py
- ProfilingConfig
- BenchmarkTaskType
- test_prediction.py
- LLMProvider
- ExecutionBackend
- test_colab_mcp_backend.py
- DatasetInputSpec
- SavedModelMetadata
- dataset_workspace.py
- benchmark_page.py
- url_loader.py
- PredictionWorkflowService
- mlflow_query.py
- registry_page.py
- ValidationRuleConfig
- generate_job_notebook
- test_uci_real_datasets.py
- JobRepository
- PredictionHistoryStore
- IngestionSourceType
- pyrightconfig.json
- AppSettings
- validation/schemas.py
- RegistryService
- ColabMCPExecutionBackend
- ExperimentResultBundle
- PredictionService
- loader.py
- AppMetadataStore
- storage/models.py
- test_modeling_architecture.py
- batch_uci_runner.py
- gather_with_concurrency
- models_page.py
- pycaret/service.py
- pycaret/summary.py
- ui_cache.py
- Architecture Guide
- test_cli.py
- ._setup
- settings_page.py
- cuda_summary
- uci_loader.py
- safe_http.py
- SQLiteConnector
- registry_service.py
- AppJobType
- validate_dataset
- prediction_page.py
- scorer.py
- BenchmarkConfig
- run_app_rules
- .log_bundle
- FlamlConfig
- list_registered_models
- CheckpointResolver
- log_exception
- TabFMConfig
- correlation_scope
- observability/__init__.py
- registry.py
- safe_fetch_async
- SafeFetchPolicy
- RunHistoryItem
- RunSortField
- HistoryService
- test_ingestion_utils.py
- YDataProfilingService
- validate_public_release_metadata
- ExperimentWorkflowService
- CSVLoader
- experiment_page.py
- Histogram
- update-graph.cjs
- compare_page.py
- BatchRunItemRecord
- ComparisonService
- capture_screenshots.py
- verify_optional_deps.py
- test_observability.py
- PredictionRequest
- LocalArtifactManager
- autorun.py
- Path
- ProfilingMode
- Path
- RunHistoryFilter
- test_product_improvements.py
- BenchmarkSavedModelMetadata
- registry/errors.py
- drift.py
- tracking/errors.py
- test_tracking.py
- redact_key_in_text
- TestSSRFHostBlocking
- explain_global
- TimesFMConfig
- load_dataset_async
- foundation_models_page.py
- SavedLocalModelRecord
- is_ydata_available
- migrations.py
- ProjectRecord
- install_correlation_filter
- MLflowExperimentTracker
- .ua/tmp/ua-arch-write-layers.js
- build_provenance
- _FakeMLflow
- extract_summary
- KaggleLoader
- RunHistorySort
- AutoTabML Studio Social Preview
- test_integration_optional_deps.py
- build-batches-10-18.cjs
- test_modeling_exception_handling.py
- ExperimentSetupSummary
- Code of Conduct
- AutoML Modeling
- build-batches-6-9.cjs
- ArtifactKind
- start_span
- safe_download_to_path_async
- ua-assemble.cjs
- export_deployment_bundle
- test_safe_http.py
- is_flaml_available
- mask_secret
- Benchmark
- History view
- EDA / Profiling Dashboard
- Settings view
- calculate_model_cost
- Counter
- Benchmark configuration form
- Data Validation
- compare_runner.py
- TestCliOutputEncoding
- TestLabelMaps
- store.py
- TestStartupColabMCPDiagnostics
- MLflow run comparison view
- Dashboard
- Loaded dataset 'diabetes'
- Prediction Center Interface
- e2e-demo-classifier
- IngestionError
- extract-ua-batches-10-13.cjs
- process-ua-batches-19-26.cjs
- get_or_init_state
- JobRecord
- ua-fingerprint-input.cjs
- _is_blocked_ip
- default_compare_metric_for_task
- Load Dataset Sources
- Recent Prediction Jobs
- AutoTabML Studio v0.2.0
- test_profiling.py
- .load_raw_dataframe
- TestRuntimeState
- .ua/tmp/ua-arch-analyze.js
- .trash-1786689789/tmp/ua-arch-analyze.js
- app/__init__.py
- modeling/__init__.py
- Highlights
- ua-arch-write-layers.js
- ydata_runner.py
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
- BackgroundJobService
- TestRunSummary
- .trash-1786706008/tmp/ua-arch-analyze.js
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
- `main()` --calls--> `Counter`  [INFERRED]
  .ua/tmp/ua-tour-analyze.py → app/observability/metrics.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **** —  [EXTRACTED 1.00]
- **docs_assets_screenshots_dashboard_overview_workspace_overview** — docs_assets_screenshots_dashboard_overview_dashboard, docs_assets_screenshots_dashboard_overview_active_dataset, docs_assets_screenshots_dashboard_overview_workspace_metrics, docs_assets_screenshots_dashboard_overview_recent_local_jobs [EXTRACTED 0.98]
- **docs_assets_screenshots_dataset_intake_active_dataset_workflows** — docs_assets_screenshots_dataset_intake_diabetes_dataset, docs_assets_screenshots_dataset_intake_validation, docs_assets_screenshots_dataset_intake_profiling, docs_assets_screenshots_dataset_intake_benchmark, docs_assets_screenshots_dataset_intake_experiment [EXTRACTED 1.00]
- **Quality Assurance Delivery Surface** — github_workflows_ci_workflow, github_workflows_security_workflow, github_workflows_release_readiness_workflow, docs_developer_guide_test_strategy [INFERRED 0.85]
- **Local-First Workbench Documentation** — readme_local_first_workbench, docs_architecture_local_first, docs_operations_local_runtime_layout [INFERRED 0.85]
- **Primary AutoTabML Journey** — docs_autotabml_studio_architecture_studio_workbench, docs_autotabml_studio_architecture_dataset_ingestion, docs_autotabml_studio_architecture_automl_modeling, docs_autotabml_studio_architecture_prediction_service, docs_autotabml_studio_architecture_trusted_artifacts [EXTRACTED 1.00]

## Communities (197 total, 22 thin omitted)

### Community 0 - "test_pycaret_experiment.py"
Cohesion: 0.05
Nodes (41): ExperimentArtifactsWriter, Path, Write experiment artifacts to disk and return their paths., Shared-path artifact writer for experiment result bundles., write_experiment_artifacts(), CustomMetricSpec, ExperimentArtifactBundle, ExperimentCompareConfig (+33 more)

### Community 1 - "lazypredict_runner.py"
Cohesion: 0.08
Nodes (34): BenchmarkConfigurationError, BenchmarkDependencyError, BenchmarkError, BenchmarkExecutionError, BenchmarkTrackingError, Exception, Custom exceptions for the benchmark layer., Raised when a benchmark configuration is invalid. (+26 more)

### Community 2 - "cli.py"
Cohesion: 0.07
Nodes (100): _add_prediction_model_source_args(), _build_prediction_request_kwargs(), _build_prediction_service(), _build_pycaret_service(), _cli_error(), cmd_auto_run(), cmd_batch_history(), cmd_batch_show() (+92 more)

### Community 3 - "test_flaml_automl.py"
Cohesion: 0.07
Nodes (35): _build_flaml_service(), FlamlSettings, Configuration for FLAML AutoML workflows., FlamlAutoMLService, metric_sort_direction(), Path, Return the expected ordering direction for a metric name., FLAML-backed AutoML service. (+27 more)

### Community 4 - "ProfilingConfig"
Cohesion: 0.24
Nodes (10): ProfilingConfig, User-facing configuration for a profiling run., maybe_sample(), DataFrame, Automatic mode selection and DataFrame sampling for profiling., Choose profiling mode based on dataset size and config thresholds., Return (possibly sampled df, was_sampled, sample_size_used). Sampling is…, select_profiling_mode() (+2 more)

### Community 5 - "BenchmarkTaskType"
Cohesion: 0.05
Nodes (75): BenchmarkTargetError, Raised when the selected benchmark target is invalid., Raised when the requested benchmark task type is unsupported., UnsupportedBenchmarkTaskError, BenchmarkTaskType, Supported benchmark tasks., benchmark_reliability_warnings(), choose_stratify_target() (+67 more)

### Community 6 - "test_prediction.py"
Cohesion: 0.06
Nodes (63): DataFrame, Path, Artifact generation for prediction jobs., Persist prediction artifacts and return their paths., _render_markdown(), write_prediction_artifacts(), Prediction service facade., Run batch prediction. (+55 more)

### Community 7 - "LLMProvider"
Cohesion: 0.04
Nodes (49): LLMProvider, Enum, Enumerations for AutoTabML Studio configuration., Supported LLM providers., AnthropicProvider, Any, Anthropic provider — model discovery and text generation via the official SDK.…, BaseProvider (+41 more)

### Community 8 - "ExecutionBackend"
Cohesion: 0.05
Nodes (78): ExecutionBackend, str, Execution backends – where ML jobs actually run., First-class workspace modes., WorkspaceMode, ArtifactSettings, BenchmarkSettings, DatabaseSettings (+70 more)

### Community 9 - "test_colab_mcp_backend.py"
Cohesion: 0.09
Nodes (21): BaseExecutionBackend, Any, Abstract execution backend interface., Common interface for all execution backends., Check that the backend is reachable / properly configured., Prepare any runtime context needed before a job runs., Execute a job payload and return results. Concrete implementations will be…, Colab MCP execution backend – connects to Google Colab via MCP. This backend… (+13 more)

### Community 10 - "DatasetInputSpec"
Cohesion: 0.07
Nodes (31): load_dataset(), Public entry point for full dataset loading., DatasetInputSpec, model_validator, Canonical input contract for all supported ingestion paths., Return the human-readable locator for lineage and logging., Validate source-specific input requirements., Fetch a dataset from the UCI ML Repository by ID or name. (+23 more)

### Community 11 - "SavedModelMetadata"
Cohesion: 0.11
Nodes (18): Stable saved-model metadata contract for future prediction flows., SavedModelMetadata, LocalPyCaretModelLoader, Path, Load local saved PyCaret model artifacts., compute_sha256(), Return the SHA256 digest for a file on disk., Write a SHA256 checksum sidecar for an artifact and return the sidecar path. (+10 more)

### Community 12 - "dataset_workspace.py"
Cohesion: 0.09
Nodes (38): build_local_path_input_spec(), build_url_input_spec(), _clear_dataset_results(), clear_loaded_datasets(), _dataset_identity_key(), get_active_dataset_name(), get_loaded_datasets(), infer_local_source_type() (+30 more)

### Community 13 - "benchmark_page.py"
Cohesion: 0.08
Nodes (33): AutoTabML Studio – Streamlit entry point., default_ranking_metric_for_task(), Streamlit page for LazyPredict benchmark execution and results., Resolve a LazyPredict model name to its sklearn-compatible estimator class., Render controls to retrain and save the best model from benchmark results., Return the default ranking metric to prefill in the UI., render_benchmark_page(), _render_result_bundle() (+25 more)

### Community 14 - "url_loader.py"
Cohesion: 0.07
Nodes (47): Any, DataFrame, Path, CSV and delimiter-aware text loader., ParseFailureError, Custom exceptions for dataset ingestion., Raised when a source type or locator is unsupported., Raised when a remote resource cannot be reached or inspected safely. (+39 more)

### Community 15 - "PredictionWorkflowService"
Cohesion: 0.08
Nodes (20): Page-layer services that keep Streamlit pages focused on rendering., ModelTestingEvaluation, ModelTestingRunResult, ModelTestingSelection, PredictionExecutionConfig, PredictionWorkflowService, Any, DataFrame (+12 more)

### Community 16 - "mlflow_query.py"
Cohesion: 0.07
Nodes (56): create_model_version(), create_registered_model(), delete_model_alias(), delete_model_version_tag(), _extract_dataset_name(), _extract_tags(), _get_client(), get_experiment_by_name() (+48 more)

### Community 17 - "registry_page.py"
Cohesion: 0.12
Nodes (24): Streamlit profiling / EDA page – run and view dataset profiling results., Render profiling summary cards., render_profiling_page(), _render_summary(), Streamlit page for the model registry., Render the manual model registration form., _render_register_section(), render_registry_page() (+16 more)

### Community 18 - "ValidationRuleConfig"
Cohesion: 0.08
Nodes (28): Configuration for the data validation layer., ValidationSettings, User-facing configuration for a validation run., ValidationRuleConfig, constant_df(), duplicate_rows_df(), empty_df(), good_df() (+20 more)

### Community 19 - "generate_job_notebook"
Cohesion: 0.09
Nodes (38): _benchmark_cells(), _experiment_cells(), _flaml_cells(), generate_job_notebook(), _json_literal(), _json_string(), _markdown_cell(), NotebookGenerationError (+30 more)

### Community 20 - "test_uci_real_datasets.py"
Cohesion: 0.11
Nodes (26): auto_mpg_dataset(), _fetch_uci(), heart_disease_dataset(), iris_dataset(), DataFrame, fixture, Path, Real-dataset integration tests using UCI ML Repository. These tests fetch… (+18 more)

### Community 21 - "JobRepository"
Cohesion: 0.20
Nodes (4): JobRepository, Connection, CRUD operations for job rows., Path

### Community 22 - "PredictionHistoryStore"
Cohesion: 0.09
Nodes (18): BasePredictionService, ABC, Abstract prediction service contract., Return discoverable local saved models., Load a normalized model for prediction., Run one-row prediction., PredictionHistoryStore, Path (+10 more)

### Community 23 - "IngestionSourceType"
Cohesion: 0.07
Nodes (35): Base abstractions shared by all ingestion loaders., In-memory pandas DataFrame loader., Tabular ingestion entry points for AutoTabML Studio., Kaggle dataset loader with explicit, realistic boundaries., DatasetMetadata, LoadedDataset, BaseModel, DataFrame (+27 more)

### Community 24 - "pyrightconfig.json"
Cohesion: 0.04
Nodes (45): exclude, ignore, include, pythonPlatform, pythonVersion, reportArgumentType, reportAttributeAccessIssue, reportCallIssue (+37 more)

### Community 25 - "AppSettings"
Cohesion: 0.05
Nodes (22): AppSettings, model_validator, setter, Top-level application configuration with safe defaults., Backward-compatible alias for the canonical MLflow settings section., Return the verified stable fallback model id for the given (or current)…, default_settings(), fixture (+14 more)

### Community 26 - "validation/schemas.py"
Cohesion: 0.07
Nodes (39): Exception, Custom exceptions for the validation layer., Raised when the validation infrastructure cannot be initialized., Raised when a validation rule configuration is invalid., Base exception for validation failures., RuleConfigError, ValidationError, ValidationSetupError (+31 more)

### Community 27 - "RegistryService"
Cohesion: 0.14
Nodes (15): High-level service for MLflow model registry operations., RegistryService, _safe_version_sort_key(), PromotionRequest, Request to promote a model version., _FakeModelVersion, _FakeRegisteredModel, _patch_registry() (+7 more)

### Community 28 - "ColabMCPExecutionBackend"
Cohesion: 0.09
Nodes (15): ColabMCPExecutionBackend, Any, Execute a job by calling Colab MCP tools. ``job_payload`` must include a…, Call ``open_colab_browser_connection`` to link to a Colab notebook., Return the currently available MCP tool names., Tear down the MCP session and server subprocess., Execution backend that delegates work to a Google Colab runtime via MCP., Check that ``uvx`` is installed and the MCP SDK is importable. (+7 more)

### Community 29 - "ExperimentResultBundle"
Cohesion: 0.12
Nodes (20): PyCaretExecutionError, Raised when a PyCaret operation fails., PyCaret-backed experiment lab for AutoTabML Studio., _build_metrics(), _build_run_name(), ExperimentResultBundle, ExperimentRuntimeState, ModelSelectionSpec (+12 more)

### Community 30 - "PredictionService"
Cohesion: 0.12
Nodes (24): DriftBaseline, PredictionService, Path, Production-style local-first prediction service., PredictionValidationError, Raised when prediction-time input validation fails., How prediction-time schema differences should be handled., SchemaValidationMode (+16 more)

### Community 31 - "loader.py"
Cohesion: 0.06
Nodes (62): load_saved_benchmark_model_metadata_file(), Trusted benchmark model discovery and loading helpers., Parse a benchmark saved-model metadata file, returning None when invalid., ModelDiscoveryError, Raised when a model source cannot be discovered or resolved cleanly., Model-loading wrappers for prediction flows., AvailableModelReference, Normalized model reference shown in discovery and selection UIs. (+54 more)

### Community 32 - "AppMetadataStore"
Cohesion: 0.12
Nodes (17): AppMetadataStore, Repository facade for local workspace metadata. The store composes the per-…, _dataset(), _job(), _now(), datetime, fixture, Path (+9 more)

### Community 33 - "storage/models.py"
Cohesion: 0.10
Nodes (17): BatchItemStatus, BatchRunStatus, Enum, str, Typed records for the local app metadata database., Overall batch run status., Individual dataset run status within a batch., BaseRepository (+9 more)

### Community 34 - "test_modeling_architecture.py"
Cohesion: 0.04
Nodes (50): BaseService, BaseTracker, get_mlflow_module(), is_mlflow_available(), mlflow_exception_types(), BaseException, Logger, Shared base classes for modeling services, trackers, and artifact writers. (+42 more)

### Community 35 - "batch_uci_runner.py"
Cohesion: 0.11
Nodes (25): build_200_dataset_list(), main(), Batch UCI dataset runner – runs validate → profile → benchmark for 200 NEW UCI…, Combine the new UCI datasets + extra re-framed datasets to get 200., _build_resume_state(), _declared_target_from_item(), _detect_target_and_task(), main() (+17 more)

### Community 36 - "gather_with_concurrency"
Cohesion: 0.10
Nodes (25): gather_with_concurrency(), Any, BaseException, T, Small async concurrency helpers used across the app. These helpers exist so…, Run awaitables concurrently with at most ``limit`` running at a time. ``limit``…, Run a blocking ``func`` over many argument batches concurrently in threads.…, to_thread_many() (+17 more)

### Community 37 - "models_page.py"
Cohesion: 0.09
Nodes (36): Streamlit page for model testing – evaluate a trained model on real-world data., Show evaluation metrics when ground-truth labels are available., _render_evaluation_metrics(), render_model_testing_page(), Path, Streamlit page for browsing all saved models with details., Render an expander card for a PyCaret experiment model., Render an expander card for a benchmark-saved model. (+28 more)

### Community 38 - "pycaret/service.py"
Cohesion: 0.04
Nodes (85): Cross-cutting error-handling utilities. This module provides: *…, Exception, PyCaretDependencyError, PyCaretExperimentError, PyCaretTargetError, PyCaretTrackingError, Custom exceptions for the PyCaret experiment layer., Raised when the optional PyCaret dependency is unavailable. (+77 more)

### Community 39 - "pycaret/summary.py"
Cohesion: 0.08
Nodes (41): Artifact generation for experiment runs., _render_markdown(), PyCaretConfigurationError, Raised when experiment configuration is invalid., add_custom_metric(), list_available_metrics(), Metric catalog inspection and safe custom metric registration., Return normalized metric rows from the active experiment. (+33 more)

### Community 40 - "ui_cache.py"
Cohesion: 0.10
Nodes (32): _section_cache_controls(), _get_experiment_workflow_service_resource(), _get_flaml_automl_service_resource(), get_history_service(), _get_history_service_resource(), _get_metadata_store_resource(), get_prediction_service(), _get_prediction_service_resource() (+24 more)

### Community 41 - "Architecture Guide"
Cohesion: 0.06
Nodes (35): Changelog, AutoTabML Container Service, Docker Compose Stack, Architecture Guide, Architecture Diagram HTML, Local-First Architecture, Reproducible Runs, Security Model (+27 more)

### Community 42 - "test_cli.py"
Cohesion: 0.06
Nodes (35): initialize_local_runtime(), BaseModel, Startup checks and local-runtime initialization for AutoTabML Studio., Report missing prerequisites for the Colab MCP backend., One startup diagnostic item., Initialization result for local app runtime resources., Prepare conservative local runtime resources and collect actionable diagnostics., StartupIssue (+27 more)

### Community 43 - "._setup"
Cohesion: 0.15
Nodes (11): _classification_csv(), fixture, parametrize, Path, Verify the startup MLflow URI validation catches bad config., Write a small but realistic classification dataset (no network)., Verify the real startup path creates all expected resources., Common setup: settings, metadata store, loaded dataset. (+3 more)

### Community 44 - "settings_page.py"
Cohesion: 0.09
Nodes (30): _find_uvx(), Return the path to ``uvx`` if it is installed, else *None*., _fetch_models(), Settings / Runtime configuration page for Streamlit., Lightweight GPU status for the Essentials tab — no controls., Simple on/off for run descriptions on the Essentials tab., Run an async coroutine from sync Streamlit code. Streamlit may already have a…, Build a provider instance and fetch models, updating state. (+22 more)

### Community 45 - "cuda_summary"
Cohesion: 0.11
Nodes (20): cuda_device_name(), cuda_summary(), _driver_probe(), is_cuda_available(), CUDA / GPU detection utilities for AutoTabML Studio., Lightweight probe for the NVIDIA driver library without importing torch. Loads…, Return True when a CUDA-capable GPU is reachable from the current Python…, Return the name of the first CUDA device, or None. (+12 more)

### Community 46 - "uci_loader.py"
Cohesion: 0.23
Nodes (12): _import_ucimlrepo(), list_available_uci_datasets(), list_available_uci_datasets_async(), _parse_catalog_output(), Any, DataFrame, UCI ML Repository dataset loader via the ``ucimlrepo`` package., Return structured UCI catalog rows by parsing ``list_available_datasets``… (+4 more)

### Community 47 - "safe_http.py"
Cohesion: 0.18
Nodes (25): _attempt_download_to_path(), _attempt_download_to_path_async(), _attempt_fetch(), _attempt_fetch_async(), _check_advertised_size(), _check_response_headers(), _normalize_content_type(), Bounded, SSRF-resistant HTTP fetch utilities. Used by ingestion code paths that… (+17 more)

### Community 48 - "SQLiteConnector"
Cohesion: 0.11
Nodes (14): Connection, Path, T, Open SQLite connections with consistent PRAGMAs and atomic write helpers., Yield a configured connection for read operations., Run a read-only callback using a configured connection., Run a write callback in an atomic transaction with lock retries., SQLiteConnector (+6 more)

### Community 49 - "registry_service.py"
Cohesion: 0.09
Nodes (21): Model registry and promotion workflows for AutoTabML Studio., Model registry service – list, inspect, register, and tag models., List all registered models., Get a specific model version., Get the model version currently assigned to an alias., Register a model and create its first version. Creates the registered model if…, PromotionAction, PromotionResult (+13 more)

### Community 50 - "AppJobType"
Cohesion: 0.10
Nodes (17): AppJobType, Local app job categories stored outside MLflow., _build_footer(), _build_llm_prompt(), generate_llm_description(), generate_template_description(), _generic_template(), _job_icon() (+9 more)

### Community 51 - "validate_dataset"
Cohesion: 0.10
Nodes (26): Path, Produce validation artifacts: JSON summary, Markdown report., Write validation artifacts to disk and return the bundle., _render_markdown(), write_artifacts(), is_gx_available(), Return True if great_expectations is importable., BaseModel (+18 more)

### Community 52 - "prediction_page.py"
Cohesion: 0.11
Nodes (29): _prediction_task_type_input(), Streamlit page for local-first prediction workflows., Render the default saved-model browser. Advanced sources (manual path, MLflow)…, Render manual-path, MLflow run, and MLflow registry options inside a collapsed…, Render per-column inputs and return the row payload dict., _render_advanced_model_sources(), _render_artifacts(), _render_batch_panel() (+21 more)

### Community 53 - "scorer.py"
Cohesion: 0.14
Nodes (24): PredictionError, PredictionHistoryError, PredictionScoringError, Exception, Prediction-layer errors for inference workflows., Raised when a model artifact fails trust-boundary validation., Raised when a loaded model cannot score the provided data., Raised when prediction history cannot be persisted or queried. (+16 more)

### Community 54 - "BenchmarkConfig"
Cohesion: 0.06
Nodes (57): BenchmarkArtifactsWriter, Path, Artifact generation for benchmark runs., Shared-path artifact writer for benchmark result bundles., Write benchmark artifacts to disk and return their paths., _render_markdown(), write_benchmark_artifacts(), _write_score_chart() (+49 more)

### Community 55 - "run_app_rules"
Cohesion: 0.28
Nodes (27): _check_allowed_categories(), _check_constant_columns(), _check_dtype_summary(), _check_duplicate_column_names(), _check_duplicate_rows(), _check_fully_null_columns(), _check_id_columns(), _check_leakage_heuristics() (+19 more)

### Community 56 - ".log_bundle"
Cohesion: 0.09
Nodes (16): BaseArtifacts, Any, Path, Log params, metrics, and artifacts for one modeling bundle., Return True if the subclass-specific MLflow boundary is available., Return the subclass-specific MLflow module handle., Return the operation label used for structured error logging., Return the MLflow run name for the bundle. (+8 more)

### Community 57 - "FlamlConfig"
Cohesion: 0.10
Nodes (13): FlamlConfig, FlamlSearchConfig, Configuration for the FLAML AutoML search., Top-level FLAML AutoML configuration., classification_df(), _FakeAutoML, _make_service(), DataFrame (+5 more)

### Community 58 - "list_registered_models"
Cohesion: 0.21
Nodes (19): list_registered_models(), Return all registered models from the MLflow model registry. Uses paginated…, _clear_cache(), _CountingClient, _FakeRegisteredModel, _FakeVersion, _patch_client(), fixture (+11 more)

### Community 59 - "CheckpointResolver"
Cohesion: 0.10
Nodes (28): CheckpointResolver, CheckpointSpec, ModelDownloadRequiredError, Any, Path, RuntimeError, Pinned Hugging Face checkpoint resolution with explicit download consent., Immutable model repository identity. (+20 more)

### Community 60 - "log_exception"
Cohesion: 0.13
Nodes (20): AutoTabMLError, log_and_wrap(), log_exception(), Any, BaseException, Exception, Logger, Opt-in umbrella base class for application-level domain errors. (+12 more)

### Community 61 - "TabFMConfig"
Cohesion: 0.19
Nodes (14): BaseModel, Configuration for a local TabFM holdout evaluation., TabFMConfig, _cached_resolver(), _Classifier, Path, _Regressor, test_checkpoint_resolver_requires_confirmation_before_download() (+6 more)

### Community 62 - "correlation_scope"
Cohesion: 0.15
Nodes (19): bind_context(), correlation_scope(), current_context(), _empty_context_default(), new_correlation_id(), Any, Correlation context for log/metric/trace records. A small wrapper around…, Return a *copy* of the current correlation context mapping. (+11 more)

### Community 63 - "observability/__init__.py"
Cohesion: 0.11
Nodes (16): Production-grade observability primitives for AutoTabML Studio. This package…, get_metrics_backend(), InMemoryMetricsBackend, MetricsBackend, NoopMetricsBackend, Protocol, Pluggable metrics façade. Three primitive types are exposed – :class:`Counter`,…, Install ``backend`` as the active sink and return the previous one. (+8 more)

### Community 64 - "registry.py"
Cohesion: 0.13
Nodes (17): build_streamlit_navigation(), default_page_label(), get_nav_sections(), get_page_by_label(), get_page_registry(), PageSpec, Central registry for Streamlit page navigation and rendering., Declarative Streamlit page registration. (+9 more)

### Community 65 - "safe_fetch_async"
Cohesion: 0.15
Nodes (16): BaseException, Async counterpart of :func:`safe_fetch` with identical retry semantics., Fetch many URLs concurrently with bounded parallelism. Per-URL guards (SSRF,…, safe_fetch_async(), safe_fetch_many_async(), public_dns(), asyncio, fixture (+8 more)

### Community 66 - "SafeFetchPolicy"
Cohesion: 0.16
Nodes (14): ValueError, Fetch ``url`` using the SSRF-resistant, bounded HTTP client. Retries are…, Convenience wrapper returning decoded text., Fetch up to ``sample_size`` bytes for content sniffing. Implemented on top of…, Async counterpart of :func:`safe_stream_sample`., Tunable knobs for ``safe_fetch``., safe_fetch(), safe_fetch_text() (+6 more)

### Community 67 - "RunHistoryItem"
Cohesion: 0.12
Nodes (27): _check_comparability(), _compute_config_differences(), _compute_metric_deltas(), Side-by-side run comparison service., Compare two runs and return a structured bundle with deltas and warnings., Run history service – list, filter, and inspect MLflow runs., _sort_runs(), Run history, comparison, and MLflow query layer for AutoTabML Studio. (+19 more)

### Community 68 - "RunSortField"
Cohesion: 0.20
Nodes (11): Enum, str, Filtering and sorting helpers for run history queries., Fields available for sorting run history., Ascending or descending sort., RunSortField, SortDirection, Return True if MLflow cannot natively order by ``field``. (+3 more)

### Community 69 - "HistoryService"
Cohesion: 0.21
Nodes (10): HistoryService, Fetch extended detail for a single run., High-level service for querying and inspecting run history., _FakeMLflowExperiment, _FakeMLflowRun, _patch_mlflow_query(), Patch mlflow_query module functions for testing., Regression tests for run-ID prefix resolution (HistoryService.resolve_run_id). (+2 more)

### Community 70 - "test_ingestion_utils.py"
Cohesion: 0.13
Nodes (18): compute_content_hash(), compute_schema_hash(), detect_file_extension(), extract_dataset_metadata(), Any, DataFrame, Metadata extraction and deterministic hashing for ingested datasets., Convert values to a stable, JSON-serializable representation. (+10 more)

### Community 71 - "YDataProfilingService"
Cohesion: 0.17
Nodes (10): Any, DataFrame, Path, Suppress known non-actionable third-party warnings during profiling., Profiling service backed by ydata-profiling., _suppress_profiling_runtime_noise(), YDataProfilingService, skipif (+2 more)

### Community 72 - "validate_public_release_metadata"
Cohesion: 0.15
Nodes (18): check_public_release_metadata(), _collect_contacts(), _has_license_metadata(), load_project_metadata(), main(), _normalized(), Any, Path (+10 more)

### Community 73 - "ExperimentWorkflowService"
Cohesion: 0.07
Nodes (33): ExperimentWorkflowService, Verdict for a tuned-vs-baseline metric comparison., Encapsulate Train & Tune page orchestration., TuningInterpretation, dataframe_to_safe_csv(), Any, DataFrame, Safe CSV export helpers. Excel and similar spreadsheet tools can evaluate cells… (+25 more)

### Community 74 - "CSVLoader"
Cohesion: 0.09
Nodes (28): BaseLoader, Common load lifecycle for all dataset sources., CSVLoader, Load local or remote delimited tabular files into pandas., DataFrameLoader, Accept a pandas DataFrame without mutating caller-owned data., ExcelLoader, Any (+20 more)

### Community 75 - "experiment_page.py"
Cohesion: 0.08
Nodes (41): flaml_install_guidance(), Return a user-facing installation hint for environments without FLAML., _build_schema_frame(), DataFrame, Dedicated Streamlit page for dataset intake and active selection., render_dataset_intake_page(), _render_uci_source_details(), go_to_page() (+33 more)

### Community 76 - "Histogram"
Cohesion: 0.12
Nodes (14): Gauge, Histogram, _merge_labels(), _Metric, Any, Shared base – holds a metric name and forwards to the active backend., Point-in-time numeric value., Distribution of observed values (e.g. durations in seconds). (+6 more)

### Community 77 - "update-graph.cjs"
Cohesion: 0.08
Nodes (18): assignedIds, changed, configLayer, dataLayer, docsLayer, fs, graph, graphPath (+10 more)

### Community 78 - "compare_page.py"
Cohesion: 0.27
Nodes (12): _find_model_column(), _find_score_column(), _leaderboard_rows_to_df(), _load_leaderboard(), DataFrame, Streamlit page for comparing algorithm performance on a dataset., Try to load a leaderboard from the job's artifact paths., Convert a list of leaderboard row dicts to a clean DataFrame. (+4 more)

### Community 79 - "BatchRunItemRecord"
Cohesion: 0.08
Nodes (19): BatchRunItemRecord, BatchRunRecord, Tracks an overall batch execution run., Tracks a single dataset within a batch run through validate/profile/benchmark., Connection, T, BatchRunRepository, Connection (+11 more)

### Community 80 - "ComparisonService"
Cohesion: 0.26
Nodes (6): ComparisonService, Produce structured comparisons between two runs., _make_run(), Path, TestComparisonArtifacts, TestComparisonService

### Community 81 - "capture_screenshots.py"
Cohesion: 0.18
Nodes (22): Page, capture(), choose_streamlit_option(), main(), navigate_to_page(), Path, Automated screenshot capture for AutoTabML Studio using Playwright. Launches a…, Navigate to Profiling and run it. (+14 more)

### Community 82 - "verify_optional_deps.py"
Cohesion: 0.42
Nodes (13): is_lazypredict_available(), Return True if lazypredict is importable., main(), _make_iris_df(), DataFrame, _result(), _section(), verify_gx() (+5 more)

### Community 83 - "test_observability.py"
Cohesion: 0.15
Nodes (21): clear_context(), Reset the correlation context to an empty mapping., configure_observability_logging(), JsonFormatter, Configure the root logger for structured/observable output. Parameters…, Render :class:`LogRecord` as a single-line JSON document. The output is stable…, Decorator that wraps a function call in :func:`start_span`., traced() (+13 more)

### Community 84 - "PredictionRequest"
Cohesion: 0.07
Nodes (27): ModelLoadError, Raised when a prediction model cannot be loaded., LocalTabFMContextLoader, MLflowModelLoader, ModelLoader, ABC, Any, Load MLflow-backed pyfunc models. (+19 more)

### Community 85 - "LocalArtifactManager"
Cohesion: 0.20
Nodes (5): LocalArtifactManager, DataFrame, Path, Create, write, and conservatively clean local workspace artifacts., ModelFactory

### Community 86 - "autorun.py"
Cohesion: 0.16
Nodes (16): AutoRunMode, AutoRunPlan, AutoRunResult, BaseModel, Enum, str, Guided, engine-aware Auto Run planning and execution., evaluate_model() (+8 more)

### Community 87 - "Path"
Cohesion: 0.08
Nodes (14): _build_input_spec(), Build an ingestion input spec from a CLI dataset locator., Path, Regression: experiment-tune/evaluate/save must catch setup errors cleanly., Happy-path coverage for experiment-tune, experiment-evaluate, experiment-save., Direct functional tests for cmd_history_list., Direct functional tests for cmd_profile., TestBuildInputSpec (+6 more)

### Community 88 - "ProfilingMode"
Cohesion: 0.13
Nodes (17): ProfilingMode, str, Profiling report modes., Profiling artifact generation helpers., Automated profiling layer for AutoTabML Studio., ProfilingArtifactBundle, ProfilingResultSummary, BaseModel (+9 more)

### Community 89 - "Path"
Cohesion: 0.16
Nodes (7): Persist settings to ~/.autotabml/settings.json (secrets excluded)., save_settings(), Save non-secret settings to disk., Path, As of v0.2.0 the project builds with hatchling; the profiling extra no longer…, TestPackagingMetadata, TestSettingsPersistence

### Community 90 - "RunHistoryFilter"
Cohesion: 0.20
Nodes (9): build_mlflow_filter_string(), Declarative filter for run history queries., Build an MLflow-compatible filter string from a RunHistoryFilter., RunHistoryFilter, Resolve experiment name(s) to ids, building a name lookup map., Apply filters that cannot be expressed in MLflow filter syntax., List runs with optional filtering and sorting., Resolve a possibly truncated run ID prefix to a full 32-char ID. If… (+1 more)

### Community 91 - "test_product_improvements.py"
Cohesion: 0.21
Nodes (15): AutoRunConfig, plan_auto_run(), Any, DataFrame, Path, Run FLAML on a training split and evaluate on an untouched holdout., run_auto_run(), suggest_targets() (+7 more)

### Community 92 - "BenchmarkSavedModelMetadata"
Cohesion: 0.16
Nodes (13): discover_saved_benchmark_models(), load_saved_benchmark_model(), Path, Discover trusted benchmark models from checksum-backed metadata sidecars., Load a trusted benchmark model from a skops artifact., BenchmarkSavedModelMetadata, Metadata sidecar for a benchmark-saved local model., Path (+5 more)

### Community 93 - "registry/errors.py"
Cohesion: 0.13
Nodes (15): ModelNotFoundError, PromotionError, Exception, Custom exceptions for the model registry layer., Raised when the MLflow model registry backend is not available., Raised when a registered model cannot be found., Raised when a model version cannot be found., Raised when a promotion action cannot be completed. (+7 more)

### Community 94 - "drift.py"
Cohesion: 0.22
Nodes (17): build_drift_baseline(), _categorical_proportions(), compare_drift(), DriftLevel, DriftReport, FeatureBaseline, FeatureDrift, _level() (+9 more)

### Community 95 - "tracking/errors.py"
Cohesion: 0.21
Nodes (12): ComparisonError, ExperimentNotFoundError, Exception, Custom exceptions for the tracking and history layer., Raised when MLflow tracking is not available or configured., Raised when a requested run cannot be found., Raised when a requested MLflow experiment cannot be found., Raised when a comparison cannot be completed. (+4 more)

### Community 96 - "test_tracking.py"
Cohesion: 0.18
Nodes (9): Path, Artifact generation for comparison bundles., Write comparison artifacts and return their paths., _render_markdown(), write_comparison_artifacts(), _extract_primary_metric(), _safe_run_status(), Tests for the tracking layer – history, comparison, and MLflow query wrappers. (+1 more)

### Community 97 - "redact_key_in_text"
Cohesion: 0.17
Nodes (7): LogRecord, Replace anything that looks like an API key or credential in free text., redact_key_in_text(), Tests for security / masking helpers., TestLoggingRedaction, TestRedactKeyInText, TestSafeErrorMessage

### Community 98 - "TestSSRFHostBlocking"
Cohesion: 0.16
Nodes (7): public_dns(), fixture, MonkeyPatch, parametrize, Force hostname resolution to a *public* IP regardless of the host string. This…, TestSchemeAndCredentialGuards, TestSSRFHostBlocking

### Community 99 - "explain_global"
Cohesion: 0.24
Nodes (15): explain_global(), explain_prediction(), FeatureContribution, ModelExplanation, Any, BaseModel, DataFrame, ndarray (+7 more)

### Community 100 - "TimesFMConfig"
Cohesion: 0.19
Nodes (20): _backtest_group(), _backtest_metrics(), _build_model(), _forecast_group(), _output_frame(), _prepare_series(), Any, BaseModel (+12 more)

### Community 101 - "load_dataset_async"
Cohesion: 0.27
Nodes (7): load_dataset_async(), Async public entry point for full dataset loading., asyncio, mock, MonkeyPatch, Path, TestAsyncIngestionFactory

### Community 102 - "foundation_models_page.py"
Cohesion: 0.18
Nodes (18): log_foundation_run(), Any, Path, Minimal MLflow summary logging for foundation-model runs., Log aggregate configuration, metrics, and the non-sensitive summary artifact., _dependency_status(), _download_artifacts(), cache_resource (+10 more)

### Community 103 - "SavedLocalModelRecord"
Cohesion: 0.20
Nodes (8): BaseModel, Local saved-model metadata kept outside MLflow registry state., SavedLocalModelRecord, Connection, Path, Row, CRUD operations for saved local model rows., SavedModelRepository

### Community 104 - "is_ydata_available"
Cohesion: 0.25
Nodes (4): is_ydata_available(), Return True if ydata-profiling is importable., Verify each optional dep has a clean availability check., TestOptionalDependencyProbes

### Community 105 - "migrations.py"
Cohesion: 0.31
Nodes (13): apply_migrations(), _detect_legacy_version(), _ensure_version_table(), _get_applied_versions(), Migration, Connection, Incremental SQLite schema migrations for the local metadata store., Create the version table and apply any pending schema migrations. (+5 more)

### Community 106 - "ProjectRecord"
Cohesion: 0.20
Nodes (7): ProjectRecord, Local project metadata for workspace navigation convenience., ProjectRepository, Connection, Row, CRUD operations for project rows., Upsert using an existing connection (used during initial migration).

### Community 107 - "install_correlation_filter"
Cohesion: 0.14
Nodes (14): configure_logging(), _configure_noisy_dependency_loggers(), Logging configuration for AutoTabML Studio. Thin compatibility shim over…, Reduce non-actionable third-party log noise in normal app and batch runs., Configure structured logging to stderr (12-factor compliant). Honors the…, CorrelationFilter, install_correlation_filter(), Logger (+6 more)

### Community 108 - "MLflowExperimentTracker"
Cohesion: 0.21
Nodes (8): _build_params(), _log_artifacts(), MLflowExperimentTracker, Any, Path, Lightweight explicit MLflow tracker for experiment runs., Backward-compatible experiment tracker entrypoint., _stringify_param()

### Community 109 - ".ua/tmp/ua-arch-write-layers.js"
Cohesion: 0.15
Nodes (12): assignments, definitions, duplicates, freshAssignments, fs, invented, known, layers (+4 more)

### Community 110 - "build_provenance"
Cohesion: 0.22
Nodes (17): build_provenance(), _engine_packages(), _git_dirty(), _git_value(), _json_safe(), _package_version(), ProvenanceManifest, Any (+9 more)

### Community 112 - "extract_summary"
Cohesion: 0.22
Nodes (8): extract_summary(), _get_high_cardinality_columns(), Any, DataFrame, Extract structured summary for profiling runs. Core quick-access metrics are…, Build a ProfilingResultSummary from a ydata-profiling report object. The HTML…, Identify columns with very high cardinality relative to row count., Test that summary extraction works using direct DataFrame analysis.

### Community 113 - "KaggleLoader"
Cohesion: 0.33
Nodes (5): KaggleLoader, Any, DataFrame, Path, Download a Kaggle dataset locally, then delegate to a concrete file loader.

### Community 114 - "RunHistorySort"
Cohesion: 0.29
Nodes (7): BaseModel, Sort specification for run history queries., RunHistorySort, _build_order_by(), Translate a :class:`RunHistorySort` into MLflow ``order_by`` clauses. Only…, Validates that history sorting is correct and not partial-dataset., TestHistoryServiceSorting

### Community 115 - "AutoTabML Studio Social Preview"
Cohesion: 0.17
Nodes (12): AutoTabML Studio Social Preview, Benchmarking, CLI workflows, Dashboard, Local-first tabular ML workbench, MLflow-backed history, Model Registry, Portfolio-ready local ML tooling (+4 more)

### Community 116 - "test_integration_optional_deps.py"
Cohesion: 0.24
Nodes (11): DataFrame, parametrize, Path, Optional integration checks for heavy dependency imports. These tests are…, _small_classification_df(), _sqlite_uri(), test_benchmark_executes_real_lazypredict_and_mlflow(), test_gx_validation_executes_real_expectations() (+3 more)

### Community 117 - "build-batches-10-18.cjs"
Cohesion: 0.26
Nodes (11): complexity(), fileSummary(), fs, make(), memberSummary(), path, special, tagsFor() (+3 more)

### Community 118 - "test_modeling_exception_handling.py"
Cohesion: 0.36
Nodes (7): _FailingRunContext, _fake_mlflow_module(), SimpleNamespace, test_benchmark_tracker_returns_warning_on_mlflow_failure(), test_flaml_tracker_preserves_existing_run_id_on_mlflow_failure(), test_lazypredict_service_wraps_split_failure(), test_pycaret_tracker_preserves_existing_run_id_on_mlflow_failure()

### Community 119 - "ExperimentSetupSummary"
Cohesion: 0.25
Nodes (6): ExperimentSetupSummary, _is_json_safe_metric_kwarg(), Any, field_validator, Return a serializable setup summary model., Serializable record of the normalized setup stage.

### Community 120 - "Code of Conduct"
Cohesion: 0.18
Nodes (11): Community interest, Code of Conduct, Enforcement actions, Giving and accepting feedback gracefully, Harassment-free participation, Maintainer enforcement responsibility, Positive environment, Project spaces (+3 more)

### Community 121 - "AutoML Modeling"
Cohesion: 0.18
Nodes (11): AutoML Modeling, Background Jobs, Data Quality, Dataset Ingestion, Deployment Bundle, External Providers, Metadata Store, ML Practitioner (+3 more)

### Community 122 - "build-batches-6-9.cjs"
Cohesion: 0.27
Nodes (10): comp(), emit(), fileSummary(), fs, make(), memberSummary(), path, purpose (+2 more)

### Community 123 - "ArtifactKind"
Cohesion: 0.18
Nodes (11): Local artifact management utilities., ArtifactKind, datetime, Enum, str, Centralized local artifact path and lifecycle management. This manager only…, Supported local artifact directory kinds., _artifact_settings_for() (+3 more)

### Community 124 - "start_span"
Cohesion: 0.15
Nodes (13): _NoopSpan, Any, BaseException, Protocol, Optional OpenTelemetry tracing with a stdlib-only fallback. We deliberately do…, The minimal span surface used by AutoTabML Studio call sites., Open a span named ``name`` and yield a :class:`SpanLike`. When OpenTelemetry is…, SpanLike (+5 more)

### Community 125 - "safe_download_to_path_async"
Cohesion: 0.28
Nodes (8): Path, RuntimeError, Successful streamed download metadata from :func:`safe_download_to_path`., Stream ``url`` into ``destination_path`` with the same guards as…, Async counterpart of :func:`safe_download_to_path`., safe_download_to_path(), safe_download_to_path_async(), SafeDownloadResult

### Community 126 - "ua-assemble.cjs"
Cohesion: 0.20
Nodes (9): dir, fs, graph, ids, layers, output, path, scan (+1 more)

### Community 127 - "export_deployment_bundle"
Cohesion: 0.36
Nodes (8): _build_current_wheel(), DeploymentBundle, export_deployment_bundle(), BaseModel, Path, Portable prediction bundle generation., Create API, Docker, and command-line deployment assets for one trusted model., _sha256()

### Community 128 - "test_safe_http.py"
Cohesion: 0.19
Nodes (10): Raised when a response declares a disallowed Content-Type., Async convenience wrapper returning decoded text., safe_fetch_text_async(), UnsafeContentTypeError, Security tests for the bounded HTTP fetch utility (SSRF guard)., Confirm ingestion-facing wrappers translate guard errors into RemoteAccessError., TestContentTypeGuards, TestDownloadToPath (+2 more)

### Community 129 - "is_flaml_available"
Cohesion: 0.40
Nodes (5): is_flaml_available(), _probe_flaml_import_error(), Exception, Return the import-time failure when FLAML is unusable., Return True when FLAML is importable.

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

### Community 136 - "Counter"
Cohesion: 0.25
Nodes (6): Counter, Monotonically increasing counter., test_counter_attaches_correlation_labels(), test_counter_increments_in_memory(), main(), main()

### Community 137 - "Benchmark configuration form"
Cohesion: 0.29
Nodes (7): Benchmark configuration form, Experiment Lab, Experiment navigation item, Navigation sidebar, Run Benchmark button, Target column selector, Task type selector (regression)

### Community 138 - "Data Validation"
Cohesion: 0.29
Nodes (7): Active Dataset: diabetes, Data Validation, Dataset Dimensions: 442 Rows, 11 Columns, Data Validation Screen, Target Column: target, Validation Complete, Validation Results: 5 Passed, 0 Warnings, 0 Failed

### Community 139 - "compare_runner.py"
Cohesion: 0.33
Nodes (6): create_model(), Any, compare_models and create_model wrappers., Execute compare_models with explicit, testable kwargs., Create one concrete model from a model id., run_compare_models()

### Community 140 - "TestCliOutputEncoding"
Cohesion: 0.29
Nodes (3): Regression tests for CLI output encoding on Windows cp1252., Ensure compare CLI output uses ASCII-safe arrows (->), not Unicode → (\u2192)., TestCliOutputEncoding

### Community 142 - "store.py"
Cohesion: 0.17
Nodes (8): DatasetRecord, Locally persisted dataset lineage record., DatasetRepository, Connection, Row, CRUD operations for dataset rows., Reusable SQLite connector with safe defaults for local metadata storage., SQLite-backed local app metadata store. This store is a thin facade over the…

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
Cohesion: 0.08
Nodes (21): Any, DataFrame, Return a raw DataFrame and source details before normalization., Async counterpart of :meth:`load_raw_dataframe`. Loaders with native async I/O…, Load, normalize, and enrich a dataset from the supplied input spec., Async counterpart of :meth:`load` for I/O-heavy loader implementations., Load only a preview slice where the underlying loader supports it., Async counterpart of :meth:`preview`. (+13 more)

### Community 150 - "extract-ua-batches-10-13.cjs"
Cohesion: 0.33
Nodes (5): cp, fs, path, projectRoot, uaDir

### Community 151 - "process-ua-batches-19-26.cjs"
Cohesion: 0.33
Nodes (5): cp, fs, path, projectRoot, uaDir

### Community 152 - "get_or_init_state"
Cohesion: 0.08
Nodes (38): Guided Auto Run page., _render_active_job(), render_autorun_page(), _detect_completed_steps(), _friendly_job_name(), _load_example_dataset(), Dashboard page – professional workspace home., Return the highest *completed* step number (0 = nothing done yet). Heuristic… (+30 more)

### Community 153 - "JobRecord"
Cohesion: 0.17
Nodes (21): main(), Path, Subprocess entry point for persistent Auto Run jobs., _update(), Local metadata storage exports., AppJobStatus, JobRecord, Coarse local execution status. (+13 more)

### Community 154 - "ua-fingerprint-input.cjs"
Cohesion: 0.40
Nodes (4): fs, output, path, scan

### Community 155 - "_is_blocked_ip"
Cohesion: 0.50
Nodes (4): _is_blocked_ip(), Return a human-readable reason if ``addr`` is in a blocked range., IPv4Address, IPv6Address

### Community 156 - "default_compare_metric_for_task"
Cohesion: 0.40
Nodes (4): default_compare_metric_for_task(), default_tune_metric_for_task(), Return the UI default compare metric for the chosen task., Return the UI default tune metric for the chosen task.

### Community 157 - "Load Dataset Sources"
Cohesion: 0.50
Nodes (4): Dataset Intake page, Load Dataset Sources, Local dataset path: E:\Github\AutoTabML-Studio\datasets\sklearn\Diabetes\diabetes.csv, Local Path tab

### Community 158 - "Recent Prediction Jobs"
Cohesion: 0.50
Nodes (4): Model Source, Prediction Job Mode, Prediction Job Status, Recent Prediction Jobs

### Community 159 - "AutoTabML Studio v0.2.0"
Cohesion: 0.50
Nodes (4): 0.1.x to 0.2.0 migration guide, v0.2.0 release notes, AutoTabML Studio v0.2.0, 0.1.x to 0.2.0 upgrade summary

### Community 160 - "test_profiling.py"
Cohesion: 0.15
Nodes (14): ProfilingSettings, Configuration for the automated profiling layer., profiling_install_guidance(), Return a user-facing installation hint for profiling dependencies., ImportError, large_df(), fixture, Tests for the profiling layer. GX and ydata-profiling are NOT required for… (+6 more)

### Community 162 - "TestRuntimeState"
Cohesion: 0.17
Nodes (5): Switching to colab_mcp while provider=ollama should auto-reset provider., Switching backend when provider is valid for both should keep it., Setting the same backend value should not clear anything., backend_valid was removed — make sure it's not in to_dict., TestRuntimeState

### Community 167 - "Highlights"
Cohesion: 0.20
Nodes (9): Desktop and AI-provider experience, Evaluation and delivery, Guided Auto Run, Highlights, Performance and maintainability, Release Notes — v0.3.0, Security, Upgrade (+1 more)

### Community 168 - "ua-arch-write-layers.js"
Cohesion: 0.20
Nodes (9): assigned, definitions, duplicates, fs, known, layers, missing, results (+1 more)

### Community 169 - "ydata_runner.py"
Cohesion: 0.31
Nodes (7): ProfilingError, ProfilingSetupError, Exception, Custom exceptions for the profiling layer., Raised when the profiling library cannot be initialized., Base exception for profiling failures., YData Profiling runner – wraps ydata-profiling behind a clean interface. All…

### Community 191 - "BackgroundJobService"
Cohesion: 0.32
Nodes (4): BackgroundJobService, DataFrame, Path, Submit and control one local training process at a time.

### Community 200 - "TestRunSummary"
Cohesion: 0.50
Nodes (3): Return a compact one-line summary of a run., run_summary_line(), TestRunSummary

### Community 203 - ".trash-1786706008/tmp/ua-arch-analyze.js"
Cohesion: 0.50
Nodes (3): fs, [inputPath, outputPath], path

### Community 205 - "_clear_registry_cache"
Cohesion: 0.67
Nodes (3): _clear_registry_cache(), fixture, Drop the in-process registry cache before and after every test.

## Knowledge Gaps
- **232 isolated node(s):** `fs`, `path`, `ua`, `purpose`, `fs` (+227 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **22 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `ExecutionBackend` connect `ExecutionBackend` to `test_pycaret_experiment.py`, `lazypredict_runner.py`, `test_flaml_automl.py`, `BenchmarkTaskType`, `LLMProvider`, `test_colab_mcp_backend.py`, `SavedModelMetadata`, `TestCliOutputEncoding`, `TestStartupColabMCPDiagnostics`, `ValidationRuleConfig`, `get_or_init_state`, `AppSettings`, `ColabMCPExecutionBackend`, `ExperimentResultBundle`, `test_profiling.py`, `test_modeling_architecture.py`, `TestRuntimeState`, `pycaret/service.py`, `pycaret/summary.py`, `test_cli.py`, `settings_page.py`, `BenchmarkConfig`, `FlamlConfig`, `Path`, `ProfilingMode`, `Path`, `BenchmarkSavedModelMetadata`, `_FakeMLflow`, `test_modeling_exception_handling.py`, `ExperimentSetupSummary`?**
  _High betweenness centrality (0.097) - this node is a cross-community bridge._
- **Why does `DatasetInputSpec` connect `DatasetInputSpec` to `test_pycaret_experiment.py`, `cli.py`, `test_prediction.py`, `dataset_workspace.py`, `url_loader.py`, `test_uci_real_datasets.py`, `IngestionError`, `PredictionHistoryStore`, `IngestionSourceType`, `get_or_init_state`, `PredictionService`, `loader.py`, `.load_raw_dataframe`, `batch_uci_runner.py`, `ui_cache.py`, `._setup`, `uci_loader.py`, `validate_dataset`, `BenchmarkConfig`, `test_ingestion_utils.py`, `CSVLoader`, `PredictionRequest`, `Path`, `load_dataset_async`, `KaggleLoader`?**
  _High betweenness centrality (0.096) - this node is a cross-community bridge._
- **Why does `AppSettings` connect `AppSettings` to `cli.py`, `test_flaml_automl.py`, `test_prediction.py`, `LLMProvider`, `ExecutionBackend`, `test_colab_mcp_backend.py`, `SavedModelMetadata`, `TestCliOutputEncoding`, `TestStartupColabMCPDiagnostics`, `PredictionHistoryStore`, `IngestionSourceType`, `get_or_init_state`, `JobRecord`, `ColabMCPExecutionBackend`, `models_page.py`, `ui_cache.py`, `test_cli.py`, `._setup`, `settings_page.py`, `SQLiteConnector`, `validate_dataset`, `BenchmarkConfig`, `FlamlConfig`, `RunHistoryItem`, `experiment_page.py`, `BatchRunItemRecord`, `PredictionRequest`, `Path`, `Path`, `test_product_improvements.py`, `is_ydata_available`, `ArtifactKind`?**
  _High betweenness centrality (0.061) - this node is a cross-community bridge._
- **Are the 167 inferred relationships involving `ExecutionBackend` (e.g. with `BaseExecutionBackend` and `ColabMCPExecutionBackend`) actually correct?**
  _`ExecutionBackend` has 167 INFERRED edges - model-reasoned connections that need verification._
- **Are the 81 inferred relationships involving `AppSettings` (e.g. with `ExecutionBackend` and `LLMProvider`) actually correct?**
  _`AppSettings` has 81 INFERRED edges - model-reasoned connections that need verification._
- **Are the 53 inferred relationships involving `DatasetInputSpec` (e.g. with `BaseLoader` and `CSVLoader`) actually correct?**
  _`DatasetInputSpec` has 53 INFERRED edges - model-reasoned connections that need verification._
- **Are the 129 inferred relationships involving `WorkspaceMode` (e.g. with `AppSettings` and `ArtifactSettings`) actually correct?**
  _`WorkspaceMode` has 129 INFERRED edges - model-reasoned connections that need verification._