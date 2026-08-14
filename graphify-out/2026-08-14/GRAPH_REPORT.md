# Graph Report - .  (2026-08-14)

## Corpus Check
- 281 files · ~230,103 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 4257 nodes · 12280 edges · 164 communities (149 shown, 15 thin omitted)
- Extraction: 91% EXTRACTED · 9% INFERRED · 0% AMBIGUOUS · INFERRED: 1163 edges (avg confidence: 0.51)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- App Settings
- Pycaret Service
- Prediction Base
- Safe Fetch Policy
- Benchmark Config
- Execution Backend
- Test Flaml Automl
- App Cli
- Dataset Workspace
- Benchmark Task Type
- Url Loader
- Test Hardening Smoke
- Test Ingestion Utils
- Ingestion Source Type
- Mlflow Query
- Benchmark Page
- Dataset Input Spec
- Saved Model Metadata
- Prediction Page
- Validation Rule Config
- Test Modeling Exception Handling
- Experiment Page
- Generate Job Notebook
- Flaml Auto Mlservice
- Prediction Workflow Service
- Get Or Init State
- Ui Labels
- Test Tracking
- Config Llmprovider
- Validate Dataset
- Experiment Result Bundle
- Prediction Loader
- Pyrightconfig Json
- Gather With Concurrency
- Log Exception
- Flaml Config
- Build Input Spec
- Test Prediction
- Validation Service
- Trusted Artifacts
- Settings Page
- Gx Runner
- Test Profiling
- Run History Item
- Registry Service
- Profiling Mode
- Batch Uci Runner
- Test Observability
- Registry Service
- Sqlite Connector
- Cuda Summary
- Model Source Type
- Gemini Provider
- App Metadata Store
- Generate Template Description
- Observability Init
- List Available Uci Datasets
- Dataframe To Safe Csv
- Install Correlation Filter
- Modeling Base
- History Service
- Run App Rules
- List Registered Models
- Ui Cache
- Ydata Runner
- Validate Public Release Metadata
- Base Repository
- Validation Settings
- App Main
- Batch Run Record
- Benchmark Result Bundle
- Profile Dataset
- Test Cli
- Capture Screenshots
- Test Storage Repositories
- Observability Histogram
- Comparison Service
- Local Artifact Manager
- Excel Loader
- Base Tracker
- Colab Mcpexecution Backend
- Test Colab Mcp Backend
- Test Packaging Metadata
- Htmltable Loader
- Start Span
- Registry Errors
- Base Artifacts
- Ydata Profiling Service
- Base Execution Backend
- Redact Key In Text
- Notebook Page
- Test Ui Cache
- Storage Migrations
- Batch Run Item Record
- Project Record
- Saved Local Model Record
- Dataset Record
- Tracking Errors
- Test Mlflow Normalization
- Auto Tab Ml Studio Social
- Test Runtime State
- Prepare Session
- Test Optional Dependency Probes
- Code Of Conduct
- Artifact Kind
- Build Provider
- Ollama Provider
- Test Colab Mcpreal Handshake
- Experiment Setup Summary
- Base Prediction Service
- Profiling Init
- Anthropic Provider
- Resolve Default Model
- Mask Secret
- Screenshots Benchmark
- History View
- Eda Profiling Dashboard
- Settings View
- Test Artifact Manager
- Dummy Tracker
- Resolve Task Type
- Compare Runner
- Auto Tab Ml Studio V0
- Benchmark Configuration Form
- Data Validation
- Test Cli Output Encoding
- Test Benchmark Executes Real Lazypredict
- Test Label Maps
- Build Backend
- Mlflow Run Comparison View
- Operations Guide
- Screenshots Dashboard
- Loaded Dataset Diabetes
- Prediction Center Interface
- Model Registry
- Colab Mcp Spike
- Test Startup Colab Mcpdiagnostics
- Get Allowed Providers
- Ci Workflow
- Tests Conftest
- Is Blocked Ip
- Auto Tab Ml Studio
- Load Dataset Sources
- Recent Prediction Jobs
- Public Dns
- Load Raw Dataframe
- Probe Flaml Import Error
- Clear Registry Cache
- App Init
- Modeling Init
- Run Benchmark Action
- Save Remaining Sklearn
- Demo Assets Documentation
- Auto Tab Ml Social Preview
- Documentation Index
- Dependabot Dependency Update Policy
- Autotabml Studio
- Auto Tab Ml Studio Readme

## God Nodes (most connected - your core abstractions)
1. `ExecutionBackend` - 216 edges
2. `AppSettings` - 185 edges
3. `DatasetInputSpec` - 180 edges
4. `WorkspaceMode` - 166 edges
5. `IngestionSourceType` - 91 edges
6. `AppMetadataStore` - 80 edges
7. `log_exception()` - 77 edges
8. `ProfilingMode` - 66 edges
9. `LLMProvider` - 65 edges
10. `ValidationRuleConfig` - 60 edges

## Surprising Connections (you probably didn't know these)
- `TestBuildBackend` --uses--> `ColabMCPExecutionBackend`  [INFERRED]
  tests/test_colab_mcp_backend.py → app/backends/colab_mcp_backend.py
- `TestDefaultBackendIsColabMCP` --uses--> `ColabMCPExecutionBackend`  [INFERRED]
  tests/test_colab_mcp_backend.py → app/backends/colab_mcp_backend.py
- `TestEnumOrdering` --uses--> `ColabMCPExecutionBackend`  [INFERRED]
  tests/test_colab_mcp_backend.py → app/backends/colab_mcp_backend.py
- `TestLocalExecutionBackend` --uses--> `ColabMCPExecutionBackend`  [INFERRED]
  tests/test_colab_mcp_backend.py → app/backends/colab_mcp_backend.py
- `TestPackagingColabExtra` --uses--> `ColabMCPExecutionBackend`  [INFERRED]
  tests/test_colab_mcp_backend.py → app/backends/colab_mcp_backend.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **** —  [EXTRACTED 1.00]
- **** —  [EXTRACTED 1.00]
- **docs_assets_screenshots_dashboard_overview_workspace_overview** — docs_assets_screenshots_dashboard_overview_dashboard, docs_assets_screenshots_dashboard_overview_active_dataset, docs_assets_screenshots_dashboard_overview_workspace_metrics, docs_assets_screenshots_dashboard_overview_recent_local_jobs [EXTRACTED 0.98]
- **docs_assets_screenshots_dataset_intake_active_dataset_workflows** — docs_assets_screenshots_dataset_intake_diabetes_dataset, docs_assets_screenshots_dataset_intake_validation, docs_assets_screenshots_dataset_intake_profiling, docs_assets_screenshots_dataset_intake_benchmark, docs_assets_screenshots_dataset_intake_experiment [EXTRACTED 1.00]

## Communities (164 total, 15 thin omitted)

### Community 0 - "App Settings"
Cohesion: 0.03
Nodes (53): AppSettings, model_validator, Top-level application configuration with safe defaults., Settings persistence – load / save runtime settings to a local JSON file.…, format_startup_issues(), initialize_local_runtime(), BaseModel, Startup checks and local-runtime initialization for AutoTabML Studio. (+45 more)

### Community 1 - "Pycaret Service"
Cohesion: 0.04
Nodes (98): Cross-cutting error-handling utilities. This module provides: *…, Artifact generation for experiment runs., _render_markdown(), Exception, PyCaretConfigurationError, PyCaretDependencyError, PyCaretExperimentError, PyCaretTargetError (+90 more)

### Community 2 - "Prediction Base"
Cohesion: 0.05
Nodes (81): DataFrame, Path, Artifact generation for prediction jobs., Persist prediction artifacts and return their paths., _render_markdown(), write_prediction_artifacts(), PredictionService, Prediction service facade. (+73 more)

### Community 3 - "Safe Fetch Policy"
Cohesion: 0.05
Nodes (71): _attempt_download_to_path(), _attempt_download_to_path_async(), _attempt_fetch(), _attempt_fetch_async(), _check_advertised_size(), _check_response_headers(), _normalize_content_type(), BaseException (+63 more)

### Community 4 - "Benchmark Config"
Cohesion: 0.05
Nodes (58): BenchmarkSettings, Configuration for baseline model benchmarking., BenchmarkConfigurationError, Raised when a benchmark configuration is invalid., Benchmarking foundation for AutoTabML Studio., _lazypredict_gpu_usable(), LazyPredictBenchmarkService, DataFrame (+50 more)

### Community 5 - "Execution Backend"
Cohesion: 0.06
Nodes (47): ExecutionBackend, str, Execution backends – where ML jobs actually run., First-class workspace modes., WorkspaceMode, PyCaretExperimentSettings, Configuration for deeper PyCaret experiment workflows., BaseBenchmarkService (+39 more)

### Community 6 - "Test Flaml Automl"
Cohesion: 0.05
Nodes (64): FlamlArtifactsWriter, Path, Artifact generation for FLAML AutoML runs., Shared-path artifact writer for FLAML result bundles., Write FLAML artifacts to disk and return their paths., write_flaml_artifacts(), FlamlAutoMLError, FlamlConfigurationError (+56 more)

### Community 7 - "App Cli"
Cohesion: 0.08
Nodes (81): _add_prediction_model_source_args(), _build_prediction_request_kwargs(), _build_prediction_service(), _build_pycaret_service(), _cli_error(), cmd_batch_history(), cmd_batch_show(), cmd_benchmark() (+73 more)

### Community 8 - "Dataset Workspace"
Cohesion: 0.07
Nodes (51): LoadedDataset, DataFrame, Loaded and normalized tabular dataset returned by ingestion., Return a safe preview copy of the first *rows* records., build_local_path_input_spec(), build_url_input_spec(), _clear_dataset_results(), clear_loaded_datasets() (+43 more)

### Community 9 - "Benchmark Task Type"
Cohesion: 0.06
Nodes (57): BenchmarkDependencyError, BenchmarkError, BenchmarkExecutionError, BenchmarkTargetError, BenchmarkTrackingError, Exception, Custom exceptions for the benchmark layer., Raised when an optional runtime dependency is unavailable. (+49 more)

### Community 10 - "Url Loader"
Cohesion: 0.08
Nodes (50): CSVLoader, Any, DataFrame, Path, CSV and delimiter-aware text loader., Load local or remote delimited tabular files into pandas., ParseFailureError, Custom exceptions for dataset ingestion. (+42 more)

### Community 11 - "Test Hardening Smoke"
Cohesion: 0.06
Nodes (37): BenchmarkArtifactBundle, BenchmarkSummary, BaseModel, Artifact paths emitted for a benchmark run., Roll-up summary for a benchmark run., ExperimentArtifactsWriter, Path, Write experiment artifacts to disk and return their paths. (+29 more)

### Community 12 - "Test Ingestion Utils"
Cohesion: 0.05
Nodes (43): Any, DataFrame, Return a raw DataFrame and source details before normalization., Async counterpart of :meth:`load_raw_dataframe`. Loaders with native async I/O…, Load, normalize, and enrich a dataset from the supplied input spec., Async counterpart of :meth:`load` for I/O-heavy loader implementations., Load only a preview slice where the underlying loader supports it., Async counterpart of :meth:`preview`. (+35 more)

### Community 13 - "Ingestion Source Type"
Cohesion: 0.08
Nodes (33): BaseLoader, Base abstractions shared by all ingestion loaders., Common load lifecycle for all dataset sources., DataFrameLoader, In-memory pandas DataFrame loader., Accept a pandas DataFrame without mutating caller-owned data., get_loader(), preview_dataset() (+25 more)

### Community 14 - "Mlflow Query"
Cohesion: 0.07
Nodes (56): create_model_version(), create_registered_model(), delete_model_alias(), delete_model_version_tag(), _extract_dataset_name(), _extract_tags(), _get_client(), get_experiment_by_name() (+48 more)

### Community 15 - "Benchmark Page"
Cohesion: 0.06
Nodes (46): flaml_install_guidance(), is_flaml_available(), Return True when FLAML is importable., Return a user-facing installation hint for environments without FLAML., Streamlit page for LazyPredict benchmark execution and results., Resolve a LazyPredict model name to its sklearn-compatible estimator class., Render controls to retrain and save the best model from benchmark results., render_benchmark_page() (+38 more)

### Community 16 - "Dataset Input Spec"
Cohesion: 0.09
Nodes (21): load_dataset(), Public entry point for full dataset loading., DatasetInputSpec, model_validator, Canonical input contract for all supported ingestion paths., Return the human-readable locator for lineage and logging., Validate source-specific input requirements., mock (+13 more)

### Community 17 - "Saved Model Metadata"
Cohesion: 0.07
Nodes (37): discover_saved_benchmark_models(), Discover trusted benchmark models from checksum-backed metadata sidecars., build_saved_model_metadata(), load_experiment_snapshot(), load_model_artifact(), datetime, Path, Finalize, save, and load helpers for experiment artifacts. (+29 more)

### Community 18 - "Prediction Page"
Cohesion: 0.08
Nodes (46): Optional MLflow side-by-side run comparison., _render_mlflow_comparison(), Inline dataset loader shown when no dataset is active., Persist one uploaded file to the app temp area and return an input spec., _render_inline_dataset_loader(), uploaded_file_to_input_spec(), Optional collapsible section to browse raw MLflow runs., _render_mlflow_section() (+38 more)

### Community 19 - "Validation Rule Config"
Cohesion: 0.08
Nodes (26): User-facing configuration for a validation run., ValidationRuleConfig, constant_df(), duplicate_rows_df(), empty_df(), good_df(), null_heavy_df(), DataFrame (+18 more)

### Community 20 - "Test Modeling Exception Handling"
Cohesion: 0.06
Nodes (36): get_mlflow_module(), is_mlflow_available(), Return True when mlflow is importable., Import and return the mlflow module., _build_params(), _get_mlflow_module(), is_mlflow_available(), _log_artifacts() (+28 more)

### Community 21 - "Experiment Page"
Cohesion: 0.07
Nodes (34): pycaret_install_guidance(), Return a user-facing installation hint for environments without PyCaret., build_experiment_run_key(), _build_service(), default_compare_metric_for_task(), default_plot_ids_for_task(), default_tracking_mode(), default_tune_metric_for_task() (+26 more)

### Community 22 - "Generate Job Notebook"
Cohesion: 0.09
Nodes (38): _benchmark_cells(), _experiment_cells(), _flaml_cells(), generate_job_notebook(), _json_literal(), _json_string(), _markdown_cell(), NotebookGenerationError (+30 more)

### Community 23 - "Flaml Auto Mlservice"
Cohesion: 0.08
Nodes (26): _build_flaml_service(), FlamlSettings, Configuration for FLAML AutoML workflows., FlamlAutoMLService, Path, Persist the best model from a FLAML search., FLAML-backed AutoML service., Path (+18 more)

### Community 24 - "Prediction Workflow Service"
Cohesion: 0.07
Nodes (20): Page-layer services that keep Streamlit pages focused on rendering., ModelTestingEvaluation, ModelTestingRunResult, ModelTestingSelection, PredictionExecutionConfig, PredictionWorkflowService, Any, DataFrame (+12 more)

### Community 25 - "Get Or Init State"
Cohesion: 0.08
Nodes (44): _find_model_column(), _find_score_column(), _leaderboard_rows_to_df(), _load_leaderboard(), DataFrame, Streamlit page for comparing algorithm performance on a dataset., Try to load a leaderboard from the job's artifact paths., Convert a list of leaderboard row dicts to a clean DataFrame. (+36 more)

### Community 26 - "Ui Labels"
Cohesion: 0.07
Nodes (45): Streamlit page for model testing – evaluate a trained model on real-world data., Show evaluation metrics when ground-truth labels are available., _render_evaluation_metrics(), render_model_testing_page(), Streamlit page for browsing all saved models with details., Render an expander card for a PyCaret experiment model., Render an expander card for a benchmark-saved model., Render an expander card for an MLflow registry model. (+37 more)

### Community 27 - "Test Tracking"
Cohesion: 0.10
Nodes (31): cmd_history_list(), List MLflow runs from the history center., build_mlflow_filter_string(), BaseModel, Enum, str, Filtering and sorting helpers for run history queries., Fields available for sorting run history. (+23 more)

### Community 28 - "Config Llmprovider"
Cohesion: 0.09
Nodes (26): LLMProvider, Enum, Enumerations for AutoTabML Studio configuration., Supported LLM providers., Return the verified stable fallback model id for the given (or current)…, Anthropic provider — model discovery and text generation via the official SDK.…, BaseProvider, ModelItem (+18 more)

### Community 29 - "Validate Dataset"
Cohesion: 0.09
Nodes (34): benchmark_dataset(), DataFrame, Path, Run a benchmark with the LazyPredict benchmark service., Path, Convenience function: validate and optionally write artifacts. Returns…, validate_dataset(), Exercise non-trivial rule config — uniqueness, range, category checks. (+26 more)

### Community 30 - "Experiment Result Bundle"
Cohesion: 0.13
Nodes (19): Tune one selected model for an existing bundle., PyCaretExecutionError, Raised when a PyCaret operation fails., PyCaret-backed experiment lab for AutoTabML Studio., ExperimentResultBundle, ExperimentRuntimeState, ModelSelectionSpec, Stable user-facing model selection reference. (+11 more)

### Community 31 - "Prediction Loader"
Cohesion: 0.10
Nodes (37): ModelDiscoveryError, Raised when a model source cannot be discovered or resolved cleanly., Model-loading wrappers for prediction flows., AvailableModelReference, PredictionTaskType, Task types supported during prediction., Normalized model reference shown in discovery and selection UIs., build_mlflow_registered_model_uri() (+29 more)

### Community 32 - "Pyrightconfig Json"
Cohesion: 0.04
Nodes (45): exclude, ignore, include, pythonPlatform, pythonVersion, reportArgumentType, reportAttributeAccessIssue, reportCallIssue (+37 more)

### Community 33 - "Gather With Concurrency"
Cohesion: 0.08
Nodes (32): gather_with_concurrency(), Any, BaseException, T, Small async concurrency helpers used across the app. These helpers exist so…, Run awaitables concurrently with at most ``limit`` running at a time. ``limit``…, Run a blocking ``func`` over many argument batches concurrently in threads.…, to_thread_many() (+24 more)

### Community 34 - "Log Exception"
Cohesion: 0.09
Nodes (39): AutoTabMLError, log_and_wrap(), log_exception(), Any, BaseException, Exception, Logger, Opt-in umbrella base class for application-level domain errors. (+31 more)

### Community 35 - "Flaml Config"
Cohesion: 0.08
Nodes (14): FlamlConfig, FlamlSearchConfig, Configuration for the FLAML AutoML search., Top-level FLAML AutoML configuration., classification_df(), _FakeAutoML, _make_service(), DataFrame (+6 more)

### Community 36 - "Build Input Spec"
Cohesion: 0.08
Nodes (15): _build_input_spec(), _is_url(), Build an ingestion input spec from a CLI dataset locator., Path, Regression: experiment-tune/evaluate/save must catch setup errors cleanly., Happy-path coverage for experiment-tune, experiment-evaluate, experiment-save., Direct functional tests for cmd_history_list., Direct functional tests for cmd_profile. (+7 more)

### Community 37 - "Test Prediction"
Cohesion: 0.08
Nodes (21): PredictionHistoryError, Raised when prediction history cannot be persisted or queried., PredictionHistoryStore, Path, Lightweight local prediction-history storage., Persist and query recent prediction jobs via newline-delimited JSON., Append one prediction-history record to disk., Return recent prediction jobs ordered newest-first. (+13 more)

### Community 38 - "Validation Service"
Cohesion: 0.08
Nodes (30): Path, Produce validation artifacts: JSON summary, Markdown report., Write validation artifacts to disk and return the bundle., _render_markdown(), write_artifacts(), BaseValidationService, DataFrame, Base abstraction for validation services. (+22 more)

### Community 39 - "Trusted Artifacts"
Cohesion: 0.10
Nodes (35): load_saved_benchmark_model(), load_saved_benchmark_model_metadata_file(), Path, Trusted benchmark model discovery and loading helpers., Parse a benchmark saved-model metadata file, returning None when invalid., Load a trusted benchmark model from a skops artifact., BenchmarkSavedModelMetadata, Metadata sidecar for a benchmark-saved local model. (+27 more)

### Community 40 - "Settings Page"
Cohesion: 0.09
Nodes (29): _fetch_models(), Settings / Runtime configuration page for Streamlit., Lightweight GPU status for the Essentials tab — no controls., Simple on/off for run descriptions on the Essentials tab., Run an async coroutine from sync Streamlit code. Streamlit may already have a…, Build a provider instance and fetch models, updating state., Render the full Settings page., Always-visible privacy reminder at the top of Essentials. (+21 more)

### Community 41 - "Gx Runner"
Cohesion: 0.08
Nodes (35): Exception, Custom exceptions for the validation layer., Raised when the validation infrastructure cannot be initialized., Raised when a validation rule configuration is invalid., Base exception for validation failures., RuleConfigError, ValidationError, ValidationSetupError (+27 more)

### Community 42 - "Test Profiling"
Cohesion: 0.10
Nodes (25): ProfilingSettings, Configuration for the automated profiling layer., ProfilingConfig, User-facing configuration for a profiling run., maybe_sample(), DataFrame, Automatic mode selection and DataFrame sampling for profiling., Choose profiling mode based on dataset size and config thresholds. (+17 more)

### Community 43 - "Run History Item"
Cohesion: 0.10
Nodes (31): Path, Artifact generation for comparison bundles., Write comparison artifacts and return their paths., _render_markdown(), write_comparison_artifacts(), _check_comparability(), _compute_config_differences(), _compute_metric_deltas() (+23 more)

### Community 44 - "Registry Service"
Cohesion: 0.14
Nodes (15): High-level service for MLflow model registry operations., RegistryService, _safe_version_sort_key(), PromotionRequest, Request to promote a model version., _FakeModelVersion, _FakeRegisteredModel, _patch_registry() (+7 more)

### Community 45 - "Profiling Mode"
Cohesion: 0.08
Nodes (24): ProfilingMode, Enum, str, Pydantic configuration models for AutoTabML Studio., Severity levels for validation checks., Profiling report modes., ValidationSeverity, Profiling artifact generation helpers. (+16 more)

### Community 46 - "Batch Uci Runner"
Cohesion: 0.10
Nodes (32): _deep_merge(), load_settings(), Load settings from the local JSON file, falling back to defaults., Counter, Monotonically increasing counter., build_200_dataset_list(), main(), Batch UCI dataset runner – runs validate → profile → benchmark for 200 NEW UCI… (+24 more)

### Community 47 - "Test Observability"
Cohesion: 0.11
Nodes (32): bind_context(), clear_context(), correlation_scope(), current_context(), _empty_context_default(), new_correlation_id(), Any, Correlation context for log/metric/trace records. A small wrapper around… (+24 more)

### Community 48 - "Registry Service"
Cohesion: 0.09
Nodes (21): Model registry and promotion workflows for AutoTabML Studio., Model registry service – list, inspect, register, and tag models., List all registered models., Get a specific model version., Get the model version currently assigned to an alias., Register a model and create its first version. Creates the registered model if…, PromotionAction, PromotionResult (+13 more)

### Community 49 - "Sqlite Connector"
Cohesion: 0.10
Nodes (16): Connection, Path, T, Reusable SQLite connector with safe defaults for local metadata storage., Open SQLite connections with consistent PRAGMAs and atomic write helpers., Yield a configured connection for read operations., Run a read-only callback using a configured connection., Run a write callback in an atomic transaction with lock retries. (+8 more)

### Community 50 - "Cuda Summary"
Cohesion: 0.11
Nodes (20): cuda_device_name(), cuda_summary(), _driver_probe(), is_cuda_available(), CUDA / GPU detection utilities for AutoTabML Studio., Lightweight probe for the NVIDIA driver library without importing torch. Loads…, Return True when a CUDA-capable GPU is reachable from the current Python…, Return the name of the first CUDA device, or None. (+12 more)

### Community 51 - "Model Source Type"
Cohesion: 0.09
Nodes (17): ModelLoadError, Raised when a prediction model cannot be loaded., Raised when a model artifact fails trust-boundary validation., TrustedArtifactError, MLflowModelLoader, ModelLoader, ABC, Any (+9 more)

### Community 52 - "Gemini Provider"
Cohesion: 0.09
Nodes (9): GeminiProvider, Any, OpenAIProvider, Any, As of v0.2.0 the Gemini provider uses the official google-genai SDK which…, Verify that each provider is now backed by the official SDK client. The SDKs…, v0.2.0: ``httpx`` is no longer used by any LLM provider. (``httpx`` is still…, TestModelNormalization (+1 more)

### Community 53 - "App Metadata Store"
Cohesion: 0.15
Nodes (18): Local metadata storage exports., JobRecord, Local app job record used for dashboard/history convenience., _enrich_metadata_with_description(), ensure_dataset_record(), Any, Helpers that map app workflows into local metadata-store records., Add a template MLflow description to the metadata dict. (+10 more)

### Community 54 - "Generate Template Description"
Cohesion: 0.10
Nodes (16): _build_footer(), _build_llm_prompt(), generate_llm_description(), generate_template_description(), _generic_template(), _job_icon(), Any, Generate professional run descriptions for job runs. Two modes: - **Template**… (+8 more)

### Community 55 - "Observability Init"
Cohesion: 0.11
Nodes (16): Production-grade observability primitives for AutoTabML Studio. This package…, get_metrics_backend(), InMemoryMetricsBackend, MetricsBackend, NoopMetricsBackend, Protocol, Pluggable metrics façade. Three primitive types are exposed – :class:`Counter`,…, Install ``backend`` as the active sink and return the previous one. (+8 more)

### Community 56 - "List Available Uci Datasets"
Cohesion: 0.12
Nodes (19): load_dataset_async(), Async public entry point for full dataset loading., _import_ucimlrepo(), list_available_uci_datasets(), list_available_uci_datasets_async(), _parse_catalog_output(), Any, DataFrame (+11 more)

### Community 57 - "Dataframe To Safe Csv"
Cohesion: 0.11
Nodes (23): dataframe_to_safe_csv(), Any, DataFrame, Safe CSV export helpers. Excel and similar spreadsheet tools can evaluate cells…, Return a copy safe for CSV export. Sanitizes string cells, column labels, index…, Serialize ``dataframe`` to CSV with formula-safe cells and strict quoting., sanitize_csv_dataframe(), _sanitize_csv_scalar() (+15 more)

### Community 58 - "Install Correlation Filter"
Cohesion: 0.11
Nodes (24): configure_logging(), _configure_noisy_dependency_loggers(), Logging configuration for AutoTabML Studio. Thin compatibility shim over…, Reduce non-actionable third-party log noise in normal app and batch runs., Configure structured logging to stderr (12-factor compliant). Honors the…, configure_observability_logging(), CorrelationFilter, install_correlation_filter() (+16 more)

### Community 59 - "Modeling Base"
Cohesion: 0.09
Nodes (21): BaseService, mlflow_exception_types(), BaseException, Shared base classes for modeling services, trackers, and artifact writers., Return common MLflow exception types plus generic boundary failures., Common modeling-service configuration and helpers., Return a configured tracker instance when MLflow tracking is enabled., _build_params() (+13 more)

### Community 60 - "History Service"
Cohesion: 0.16
Nodes (12): HistoryService, Fetch extended detail for a single run., Resolve experiment name(s) to ids, building a name lookup map., High-level service for querying and inspecting run history., Resolve a possibly truncated run ID prefix to a full 32-char ID. If…, _FakeMLflowExperiment, _FakeMLflowRun, _patch_mlflow_query() (+4 more)

### Community 61 - "Run App Rules"
Cohesion: 0.28
Nodes (27): _check_allowed_categories(), _check_constant_columns(), _check_dtype_summary(), _check_duplicate_column_names(), _check_duplicate_rows(), _check_fully_null_columns(), _check_id_columns(), _check_leakage_heuristics() (+19 more)

### Community 62 - "List Registered Models"
Cohesion: 0.21
Nodes (19): list_registered_models(), Return all registered models from the MLflow model registry. Uses paginated…, _clear_cache(), _CountingClient, _FakeRegisteredModel, _FakeVersion, _patch_client(), fixture (+11 more)

### Community 63 - "Ui Cache"
Cohesion: 0.13
Nodes (25): _dataset_cache_signature(), _get_experiment_workflow_service_resource(), get_flaml_automl_service(), _get_flaml_automl_service_resource(), get_history_service(), _get_history_service_resource(), _get_metadata_store_resource(), get_prediction_service() (+17 more)

### Community 64 - "Ydata Runner"
Cohesion: 0.11
Nodes (19): BaseProfilingService, DataFrame, Base abstraction for profiling services., Interface that all profiling service implementations must satisfy., Run profiling and return summary + optional artifacts., ProfilingArtifactBundle, ProfilingResultSummary, BaseModel (+11 more)

### Community 65 - "Validate Public Release Metadata"
Cohesion: 0.15
Nodes (18): check_public_release_metadata(), _collect_contacts(), _has_license_metadata(), load_project_metadata(), main(), _normalized(), Any, Path (+10 more)

### Community 66 - "Base Repository"
Cohesion: 0.12
Nodes (10): BaseRepository, datetime, Path, Shared base utilities for storage repositories., Bundle of shared dependencies passed to every repository. The context holds the…, Common access patterns shared by every domain repository. Subclasses focus on…, RepositoryContext, Batch run repositories (runs and items). (+2 more)

### Community 67 - "Validation Settings"
Cohesion: 0.11
Nodes (21): ArtifactSettings, DatabaseSettings, MLflowSettings, PredictionSettings, ProviderSettings, BaseModel, field_validator, setter (+13 more)

### Community 68 - "App Main"
Cohesion: 0.15
Nodes (16): AutoTabML Studio – Streamlit entry point., default_page_label(), get_nav_sections(), get_page_by_label(), get_page_registry(), PageSpec, Central registry for Streamlit page navigation and rendering., Declarative Streamlit page registration. (+8 more)

### Community 69 - "Batch Run Record"
Cohesion: 0.12
Nodes (8): BatchRunRecord, Tracks an overall batch execution run., Connection, T, BatchRunRepository, Connection, Row, CRUD operations for batch runs and their items.

### Community 70 - "Benchmark Result Bundle"
Cohesion: 0.15
Nodes (17): BenchmarkArtifactsWriter, Path, Artifact generation for benchmark runs., Shared-path artifact writer for benchmark result bundles., Write benchmark artifacts to disk and return their paths., _render_markdown(), write_benchmark_artifacts(), _write_score_chart() (+9 more)

### Community 71 - "Profile Dataset"
Cohesion: 0.19
Nodes (22): _format_pycaret_import_error(), _probe_pycaret_import_error(), Exception, Return the import-time failure when PyCaret is unusable in the current runtime., Normalize import-time dependency failures into a stable one-line message., profile_dataset(), DataFrame, Path (+14 more)

### Community 72 - "Test Cli"
Cohesion: 0.15
Nodes (16): AppJobStatus, AppJobType, BatchItemStatus, BatchRunStatus, Enum, str, Typed records for the local app metadata database., Individual dataset run status within a batch. (+8 more)

### Community 73 - "Capture Screenshots"
Cohesion: 0.18
Nodes (22): Page, capture(), choose_streamlit_option(), main(), navigate_to_page(), Path, Automated screenshot capture for AutoTabML Studio using Playwright. Launches a…, Navigate to Profiling and run it. (+14 more)

### Community 74 - "Test Storage Repositories"
Cohesion: 0.13
Nodes (14): _dataset(), _job(), _now(), datetime, fixture, Path, Domain repository tests for the modular storage layer., store() (+6 more)

### Community 75 - "Observability Histogram"
Cohesion: 0.12
Nodes (14): Gauge, Histogram, _merge_labels(), _Metric, Any, Shared base – holds a metric name and forwards to the active backend., Point-in-time numeric value., Distribution of observed values (e.g. durations in seconds). (+6 more)

### Community 76 - "Comparison Service"
Cohesion: 0.20
Nodes (8): ComparisonService, Produce structured comparisons between two runs., Return a compact one-line summary of a run., run_summary_line(), _make_run(), Path, TestComparisonService, TestRunSummary

### Community 77 - "Local Artifact Manager"
Cohesion: 0.20
Nodes (6): LocalArtifactManager, DataFrame, Path, Create, write, and conservatively clean local workspace artifacts., Return a filesystem-safe stem for generated artifact filenames., safe_artifact_stem()

### Community 78 - "Excel Loader"
Cohesion: 0.19
Nodes (10): ExcelLoader, Any, DataFrame, Path, Load local or remote Excel workbooks., KaggleLoader, Any, DataFrame (+2 more)

### Community 79 - "Base Tracker"
Cohesion: 0.14
Nodes (10): BaseTracker, Logger, Shared MLflow tracking lifecycle for modeling bundles., Log params, metrics, and artifacts for one modeling bundle., Return True if the subclass-specific MLflow boundary is available., Return the subclass-specific MLflow module handle., Return the operation label used for structured error logging., Return the MLflow run name for the bundle. (+2 more)

### Community 80 - "Colab Mcpexecution Backend"
Cohesion: 0.22
Nodes (5): ColabMCPExecutionBackend, Execution backend that delegates work to a Google Colab runtime via MCP., Check that ``uvx`` is installed and the MCP SDK is importable., asyncio, TestColabMCPExecutionBackend

### Community 81 - "Test Colab Mcp Backend"
Cohesion: 0.18
Nodes (9): LocalExecutionBackend, Any, ExecutionSettings, Settings related to the execution backend., Tests for Colab MCP backend and notebook infrastructure., TestDefaultBackendIsColabMCP, TestEnumOrdering, TestLocalExecutionBackend (+1 more)

### Community 82 - "Test Packaging Metadata"
Cohesion: 0.18
Nodes (6): Persist settings to ~/.autotabml/settings.json (secrets excluded)., save_settings(), Path, As of v0.2.0 the project builds with hatchling; the profiling extra no longer…, TestPackagingMetadata, TestSettingsPersistence

### Community 83 - "Htmltable Loader"
Cohesion: 0.21
Nodes (8): HTMLTableLoader, Any, DataFrame, Extract HTML tables from a URL and return one table as a DataFrame., Any, DataFrame, Route a URL to the correct concrete loader., URLLoader

### Community 84 - "Start Span"
Cohesion: 0.15
Nodes (13): _NoopSpan, Any, BaseException, Protocol, Optional OpenTelemetry tracing with a stdlib-only fallback. We deliberately do…, The minimal span surface used by AutoTabML Studio call sites., Open a span named ``name`` and yield a :class:`SpanLike`. When OpenTelemetry is…, SpanLike (+5 more)

### Community 85 - "Registry Errors"
Cohesion: 0.13
Nodes (15): ModelNotFoundError, PromotionError, Exception, Custom exceptions for the model registry layer., Raised when the MLflow model registry backend is not available., Raised when a registered model cannot be found., Raised when a model version cannot be found., Raised when a promotion action cannot be completed. (+7 more)

### Community 86 - "Base Artifacts"
Cohesion: 0.16
Nodes (9): BaseArtifacts, Any, Path, Yield the artifact paths that should be uploaded to MLflow., Shared helpers for artifact path generation and persistence., Write artifacts for the current bundle and return the bundle of paths., Append warnings to the bundle and its summary without duplicates., ArtifactBundleT (+1 more)

### Community 87 - "Ydata Profiling Service"
Cohesion: 0.17
Nodes (10): Any, DataFrame, Path, Suppress known non-actionable third-party warnings during profiling., Profiling service backed by ydata-profiling., _suppress_profiling_runtime_noise(), YDataProfilingService, skipif (+2 more)

### Community 88 - "Base Execution Backend"
Cohesion: 0.16
Nodes (10): BaseExecutionBackend, Any, Abstract execution backend interface., Common interface for all execution backends., Check that the backend is reachable / properly configured., Prepare any runtime context needed before a job runs., Execute a job payload and return results. Concrete implementations will be…, Colab MCP execution backend – connects to Google Colab via MCP. This backend… (+2 more)

### Community 89 - "Redact Key In Text"
Cohesion: 0.17
Nodes (7): LogRecord, Replace anything that looks like an API key or credential in free text., redact_key_in_text(), Tests for security / masking helpers., TestLoggingRedaction, TestRedactKeyInText, TestSafeErrorMessage

### Community 90 - "Notebook Page"
Cohesion: 0.19
Nodes (15): _find_uvx(), Return the path to ``uvx`` if it is installed, else *None*., _generate_notebook_for_job(), Path, Notebook page – browse auto-generated notebooks per dataset / job. Each…, Generate and return notebook path for a job record., Render a simple preview of notebook cells., Execute *coro* from synchronous Streamlit code. (+7 more)

### Community 91 - "Test Ui Cache"
Cohesion: 0.23
Nodes (15): _section_cache_controls(), invalidate_all_ui_caches(), invalidate_dataset_cache(), invalidate_mlflow_query_cache(), invalidate_service_cache(), Clear cached dataset loads., Clear cached MLflow read models and the query-layer registry cache., Clear cached service and metadata-store resources. (+7 more)

### Community 92 - "Storage Migrations"
Cohesion: 0.31
Nodes (13): apply_migrations(), _detect_legacy_version(), _ensure_version_table(), _get_applied_versions(), Migration, Connection, Incremental SQLite schema migrations for the local metadata store., Create the version table and apply any pending schema migrations. (+5 more)

### Community 93 - "Batch Run Item Record"
Cohesion: 0.21
Nodes (11): BatchRunItemRecord, Tracks a single dataset within a batch run through validate/profile/benchmark., Path, When ALL datasets are already completed, batch record counts must still be set., Regression: targets that previously had wrong casing must match their UCI…, test_declared_target_from_item_prefers_item_id_suffix_for_resume_keys(), test_full_resume_updates_batch_record_counts(), test_known_case_sensitive_target_mappings() (+3 more)

### Community 94 - "Project Record"
Cohesion: 0.21
Nodes (7): ProjectRecord, Local project metadata for workspace navigation convenience., ProjectRepository, Connection, Row, CRUD operations for project rows., Upsert using an existing connection (used during initial migration).

### Community 95 - "Saved Local Model Record"
Cohesion: 0.21
Nodes (8): BaseModel, Local saved-model metadata kept outside MLflow registry state., SavedLocalModelRecord, Connection, Path, Row, CRUD operations for saved local model rows., SavedModelRepository

### Community 96 - "Dataset Record"
Cohesion: 0.23
Nodes (6): DatasetRecord, Locally persisted dataset lineage record., DatasetRepository, Connection, Row, CRUD operations for dataset rows.

### Community 97 - "Tracking Errors"
Cohesion: 0.21
Nodes (12): ComparisonError, ExperimentNotFoundError, Exception, Custom exceptions for the tracking and history layer., Raised when MLflow tracking is not available or configured., Raised when a requested run cannot be found., Raised when a requested MLflow experiment cannot be found., Raised when a comparison cannot be completed. (+4 more)

### Community 98 - "Test Mlflow Normalization"
Cohesion: 0.18
Nodes (4): _extract_primary_metric(), _infer_run_type(), _safe_run_status(), TestMLflowNormalization

### Community 99 - "Auto Tab Ml Studio Social"
Cohesion: 0.17
Nodes (12): AutoTabML Studio Social Preview, Benchmarking, CLI workflows, Dashboard, Local-first tabular ML workbench, MLflow-backed history, Model Registry, Portfolio-ready local ML tooling (+4 more)

### Community 100 - "Test Runtime State"
Cohesion: 0.17
Nodes (5): Switching to colab_mcp while provider=ollama should auto-reset provider., Switching backend when provider is valid for both should keep it., Setting the same backend value should not clear anything., backend_valid was removed — make sure it's not in to_dict., TestRuntimeState

### Community 101 - "Prepare Session"
Cohesion: 0.20
Nodes (6): Any, Execute a job by calling Colab MCP tools. ``job_payload`` must include a…, Call ``open_colab_browser_connection`` to link to a Colab notebook., Return the currently available MCP tool names., Tear down the MCP session and server subprocess., Spawn the Colab MCP server and establish a client session. Returns a dict with…

### Community 102 - "Test Optional Dependency Probes"
Cohesion: 0.18
Nodes (6): is_lazypredict_available(), Return True if lazypredict is importable., is_pycaret_available(), Return True when PyCaret classification and regression modules are importable., Verify each optional dep has a clean availability check., TestOptionalDependencyProbes

### Community 103 - "Code Of Conduct"
Cohesion: 0.18
Nodes (11): Community interest, Code of Conduct, Enforcement actions, Giving and accepting feedback gracefully, Harassment-free participation, Maintainer enforcement responsibility, Positive environment, Project spaces (+3 more)

### Community 104 - "Artifact Kind"
Cohesion: 0.24
Nodes (7): Local artifact management utilities., ArtifactKind, datetime, Enum, str, Centralized local artifact path and lifecycle management. This manager only…, Supported local artifact directory kinds.

### Community 105 - "Build Provider"
Cohesion: 0.33
Nodes (3): build_provider(), Instantiate the concrete provider. Falls back to environment variables for…, TestBuildProvider

### Community 106 - "Ollama Provider"
Cohesion: 0.29
Nodes (3): OllamaProvider, Any, Ollama has no auth — just check reachability.

### Community 107 - "Test Colab Mcpreal Handshake"
Cohesion: 0.20
Nodes (4): Integration test — real Colab MCP server handshake (no mocks). Marked…, Prove the real colab-mcp server spawns and the MCP handshake works., Calling a tool that needs a browser session should fail gracefully., TestColabMCPRealHandshake

### Community 108 - "Experiment Setup Summary"
Cohesion: 0.25
Nodes (6): ExperimentSetupSummary, _is_json_safe_metric_kwarg(), Any, field_validator, Return a serializable setup summary model., Serializable record of the normalized setup stage.

### Community 109 - "Base Prediction Service"
Cohesion: 0.22
Nodes (6): BasePredictionService, ABC, Abstract prediction service contract., Return discoverable local saved models., Run one-row prediction., Run batch prediction.

### Community 110 - "Profiling Init"
Cohesion: 0.31
Nodes (7): ProfilingError, ProfilingSetupError, Exception, Custom exceptions for the profiling layer., Raised when the profiling library cannot be initialized., Base exception for profiling failures., Automated profiling layer for AutoTabML Studio.

### Community 112 - "Resolve Default Model"
Cohesion: 0.43
Nodes (4): Pick the best default from a fetched model list. 1. If the hardcoded default…, resolve_default_model(), _make_items(), TestResolveDefaultModel

### Community 113 - "Mask Secret"
Cohesion: 0.39
Nodes (3): mask_secret(), Mask a secret string, keeping only *reveal* chars at each end visible.…, TestMaskSecret

### Community 114 - "Screenshots Benchmark"
Cohesion: 0.25
Nodes (8): Benchmark, Benchmark configuration screen, Random seed: 42, Sample rows: 0 (full dataset), Target column: target, Task type: regression, Test size: 0.20, Top-k shortlist: 5

### Community 115 - "History View"
Cohesion: 0.29
Nodes (8): History navigation item, History view, Inspect run selector, Navigation sidebar, Run Detail, Run history table, Run metadata columns, Selected benchmark classification run

### Community 116 - "Eda Profiling Dashboard"
Cohesion: 0.36
Nodes (8): Column Types: 11 Numeric, 0 Categorical, Data Quality Summary: 0.0% Missing, 0 Duplicates, Dataset Dimensions: 442 Rows, 11 Columns, Active Diabetes Dataset, Profiling Complete, Profiling Artifacts, EDA / Profiling Dashboard, Standard Report Mode

### Community 117 - "Settings View"
Cohesion: 0.29
Nodes (8): Accelerators, colab_mcp backend, Execution backend, Local backend, Application navigation, Promote version, Register model, Settings view

### Community 118 - "Test Artifact Manager"
Cohesion: 0.46
Nodes (4): _artifact_settings_for(), Path, Tests for centralized local artifact lifecycle management., TestLocalArtifactManager

### Community 120 - "Resolve Task Type"
Cohesion: 0.38
Nodes (4): Series, Resolve the effective FLAML task type and validate the target., resolve_task_type(), TestFlamlSetupRunner

### Community 121 - "Compare Runner"
Cohesion: 0.33
Nodes (6): create_model(), Any, compare_models and create_model wrappers., Execute compare_models with explicit, testable kwargs., Create one concrete model from a model id., run_compare_models()

### Community 122 - "Auto Tab Ml Studio V0"
Cohesion: 0.29
Nodes (7): Changelog, Developer Guide, Release Readiness workflow, 0.1.x to 0.2.0 migration guide, v0.2.0 release notes, AutoTabML Studio v0.2.0, 0.1.x to 0.2.0 upgrade summary

### Community 123 - "Benchmark Configuration Form"
Cohesion: 0.29
Nodes (7): Benchmark configuration form, Experiment Lab, Experiment navigation item, Navigation sidebar, Run Benchmark button, Target column selector, Task type selector (regression)

### Community 124 - "Data Validation"
Cohesion: 0.29
Nodes (7): Active Dataset: diabetes, Data Validation, Dataset Dimensions: 442 Rows, 11 Columns, Data Validation Screen, Target Column: target, Validation Complete, Validation Results: 5 Passed, 0 Warnings, 0 Failed

### Community 125 - "Test Cli Output Encoding"
Cohesion: 0.29
Nodes (3): Regression tests for CLI output encoding on Windows cp1252., Ensure compare CLI output uses ASCII-safe arrows (->), not Unicode → (\u2192)., TestCliOutputEncoding

### Community 126 - "Test Benchmark Executes Real Lazypredict"
Cohesion: 0.43
Nodes (7): DataFrame, Path, _small_classification_df(), _sqlite_uri(), test_benchmark_executes_real_lazypredict_and_mlflow(), test_gx_validation_executes_real_expectations(), test_ydata_profile_executes_real_report()

### Community 128 - "Build Backend"
Cohesion: 0.47
Nodes (3): build_backend(), Create the execution backend instance for the given enum value., TestBuildBackend

### Community 129 - "Mlflow Run Comparison View"
Cohesion: 0.33
Nodes (6): Artifact Availability, Saved comparison artifacts, Comparison status: not comparable, MLflow run comparison view, Compare view screenshot, Metadata verification warnings

### Community 130 - "Operations Guide"
Cohesion: 0.33
Nodes (6): Local artifact storage layout, Local Docker Compose stack, Architecture Guide, Operations Guide, MLflow local tracking and registry, Trusted local model source marker

### Community 131 - "Screenshots Dashboard"
Cohesion: 0.33
Nodes (6): dashboard-overview, Active dataset diabetes, Dashboard, Application navigation, Recent Local Jobs, Workspace metrics

### Community 132 - "Loaded Dataset Diabetes"
Cohesion: 0.33
Nodes (6): Benchmark, Loaded dataset 'diabetes', Experiment, Load Local Path, Profiling, Validation

### Community 133 - "Prediction Center Interface"
Cohesion: 0.33
Nodes (6): Discovered Local Model, Load Model Action, Model Loading Guidance, Prediction Center Interface, Prediction Navigation, Task Type Hint

### Community 134 - "Model Registry"
Cohesion: 0.33
Nodes (6): e2e-demo-classifier, Model Version Promotion, Model Registry, Model Versions, Registered Models, Registry View Screenshot

### Community 135 - "Colab Mcp Spike"
Cohesion: 0.47
Nodes (5): _check_prerequisites(), main(), Return a list of missing prerequisite descriptions., Spawn the colab-mcp server and perform the MCP handshake., _run_spike()

### Community 137 - "Get Allowed Providers"
Cohesion: 0.50
Nodes (3): get_allowed_providers(), Return providers allowed for the given execution backend., TestAllowedProviders

### Community 138 - "Ci Workflow"
Cohesion: 0.40
Nodes (5): SHA-pinned GitHub Actions, CI workflow, Security workflow, Pre-commit quality hooks, Security Policy

### Community 139 - "Tests Conftest"
Cohesion: 0.50
Nodes (4): default_settings(), fixture, Shared test fixtures., runtime_state()

### Community 140 - "Is Blocked Ip"
Cohesion: 0.50
Nodes (4): _is_blocked_ip(), Return a human-readable reason if ``addr`` is in a blocked range., IPv4Address, IPv6Address

### Community 141 - "Auto Tab Ml Studio"
Cohesion: 0.50
Nodes (4): AutoTabML Studio, Bug Report issue template, Feature Request issue template, Usage Guide

### Community 142 - "Load Dataset Sources"
Cohesion: 0.50
Nodes (4): Dataset Intake page, Load Dataset Sources, Local dataset path: E:\Github\AutoTabML-Studio\datasets\sklearn\Diabetes\diabetes.csv, Local Path tab

### Community 143 - "Recent Prediction Jobs"
Cohesion: 0.50
Nodes (4): Model Source, Prediction Job Mode, Prediction Job Status, Recent Prediction Jobs

### Community 144 - "Public Dns"
Cohesion: 0.50
Nodes (4): public_dns(), fixture, MonkeyPatch, Make hostname resolution return a public address so respx can intercept.

### Community 146 - "Probe Flaml Import Error"
Cohesion: 0.67
Nodes (3): _probe_flaml_import_error(), Exception, Return the import-time failure when FLAML is unusable.

### Community 147 - "Clear Registry Cache"
Cohesion: 0.67
Nodes (3): _clear_registry_cache(), fixture, Drop the in-process registry cache before and after every test.

## Knowledge Gaps
- **127 isolated node(s):** `autotabml-studio`, `$schema`, `app`, `tests`, `**/__pycache__` (+122 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **15 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `ExecutionBackend` connect `Execution Backend` to `Build Backend`, `App Settings`, `Pycaret Service`, `Benchmark Config`, `Test Flaml Automl`, `Test Startup Colab Mcpdiagnostics`, `Benchmark Task Type`, `Get Allowed Providers`, `Test Hardening Smoke`, `Saved Model Metadata`, `Test Modeling Exception Handling`, `Flaml Auto Mlservice`, `Get Or Init State`, `Config Llmprovider`, `Validate Dataset`, `Experiment Result Bundle`, `Flaml Config`, `Build Input Spec`, `Validation Service`, `Trusted Artifacts`, `Settings Page`, `Test Profiling`, `Profiling Mode`, `Gemini Provider`, `Modeling Base`, `Validation Settings`, `Benchmark Result Bundle`, `Test Cli`, `Colab Mcpexecution Backend`, `Test Colab Mcp Backend`, `Test Packaging Metadata`, `Base Execution Backend`, `Notebook Page`, `Test Runtime State`, `Build Provider`, `Experiment Setup Summary`, `Resolve Default Model`, `Dummy Tracker`, `Resolve Task Type`, `Test Cli Output Encoding`?**
  _High betweenness centrality (0.097) - this node is a cross-community bridge._
- **Why does `DatasetInputSpec` connect `Dataset Input Spec` to `Prediction Base`, `Execution Backend`, `App Cli`, `Dataset Workspace`, `Url Loader`, `Test Hardening Smoke`, `Test Ingestion Utils`, `Ingestion Source Type`, `Load Raw Dataframe`, `Prediction Page`, `Get Or Init State`, `Validate Dataset`, `Prediction Loader`, `Build Input Spec`, `Profiling Mode`, `Batch Uci Runner`, `Model Source Type`, `List Available Uci Datasets`, `Ui Cache`, `Excel Loader`, `Htmltable Loader`, `Test Ui Cache`?**
  _High betweenness centrality (0.090) - this node is a cross-community bridge._
- **Why does `AppSettings` connect `App Settings` to `Build Backend`, `Prediction Base`, `Execution Backend`, `Test Flaml Automl`, `App Cli`, `Test Startup Colab Mcpdiagnostics`, `Tests Conftest`, `Test Hardening Smoke`, `Saved Model Metadata`, `Prediction Page`, `Experiment Page`, `Flaml Auto Mlservice`, `Get Or Init State`, `Ui Labels`, `Config Llmprovider`, `Flaml Config`, `Build Input Spec`, `Test Prediction`, `Validation Service`, `Settings Page`, `Run History Item`, `Profiling Mode`, `Batch Uci Runner`, `Sqlite Connector`, `Model Source Type`, `App Metadata Store`, `Ui Cache`, `Validation Settings`, `Test Cli`, `Colab Mcpexecution Backend`, `Test Colab Mcp Backend`, `Test Packaging Metadata`, `Test Ui Cache`, `Batch Run Item Record`, `Test Optional Dependency Probes`, `Test Artifact Manager`, `Resolve Task Type`, `Test Cli Output Encoding`?**
  _High betweenness centrality (0.078) - this node is a cross-community bridge._
- **Are the 169 inferred relationships involving `ExecutionBackend` (e.g. with `BaseExecutionBackend` and `ColabMCPExecutionBackend`) actually correct?**
  _`ExecutionBackend` has 169 INFERRED edges - model-reasoned connections that need verification._
- **Are the 81 inferred relationships involving `AppSettings` (e.g. with `ExecutionBackend` and `LLMProvider`) actually correct?**
  _`AppSettings` has 81 INFERRED edges - model-reasoned connections that need verification._
- **Are the 53 inferred relationships involving `DatasetInputSpec` (e.g. with `BaseLoader` and `CSVLoader`) actually correct?**
  _`DatasetInputSpec` has 53 INFERRED edges - model-reasoned connections that need verification._
- **Are the 131 inferred relationships involving `WorkspaceMode` (e.g. with `AppSettings` and `ArtifactSettings`) actually correct?**
  _`WorkspaceMode` has 131 INFERRED edges - model-reasoned connections that need verification._