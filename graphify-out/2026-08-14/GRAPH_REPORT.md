# Graph Report - .  (2026-08-14)

## Corpus Check
- Large corpus: 525 files · ~1,330,942 words. Semantic extraction will be expensive (many Claude tokens). Consider running on a subfolder.

## Summary
- 4447 nodes · 12713 edges · 190 communities (165 shown, 25 thin omitted)
- Extraction: 91% EXTRACTED · 9% INFERRED · 0% AMBIGUOUS · INFERRED: 1189 edges (avg confidence: 0.52)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- PyCaret Experiment Tests
- Benchmark Schemas
- CLI Application
- FLAML Automl Tests
- Profiling Tests
- FLAML Schemas
- Prediction Schemas
- Providers Base
- Benchmark Tests
- Config Models
- UCI Loader Tests
- Security Trusted Artifacts
- Dataset Workspace
- History Page
- Ingestion URL Loader
- Services Prediction Workflow
- Tracking MLflow Query
- Notebook Page
- Validation Tests
- Notebooks Generator
- UCI Real Datasets Tests
- Storage Models
- Prediction Tests
- Ingestion Base
- Pyrightconfig Configuration
- CLI Tests
- Validation GX Runner
- Registry Tests
- Backends Colab MCP Backend
- PyCaret Service
- Prediction Base
- Prediction Selectors
- Storage Repositories Tests
- Repositories Base
- Benchmark MLflow Tracking
- Batch UCI Runner Scripts
- CLI Smoke Tests
- UI Labels
- PyCaret Selectors
- PyCaret Summary
- UI Cache
- Architecture Documentation
- CLI Tests
- Startup Components
- PyCaret Errors
- Gpu Tests
- Ingestion UCI Loader
- Security Safe HTTP
- Storage SQLite Connector
- Registry Errors
- Tracking Description Generator
- Validation Schemas
- Prediction Page
- Experiment Workflow Service Tests
- Prediction Errors
- Validation Rules
- Base Components
- Providers Tests
- MLflow Query Batch Tests
- Settings Page
- Errors Tests
- Prediction Tests
- Observability Context
- Observability Metrics
- Registry Components
- Safe HTTP Async Tests
- Safe HTTP Tests
- Tracking Schemas
- Tracking Filters
- Tracking Tests
- Ingestion Metadata
- PyCaret MLflow Tracking
- Release Metadata
- Safe CSV Properties Tests
- Ingestion Tests
- Experiment Page
- Observability Metrics
- UA Analysis Artifacts
- Compare Page
- Repositories Batch Runs
- Tracking Tests
- Capture Screenshots Scripts
- Verify Optional Deps Scripts
- Observability Tests
- Prediction Loader
- Artifacts Manager
- Autorun Components
- CLI Tests
- CLI Tests
- Config Tests
- Tracking Compare Service
- Autorun Components
- Ingestion Kaggle Loader
- Security Safe HTTP
- Drift Components
- Tracking Errors
- Ingestion HTML Table Loader
- Security Tests
- Tracking Tests
- Explainability Components
- State Session
- Ingestion Async Tests
- Providers Anthropic Provider
- Repositories Saved Models
- E2E Local Smoke Tests
- Storage Migrations
- Repositories Projects
- Logging Config
- Providers Tests
- UI Cache
- Provenance Components
- Modeling Architecture Tests
- Predictions Page
- Registry Tests
- Tracking Tests
- Autotabml Social Preview Documentation
- Integration Optional Deps Tests
- UA Analysis Artifacts
- Base Components
- Providers Ollama Provider
- Code Of Conduct
- Autotabml Studio Architecture Documentation
- UA Analysis Artifacts
- Artifacts Manager
- Observability Tracing
- Providers Tests
- UA Analysis Artifacts
- Deployment Components
- Safe HTTP Tests
- FLAML Setup Runner
- Security Tests
- Benchmark Leaderboard Documentation
- History View Documentation
- Profiling Report Documentation
- Settings View Documentation
- Artifact Manager Tests
- Modeling Architecture Tests
- Experiment Lab Documentation
- Validation Summary Documentation
- CLI Tests
- CLI Tests
- UI Helpers Tests
- Observability Logging Setup
- Security Safe HTTP
- Compare View Documentation
- Dashboard Overview Documentation
- Dataset Intake Documentation
- Prediction Center Documentation
- Registry View Documentation
- Colab MCP Backend Tests
- UA Analysis Artifacts
- UA Analysis Artifacts
- PyCaret Init
- Conftest Tests
- UA Analysis Artifacts
- Security Safe HTTP
- Safe CSV Properties Tests
- Dataset Intake Documentation
- Prediction Center Documentation
- Release Notes V0 2 0
- Config Models
- PyCaret Compare Runner
- PyCaret Tune Runner
- Safe HTTP Tests
- UA Analysis Artifacts
- Init Components
- Init Components
- Registry Service
- Registry Service
- Registry Service
- Benchmark Leaderboard Documentation
- UA Analysis Artifacts
- UA Analysis Artifacts
- Readme Documentation
- Autotabml Social Preview Documentation
- Autotabml Studio Architecture Documentation
- Dependabot Configuration
- Bug Report Configuration
- Feature Request Configuration
- Pyproject Configuration
- Pre Commit Config Configuration
- Security Components
- UA Analysis Artifacts

## God Nodes (most connected - your core abstractions)
1. `ExecutionBackend` - 210 edges
2. `AppSettings` - 202 edges
3. `DatasetInputSpec` - 183 edges
4. `WorkspaceMode` - 161 edges
5. `IngestionSourceType` - 91 edges
6. `AppMetadataStore` - 83 edges
7. `log_exception()` - 77 edges
8. `ProfilingMode` - 66 edges
9. `LLMProvider` - 65 edges
10. `ValidationRuleConfig` - 58 edges

## Surprising Connections (you probably didn't know these)
- `Coverage Gate` --semantically_similar_to--> `Hermetic Test Strategy`  [INFERRED] [semantically similar]
  .github/workflows/ci.yml → docs/developer-guide.md
- `Local-First Architecture` --semantically_similar_to--> `Local-First Automated Machine Learning Workbench`  [INFERRED] [semantically similar]
  docs/architecture.md → README.md
- `Shared Service Layer Architecture` --semantically_similar_to--> `Shared UI and CLI Service Layer`  [INFERRED] [semantically similar]
  docs/architecture.md → README.md
- `TestBuildBackend` --uses--> `ColabMCPExecutionBackend`  [INFERRED]
  tests/test_colab_mcp_backend.py → app/backends/colab_mcp_backend.py
- `TestDefaultBackendIsColabMCP` --uses--> `ColabMCPExecutionBackend`  [INFERRED]
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

## Communities (190 total, 25 thin omitted)

### Community 0 - "PyCaret Experiment Tests"
Cohesion: 0.03
Nodes (99): ExecutionBackend, str, Execution backends – where ML jobs actually run., First-class workspace modes., WorkspaceMode, PyCaretExperimentSettings, Configuration for deeper PyCaret experiment workflows., benchmark_dataset() (+91 more)

### Community 1 - "Benchmark Schemas"
Cohesion: 0.04
Nodes (97): BaseService, Common modeling-service configuration and helpers., Return a configured tracker instance when MLflow tracking is enabled., BenchmarkArtifactsWriter, Path, Artifact generation for benchmark runs., Shared-path artifact writer for benchmark result bundles., Write benchmark artifacts to disk and return their paths. (+89 more)

### Community 2 - "CLI Application"
Cohesion: 0.05
Nodes (107): BackgroundJobService, Path, Submit and control one local training process at a time., _add_prediction_model_source_args(), _build_flaml_service(), _build_prediction_request_kwargs(), _build_prediction_service(), _build_pycaret_service() (+99 more)

### Community 3 - "FLAML Automl Tests"
Cohesion: 0.05
Nodes (42): FlamlSettings, Configuration for FLAML AutoML workflows., FlamlConfig, Top-level FLAML AutoML configuration., FlamlAutoMLService, metric_sort_direction(), Path, Return the expected ordering direction for a metric name. (+34 more)

### Community 4 - "Profiling Tests"
Cohesion: 0.05
Nodes (57): Profiling artifact generation helpers., ProfilingError, ProfilingSetupError, Exception, Custom exceptions for the profiling layer., Raised when the profiling library cannot be initialized., Base exception for profiling failures., Automated profiling layer for AutoTabML Studio. (+49 more)

### Community 5 - "FLAML Schemas"
Cohesion: 0.05
Nodes (61): FlamlArtifactsWriter, Path, Artifact generation for FLAML AutoML runs., Shared-path artifact writer for FLAML result bundles., Write FLAML artifacts to disk and return their paths., write_flaml_artifacts(), FlamlAutoMLError, FlamlConfigurationError (+53 more)

### Community 6 - "Prediction Schemas"
Cohesion: 0.06
Nodes (64): DataFrame, Path, Artifact generation for prediction jobs., Persist prediction artifacts and return their paths., _render_markdown(), write_prediction_artifacts(), Prediction service facade., build_batch_prediction_result() (+56 more)

### Community 7 - "Providers Base"
Cohesion: 0.05
Nodes (37): LLMProvider, Enum, Enumerations for AutoTabML Studio configuration., Supported LLM providers., _section_provider(), Anthropic provider — model discovery and text generation via the official SDK.…, BaseProvider, ModelItem (+29 more)

### Community 8 - "Benchmark Tests"
Cohesion: 0.06
Nodes (47): BenchmarkSettings, Configuration for baseline model benchmarking., LazyPredictBenchmarkService, Path, Benchmark service backed by LazyClassifier/LazyRegressor., MLflowBenchmarkTracker, Lightweight MLflow tracking wrapper for benchmark runs., Backward-compatible benchmark tracker entrypoint. (+39 more)

### Community 9 - "Config Models"
Cohesion: 0.05
Nodes (48): BaseExecutionBackend, Any, Abstract execution backend interface., Common interface for all execution backends., Check that the backend is reachable / properly configured., Prepare any runtime context needed before a job runs., Execute a job payload and return results. Concrete implementations will be…, Colab MCP execution backend – connects to Google Colab via MCP. This backend… (+40 more)

### Community 10 - "UCI Loader Tests"
Cohesion: 0.08
Nodes (25): load_dataset(), Public entry point for full dataset loading., DatasetInputSpec, model_validator, Canonical input contract for all supported ingestion paths., Return the human-readable locator for lineage and logging., Validate source-specific input requirements., Fetch a dataset from the UCI ML Repository by ID or name. (+17 more)

### Community 11 - "Security Trusted Artifacts"
Cohesion: 0.07
Nodes (56): discover_saved_benchmark_models(), load_saved_benchmark_model(), load_saved_benchmark_model_metadata_file(), Path, Trusted benchmark model discovery and loading helpers., Parse a benchmark saved-model metadata file, returning None when invalid., Discover trusted benchmark models from checksum-backed metadata sidecars., Load a trusted benchmark model from a skops artifact. (+48 more)

### Community 12 - "Dataset Workspace"
Cohesion: 0.07
Nodes (48): AutoTabML Studio – Streamlit entry point., build_local_path_input_spec(), build_url_input_spec(), _clear_dataset_results(), clear_loaded_datasets(), _dataset_identity_key(), get_active_dataset_name(), get_active_loaded_dataset() (+40 more)

### Community 13 - "History Page"
Cohesion: 0.07
Nodes (54): Streamlit page for LazyPredict benchmark execution and results., Resolve a LazyPredict model name to its sklearn-compatible estimator class., Render controls to retrain and save the best model from benchmark results., render_benchmark_page(), _render_result_bundle(), _render_save_best_model(), _resolve_estimator_class(), Streamlit page for job and dataset run history. (+46 more)

### Community 14 - "Ingestion URL Loader"
Cohesion: 0.09
Nodes (46): Base abstractions shared by all ingestion loaders., CSV and delimiter-aware text loader., In-memory pandas DataFrame loader., ParseFailureError, Custom exceptions for dataset ingestion., Raised when a source type or locator is unsupported., Raised when a remote resource cannot be reached or inspected safely., Raised when a source is reachable but could not be parsed. (+38 more)

### Community 15 - "Services Prediction Workflow"
Cohesion: 0.07
Nodes (23): Page-layer services that keep Streamlit pages focused on rendering., ModelTestingEvaluation, ModelTestingRunResult, ModelTestingSelection, PredictionExecutionConfig, PredictionWorkflowService, Any, DataFrame (+15 more)

### Community 16 - "Tracking MLflow Query"
Cohesion: 0.08
Nodes (55): Raised when the MLflow model registry backend is not available., RegistryUnavailableError, Normalized MLflow model version metadata., RegistryVersionSummary, create_model_version(), create_registered_model(), delete_model_alias(), delete_model_version_tag() (+47 more)

### Community 17 - "Notebook Page"
Cohesion: 0.08
Nodes (48): Guided Auto Run page., _render_active_job(), render_autorun_page(), _detect_completed_steps(), _friendly_job_name(), _load_example_dataset(), Dashboard page – professional workspace home., Return the highest *completed* step number (0 = nothing done yet). Heuristic… (+40 more)

### Community 18 - "Validation Tests"
Cohesion: 0.08
Nodes (26): User-facing configuration for a validation run., ValidationRuleConfig, constant_df(), duplicate_rows_df(), empty_df(), good_df(), null_heavy_df(), DataFrame (+18 more)

### Community 19 - "Notebooks Generator"
Cohesion: 0.09
Nodes (38): _benchmark_cells(), _experiment_cells(), _flaml_cells(), generate_job_notebook(), _json_literal(), _json_string(), _markdown_cell(), NotebookGenerationError (+30 more)

### Community 20 - "UCI Real Datasets Tests"
Cohesion: 0.08
Nodes (36): profile_dataset(), DataFrame, Path, Convenience function: profile a DataFrame and optionally write artifacts., Path, Convenience function: validate and optionally write artifacts. Returns…, validate_dataset(), Exercise non-trivial rule config — uniqueness, range, category checks. (+28 more)

### Community 21 - "Storage Models"
Cohesion: 0.10
Nodes (35): main(), Path, Subprocess entry point for persistent Auto Run jobs., _update(), Persistent local subprocess jobs for long-running AutoML work., Local metadata storage exports., AppJobStatus, AppJobType (+27 more)

### Community 22 - "Prediction Tests"
Cohesion: 0.07
Nodes (27): DataFrame, Resolve one batch request into a normalized dataframe and source label., resolve_batch_dataframe(), PredictionHistoryError, Raised when prediction history cannot be persisted or queried., PredictionHistoryStore, Path, Lightweight local prediction-history storage. (+19 more)

### Community 23 - "Ingestion Base"
Cohesion: 0.07
Nodes (33): BaseLoader, Any, DataFrame, Return a raw DataFrame and source details before normalization., Async counterpart of :meth:`load_raw_dataframe`. Loaders with native async I/O…, Common load lifecycle for all dataset sources., Load, normalize, and enrich a dataset from the supplied input spec., Async counterpart of :meth:`load` for I/O-heavy loader implementations. (+25 more)

### Community 24 - "Pyrightconfig Configuration"
Cohesion: 0.04
Nodes (45): exclude, ignore, include, pythonPlatform, pythonVersion, reportArgumentType, reportAttributeAccessIssue, reportCallIssue (+37 more)

### Community 25 - "CLI Tests"
Cohesion: 0.07
Nodes (12): AppSettings, model_validator, Top-level application configuration with safe defaults., Return the verified stable fallback model id for the given (or current)…, Direct functional tests for cmd_predict_history., Direct functional tests for cmd_registry_show., Tests that exercise argparse parsing through main()., TestCmdPredictHistory (+4 more)

### Community 26 - "Validation GX Runner"
Cohesion: 0.07
Nodes (37): Exception, Custom exceptions for the validation layer., Raised when the validation infrastructure cannot be initialized., Raised when a validation rule configuration is invalid., Base exception for validation failures., RuleConfigError, ValidationError, ValidationSetupError (+29 more)

### Community 27 - "Registry Tests"
Cohesion: 0.12
Nodes (16): Execute a promotion action on a model version., High-level service for MLflow model registry operations., Get a single registered model by name., List all versions for a registered model., RegistryService, PromotionRequest, Request to promote a model version., _FakeModelVersion (+8 more)

### Community 28 - "Backends Colab MCP Backend"
Cohesion: 0.09
Nodes (15): ColabMCPExecutionBackend, Any, Execute a job by calling Colab MCP tools. ``job_payload`` must include a…, Call ``open_colab_browser_connection`` to link to a Colab notebook., Return the currently available MCP tool names., Tear down the MCP session and server subprocess., Execution backend that delegates work to a Google Colab runtime via MCP., Check that ``uvx`` is installed and the MCP SDK is importable. (+7 more)

### Community 29 - "PyCaret Service"
Cohesion: 0.16
Nodes (15): PyCaretExecutionError, Raised when a PyCaret operation fails., list_available_metrics(), Return normalized metric rows from the active experiment., ExperimentResultBundle, ModelSelectionSpec, Stable user-facing model selection reference., Complete experiment package for UI, CLI, tracking, and persistence. (+7 more)

### Community 30 - "Prediction Base"
Cohesion: 0.09
Nodes (23): DriftBaseline, Open a span named ``name`` and yield a :class:`SpanLike`. When OpenTelemetry is…, start_span(), BasePredictionService, PredictionService, ABC, Path, Abstract prediction service contract. (+15 more)

### Community 31 - "Prediction Selectors"
Cohesion: 0.10
Nodes (29): ModelLoader, ABC, Model-loading wrappers for prediction flows., Abstract model loader contract., Return True when the loader supports the requested source type., AvailableModelReference, Normalized model reference shown in discovery and selection UIs., build_mlflow_registered_model_uri() (+21 more)

### Community 32 - "Storage Repositories Tests"
Cohesion: 0.10
Nodes (18): AppMetadataStore, Path, Repository facade for local workspace metadata. The store composes the per-…, _dataset(), _job(), _now(), datetime, fixture (+10 more)

### Community 33 - "Repositories Base"
Cohesion: 0.08
Nodes (16): DatasetRecord, Locally persisted dataset lineage record., BaseRepository, datetime, Path, Shared base utilities for storage repositories., Bundle of shared dependencies passed to every repository. The context holds the…, Common access patterns shared by every domain repository. Subclasses focus on… (+8 more)

### Community 34 - "Benchmark MLflow Tracking"
Cohesion: 0.08
Nodes (27): get_mlflow_module(), is_mlflow_available(), mlflow_exception_types(), BaseException, Shared base classes for modeling services, trackers, and artifact writers., Return True when mlflow is importable., Import and return the mlflow module., Return common MLflow exception types plus generic boundary failures. (+19 more)

### Community 35 - "Batch UCI Runner Scripts"
Cohesion: 0.09
Nodes (34): Counter, Monotonically increasing counter., BatchRunItemRecord, Tracks a single dataset within a batch run through validate/profile/benchmark., build_200_dataset_list(), main(), Batch UCI dataset runner – runs validate → profile → benchmark for 200 NEW UCI…, Combine the new UCI datasets + extra re-framed datasets to get 200. (+26 more)

### Community 36 - "CLI Smoke Tests"
Cohesion: 0.10
Nodes (25): gather_with_concurrency(), Any, BaseException, T, Small async concurrency helpers used across the app. These helpers exist so…, Run awaitables concurrently with at most ``limit`` running at a time. ``limit``…, Run a blocking ``func`` over many argument batches concurrently in threads.…, to_thread_many() (+17 more)

### Community 37 - "UI Labels"
Cohesion: 0.09
Nodes (30): Path, Streamlit page for browsing all saved models with details., Render an expander card for a PyCaret experiment model., Render an expander card for a benchmark-saved model., Render an expander card for an MLflow registry model., Render an expander card for a FLAML-saved model., _render_benchmark_model_card(), _render_deployment_export() (+22 more)

### Community 38 - "PyCaret Selectors"
Cohesion: 0.09
Nodes (29): infer_task_type(), Series, Infer a benchmark task type from the target values., Validate a target series for the requested benchmark task., validate_target(), generate_evaluation_plots(), pushd(), Path (+21 more)

### Community 39 - "PyCaret Summary"
Cohesion: 0.11
Nodes (32): PyCaretConfigurationError, Raised when experiment configuration is invalid., add_custom_metric(), Metric catalog inspection and safe custom metric registration., Register a custom metric on the active experiment., Remove a previously registered custom metric., remove_custom_metric(), metric_sort_direction() (+24 more)

### Community 40 - "UI Cache"
Cohesion: 0.10
Nodes (30): _build_flaml_run_key(), _metric_options_for_task(), Streamlit page for the FLAML AutoML workflow., render_flaml_automl_page(), _render_flaml_results(), _get_experiment_workflow_service_resource(), get_flaml_automl_service(), _get_flaml_automl_service_resource() (+22 more)

### Community 41 - "Architecture Documentation"
Cohesion: 0.06
Nodes (35): Changelog, AutoTabML Container Service, Docker Compose Stack, Architecture Guide, Architecture Diagram HTML, Local-First Architecture, Reproducible Runs, Security Model (+27 more)

### Community 42 - "CLI Tests"
Cohesion: 0.07
Nodes (21): ProfilingMode, Profiling report modes., Tests for CLI helpers and default wiring., Direct functional tests for cmd_history_show., Direct functional tests for cmd_registry_register., Direct functional tests for cmd_registry_promote., Direct functional tests for cmd_compare_runs., Direct functional tests for cmd_experiment_run. (+13 more)

### Community 43 - "Startup Components"
Cohesion: 0.10
Nodes (18): initialize_local_runtime(), BaseModel, Startup checks and local-runtime initialization for AutoTabML Studio., Report missing prerequisites for the Colab MCP backend., One startup diagnostic item., Initialization result for local app runtime resources., Prepare conservative local runtime resources and collect actionable diagnostics., StartupIssue (+10 more)

### Community 44 - "PyCaret Errors"
Cohesion: 0.10
Nodes (27): Cross-cutting error-handling utilities. This module provides: *…, Exception, PyCaretDependencyError, PyCaretExperimentError, PyCaretTargetError, PyCaretTrackingError, Custom exceptions for the PyCaret experiment layer., Raised when the optional PyCaret dependency is unavailable. (+19 more)

### Community 45 - "Gpu Tests"
Cohesion: 0.10
Nodes (21): cuda_device_name(), cuda_summary(), _driver_probe(), is_cuda_available(), CUDA / GPU detection utilities for AutoTabML Studio., Lightweight probe for the NVIDIA driver library without importing torch. Loads…, Return True when a CUDA-capable GPU is reachable from the current Python…, Return the name of the first CUDA device, or None. (+13 more)

### Community 46 - "Ingestion UCI Loader"
Cohesion: 0.10
Nodes (25): EmptyDatasetError, IngestionError, Exception, Raised when a loaded dataset has no usable rows or columns., Base exception for user-facing ingestion failures., normalize_duplicate_column_names(), normalize_to_pandas(), DataFrame (+17 more)

### Community 47 - "Security Safe HTTP"
Cohesion: 0.15
Nodes (30): _attempt_download_to_path(), _attempt_download_to_path_async(), _attempt_fetch(), _attempt_fetch_async(), _check_advertised_size(), _check_response_headers(), _normalize_content_type(), Path (+22 more)

### Community 48 - "Storage SQLite Connector"
Cohesion: 0.10
Nodes (15): Connection, Path, T, Reusable SQLite connector with safe defaults for local metadata storage., Open SQLite connections with consistent PRAGMAs and atomic write helpers., Yield a configured connection for read operations., Run a read-only callback using a configured connection., Run a write callback in an atomic transaction with lock retries. (+7 more)

### Community 49 - "Registry Errors"
Cohesion: 0.12
Nodes (24): ModelNotFoundError, PromotionError, Exception, Custom exceptions for the model registry layer., Raised when a registered model cannot be found., Raised when a model version cannot be found., Raised when a promotion action cannot be completed., Base exception for registry failures. (+16 more)

### Community 50 - "Tracking Description Generator"
Cohesion: 0.11
Nodes (15): _build_footer(), _build_llm_prompt(), generate_llm_description(), generate_template_description(), _generic_template(), _job_icon(), Any, Generate professional run descriptions for job runs. Two modes: - **Template**… (+7 more)

### Community 51 - "Validation Schemas"
Cohesion: 0.12
Nodes (21): Path, Produce validation artifacts: JSON summary, Markdown report., Write validation artifacts to disk and return the bundle., _render_markdown(), write_artifacts(), Data validation layer for AutoTabML Studio., CheckSeverity, BaseModel (+13 more)

### Community 52 - "Prediction Page"
Cohesion: 0.14
Nodes (23): Persist one uploaded file to the app temp area and return an input spec., uploaded_file_to_input_spec(), _prediction_task_type_input(), Streamlit page for local-first prediction workflows., Render the default saved-model browser. Advanced sources (manual path, MLflow)…, Render per-column inputs and return the row payload dict., _render_artifacts(), _render_batch_panel() (+15 more)

### Community 53 - "Experiment Workflow Service Tests"
Cohesion: 0.16
Nodes (14): ExperimentFormValues, ExperimentWorkflowService, Page-facing workflow service for the Train & Tune (PyCaret) page., User-entered form values from the experiment page., Verdict for a tuned-vs-baseline metric comparison., Encapsulate Train & Tune page orchestration., TuningInterpretation, _form_values() (+6 more)

### Community 54 - "Prediction Errors"
Cohesion: 0.15
Nodes (25): ModelDiscoveryError, PredictionArtifactError, PredictionError, PredictionScoringError, PredictionValidationError, Exception, Prediction-layer errors for inference workflows., Raised when a model source cannot be discovered or resolved cleanly. (+17 more)

### Community 55 - "Validation Rules"
Cohesion: 0.28
Nodes (27): _check_allowed_categories(), _check_constant_columns(), _check_dtype_summary(), _check_duplicate_column_names(), _check_duplicate_rows(), _check_fully_null_columns(), _check_id_columns(), _check_leakage_heuristics() (+19 more)

### Community 56 - "Base Components"
Cohesion: 0.13
Nodes (14): BaseTracker, Any, Logger, Shared MLflow tracking lifecycle for modeling bundles., Log params, metrics, and artifacts for one modeling bundle., Return True if the subclass-specific MLflow boundary is available., Return the subclass-specific MLflow module handle., Return the operation label used for structured error logging. (+6 more)

### Community 57 - "Providers Tests"
Cohesion: 0.11
Nodes (7): GeminiProvider, Any, OpenAIProvider, Any, As of v0.2.0 the Gemini provider uses the official google-genai SDK which…, TestAllowedProviders, TestModelNormalization

### Community 58 - "MLflow Query Batch Tests"
Cohesion: 0.21
Nodes (19): list_registered_models(), Return all registered models from the MLflow model registry. Uses paginated…, _clear_cache(), _CountingClient, _FakeRegisteredModel, _FakeVersion, _patch_client(), fixture (+11 more)

### Community 59 - "Settings Page"
Cohesion: 0.13
Nodes (25): _find_uvx(), Return the path to ``uvx`` if it is installed, else *None*., Settings / Runtime configuration page for Streamlit., Lightweight GPU status for the Essentials tab — no controls., Simple on/off for run descriptions on the Essentials tab., Render the full Settings page., Always-visible privacy reminder at the top of Essentials., render_settings_page() (+17 more)

### Community 60 - "Errors Tests"
Cohesion: 0.11
Nodes (23): AutoTabMLError, log_and_wrap(), log_exception(), Any, BaseException, Exception, Logger, Opt-in umbrella base class for application-level domain errors. (+15 more)

### Community 61 - "Prediction Tests"
Cohesion: 0.16
Nodes (14): Stable saved-model metadata contract for future prediction flows., SavedModelMetadata, LocalPyCaretModelLoader, Path, Load local saved PyCaret model artifacts., load_saved_model_metadata_file(), Parse a saved-model metadata file, returning None when invalid., Path (+6 more)

### Community 62 - "Observability Context"
Cohesion: 0.13
Nodes (22): bind_context(), correlation_scope(), current_context(), _empty_context_default(), new_correlation_id(), Any, Correlation context for log/metric/trace records. A small wrapper around…, Return a *copy* of the current correlation context mapping. (+14 more)

### Community 63 - "Observability Metrics"
Cohesion: 0.11
Nodes (14): get_metrics_backend(), InMemoryMetricsBackend, MetricsBackend, NoopMetricsBackend, Protocol, Install ``backend`` as the active sink and return the previous one., Return the currently installed metrics backend., Strategy interface implemented by metric exporters. (+6 more)

### Community 64 - "Registry Components"
Cohesion: 0.13
Nodes (17): build_streamlit_navigation(), default_page_label(), get_nav_sections(), get_page_by_label(), get_page_registry(), PageSpec, Central registry for Streamlit page navigation and rendering., Declarative Streamlit page registration. (+9 more)

### Community 65 - "Safe HTTP Async Tests"
Cohesion: 0.15
Nodes (16): BaseException, Async counterpart of :func:`safe_fetch` with identical retry semantics., Fetch many URLs concurrently with bounded parallelism. Per-URL guards (SSRF,…, safe_fetch_async(), safe_fetch_many_async(), public_dns(), asyncio, fixture (+8 more)

### Community 66 - "Safe HTTP Tests"
Cohesion: 0.14
Nodes (7): Fetch ``url`` using the SSRF-resistant, bounded HTTP client. Retries are…, safe_fetch(), mock, MonkeyPatch, parametrize, TestRedirectGuards, TestSSRFHostBlocking

### Community 67 - "Tracking Schemas"
Cohesion: 0.13
Nodes (20): Path, Artifact generation for comparison bundles., Write comparison artifacts and return their paths., _render_markdown(), write_comparison_artifacts(), Run history, comparison, and MLflow query layer for AutoTabML Studio., ComparisonBundle, Enum (+12 more)

### Community 68 - "Tracking Filters"
Cohesion: 0.15
Nodes (17): build_mlflow_filter_string(), Enum, str, Filtering and sorting helpers for run history queries., Fields available for sorting run history., Ascending or descending sort., Declarative filter for run history queries., Build an MLflow-compatible filter string from a RunHistoryFilter. (+9 more)

### Community 69 - "Tracking Tests"
Cohesion: 0.18
Nodes (11): HistoryService, Fetch extended detail for a single run., Resolve experiment name(s) to ids, building a name lookup map., High-level service for querying and inspecting run history., _FakeMLflowExperiment, _FakeMLflowRun, _patch_mlflow_query(), Patch mlflow_query module functions for testing. (+3 more)

### Community 70 - "Ingestion Metadata"
Cohesion: 0.13
Nodes (18): compute_content_hash(), compute_schema_hash(), detect_file_extension(), extract_dataset_metadata(), Any, DataFrame, Metadata extraction and deterministic hashing for ingested datasets., Convert values to a stable, JSON-serializable representation. (+10 more)

### Community 71 - "PyCaret MLflow Tracking"
Cohesion: 0.12
Nodes (16): _build_metrics(), _build_params(), _build_run_name(), _get_mlflow_module(), is_mlflow_available(), _log_artifacts(), _mlflow_exception_types(), MLflowExperimentTracker (+8 more)

### Community 72 - "Release Metadata"
Cohesion: 0.15
Nodes (18): check_public_release_metadata(), _collect_contacts(), _has_license_metadata(), load_project_metadata(), main(), _normalized(), Any, Path (+10 more)

### Community 73 - "Safe CSV Properties Tests"
Cohesion: 0.14
Nodes (19): dataframe_to_safe_csv(), Any, DataFrame, Safe CSV export helpers. Excel and similar spreadsheet tools can evaluate cells…, Return a copy safe for CSV export. Sanitizes string cells, column labels, index…, Serialize ``dataframe`` to CSV with formula-safe cells and strict quoting., sanitize_csv_dataframe(), _sanitize_csv_scalar() (+11 more)

### Community 74 - "Ingestion Tests"
Cohesion: 0.16
Nodes (10): ExcelLoader, Any, DataFrame, Path, Load local or remote Excel workbooks., MonkeyPatch, Path, TestBoundedCSVReading (+2 more)

### Community 75 - "Experiment Page"
Cohesion: 0.13
Nodes (21): pycaret_install_guidance(), Return a user-facing installation hint for environments without PyCaret., build_experiment_run_key(), default_compare_metric_for_task(), default_plot_ids_for_task(), default_tracking_mode(), default_tune_metric_for_task(), Streamlit page for the Train & Tune workflow (PyCaret). (+13 more)

### Community 76 - "Observability Metrics"
Cohesion: 0.12
Nodes (15): Gauge, Histogram, _merge_labels(), _Metric, Any, Pluggable metrics façade. Three primitive types are exposed – :class:`Counter`,…, Shared base – holds a metric name and forwards to the active backend., Point-in-time numeric value. (+7 more)

### Community 77 - "UA Analysis Artifacts"
Cohesion: 0.08
Nodes (18): assignedIds, changed, configLayer, dataLayer, docsLayer, fs, graph, graphPath (+10 more)

### Community 78 - "Compare Page"
Cohesion: 0.13
Nodes (21): _find_model_column(), _find_score_column(), _leaderboard_rows_to_df(), _load_leaderboard(), DataFrame, Streamlit page for comparing algorithm performance on a dataset., Try to load a leaderboard from the job's artifact paths., Convert a list of leaderboard row dicts to a clean DataFrame. (+13 more)

### Community 79 - "Repositories Batch Runs"
Cohesion: 0.13
Nodes (8): BatchRunRecord, Tracks an overall batch execution run., Connection, T, BatchRunRepository, Connection, Row, CRUD operations for batch runs and their items.

### Community 80 - "Tracking Tests"
Cohesion: 0.19
Nodes (9): ComparisonService, Produce structured comparisons between two runs., Return a compact one-line summary of a run., run_summary_line(), _make_run(), Path, TestComparisonArtifacts, TestComparisonService (+1 more)

### Community 81 - "Capture Screenshots Scripts"
Cohesion: 0.18
Nodes (22): Page, capture(), choose_streamlit_option(), main(), navigate_to_page(), Path, Automated screenshot capture for AutoTabML Studio using Playwright. Launches a…, Navigate to Profiling and run it. (+14 more)

### Community 82 - "Verify Optional Deps Scripts"
Cohesion: 0.20
Nodes (16): is_lazypredict_available(), Return True if lazypredict is importable., main(), _make_iris_df(), DataFrame, _result(), _section(), verify_gx() (+8 more)

### Community 83 - "Observability Tests"
Cohesion: 0.13
Nodes (21): clear_context(), Reset the correlation context to an empty mapping., JsonFormatter, Render :class:`LogRecord` as a single-line JSON document. The output is stable…, Decorator that wraps a function call in :func:`start_span`., traced(), F, _make_record() (+13 more)

### Community 84 - "Prediction Loader"
Cohesion: 0.13
Nodes (11): ModelLoadError, Raised when a prediction model cannot be loaded., Raised when a model artifact fails trust-boundary validation., TrustedArtifactError, MLflowModelLoader, Any, Load MLflow-backed pyfunc models., BaseTrustedArtifactError (+3 more)

### Community 85 - "Artifacts Manager"
Cohesion: 0.20
Nodes (6): LocalArtifactManager, DataFrame, Path, Create, write, and conservatively clean local workspace artifacts., Return a filesystem-safe stem for generated artifact filenames., safe_artifact_stem()

### Community 86 - "Autorun Components"
Cohesion: 0.15
Nodes (19): AutoRunMode, AutoRunPlan, AutoRunResult, BaseModel, Enum, str, Guided, engine-aware Auto Run planning and execution., evaluate_model() (+11 more)

### Community 87 - "CLI Tests"
Cohesion: 0.13
Nodes (7): Regression: experiment-tune/evaluate/save must catch setup errors cleanly., Direct functional tests for cmd_history_list., Direct functional tests for cmd_profile., TestCmdBenchmark, TestCmdHistoryList, TestCmdProfile, TestExperimentCommandsErrorHandling

### Community 88 - "CLI Tests"
Cohesion: 0.17
Nodes (7): _build_input_spec(), Build an ingestion input spec from a CLI dataset locator., Path, Happy-path coverage for experiment-tune, experiment-evaluate, experiment-save., TestBuildInputSpec, TestExperimentCommandsHappyPath, TestCLIUCILocator

### Community 89 - "Config Tests"
Cohesion: 0.16
Nodes (7): Persist settings to ~/.autotabml/settings.json (secrets excluded)., save_settings(), Save non-secret settings to disk., Path, As of v0.2.0 the project builds with hatchling; the profiling extra no longer…, TestPackagingMetadata, TestSettingsPersistence

### Community 90 - "Tracking Compare Service"
Cohesion: 0.17
Nodes (15): _check_comparability(), _compute_config_differences(), _compute_metric_deltas(), Side-by-side run comparison service., Compare two runs and return a structured bundle with deltas and warnings., Apply filters that cannot be expressed in MLflow filter syntax., List runs with optional filtering and sorting., _sort_runs() (+7 more)

### Community 91 - "Autorun Components"
Cohesion: 0.19
Nodes (17): AutoRunConfig, plan_auto_run(), Any, DataFrame, Path, Run FLAML on a training split and evaluate on an untouched holdout., run_auto_run(), suggest_targets() (+9 more)

### Community 92 - "Ingestion Kaggle Loader"
Cohesion: 0.19
Nodes (10): CSVLoader, Any, DataFrame, Path, Load local or remote delimited tabular files into pandas., KaggleLoader, Any, DataFrame (+2 more)

### Community 93 - "Security Safe HTTP"
Cohesion: 0.21
Nodes (17): Convenience wrapper returning decoded text., Raised when a remote response exceeds the configured byte cap., Raised when a response declares a disallowed Content-Type., Async convenience wrapper returning decoded text., Tunable knobs for ``safe_fetch``., ResponseTooLargeError, safe_fetch_text(), safe_fetch_text_async() (+9 more)

### Community 94 - "Drift Components"
Cohesion: 0.22
Nodes (17): build_drift_baseline(), _categorical_proportions(), compare_drift(), DriftLevel, DriftReport, FeatureBaseline, FeatureDrift, _level() (+9 more)

### Community 95 - "Tracking Errors"
Cohesion: 0.14
Nodes (16): ComparisonError, ExperimentNotFoundError, Exception, Custom exceptions for the tracking and history layer., Raised when MLflow tracking is not available or configured., Raised when a requested run cannot be found., Raised when a requested MLflow experiment cannot be found., Raised when a comparison cannot be completed. (+8 more)

### Community 96 - "Ingestion HTML Table Loader"
Cohesion: 0.24
Nodes (6): HTMLTableLoader, Any, DataFrame, Extract HTML tables from a URL and return one table as a DataFrame., Any, DataFrame

### Community 97 - "Security Tests"
Cohesion: 0.17
Nodes (7): LogRecord, Replace anything that looks like an API key or credential in free text., redact_key_in_text(), Tests for security / masking helpers., TestLoggingRedaction, TestRedactKeyInText, TestSafeErrorMessage

### Community 98 - "Tracking Tests"
Cohesion: 0.17
Nodes (7): _extract_dataset_name(), _extract_primary_metric(), _infer_run_type(), _normalize_run(), _safe_run_status(), TestFlamlRunType, TestMLflowNormalization

### Community 99 - "Explainability Components"
Cohesion: 0.24
Nodes (15): explain_global(), explain_prediction(), FeatureContribution, ModelExplanation, Any, BaseModel, DataFrame, Series (+7 more)

### Community 100 - "State Session"
Cohesion: 0.17
Nodes (7): Any, setter, Mutable runtime state for the current user session., Return the session-only API key for the requested provider., Store a provider-specific API key in session memory only., Clear fetched models and selection when execution context changes., RuntimeState

### Community 101 - "Ingestion Async Tests"
Cohesion: 0.26
Nodes (8): load_dataset_async(), Async public entry point for full dataset loading., asyncio, mock, MonkeyPatch, Path, TestAsyncIngestionFactory, TestAsyncUCIHelpers

### Community 102 - "Providers Anthropic Provider"
Cohesion: 0.17
Nodes (5): AnthropicProvider, Any, Verify that each provider is now backed by the official SDK client. The SDKs…, v0.2.0: ``httpx`` is no longer used by any LLM provider. (``httpx`` is still…, TestOfficialSDKIntegration

### Community 103 - "Repositories Saved Models"
Cohesion: 0.20
Nodes (8): BaseModel, Local saved-model metadata kept outside MLflow registry state., SavedLocalModelRecord, Connection, Path, Row, CRUD operations for saved local model rows., SavedModelRepository

### Community 104 - "E2E Local Smoke Tests"
Cohesion: 0.15
Nodes (11): _classification_csv(), fixture, parametrize, Path, Verify the startup MLflow URI validation catches bad config., Write a small but realistic classification dataset (no network)., Verify the real startup path creates all expected resources., Common setup: settings, metadata store, loaded dataset. (+3 more)

### Community 105 - "Storage Migrations"
Cohesion: 0.31
Nodes (13): apply_migrations(), _detect_legacy_version(), _ensure_version_table(), _get_applied_versions(), Migration, Connection, Incremental SQLite schema migrations for the local metadata store., Create the version table and apply any pending schema migrations. (+5 more)

### Community 106 - "Repositories Projects"
Cohesion: 0.21
Nodes (7): ProjectRecord, Local project metadata for workspace navigation convenience., ProjectRepository, Connection, Row, CRUD operations for project rows., Upsert using an existing connection (used during initial migration).

### Community 107 - "Logging Config"
Cohesion: 0.21
Nodes (11): configure_logging(), _configure_noisy_dependency_loggers(), Logging configuration for AutoTabML Studio. Thin compatibility shim over…, Reduce non-actionable third-party log noise in normal app and batch runs., Configure structured logging to stderr (12-factor compliant). Honors the…, configure_observability_logging(), Configure the root logger for structured/observable output. Parameters…, Plain-text formatter that scrubs obvious secrets from the message. (+3 more)

### Community 108 - "Providers Tests"
Cohesion: 0.23
Nodes (9): _fetch_models(), Run an async coroutine from sync Streamlit code. Streamlit may already have a…, Build a provider instance and fetch models, updating state., _run_async(), _section_models(), Pick the best default from a fetched model list. 1. If the hardcoded default…, resolve_default_model(), _make_items() (+1 more)

### Community 109 - "UI Cache"
Cohesion: 0.24
Nodes (12): _dataset_cache_signature(), _load_dataset_cached(), load_dataset_for_ui(), Any, Return a stable cache key payload for dataset loads when possible., Load a dataset with a Streamlit cache when the input is cacheable., MonkeyPatch, Path (+4 more)

### Community 110 - "Provenance Components"
Cohesion: 0.32
Nodes (12): build_provenance(), _engine_packages(), _git_dirty(), _git_value(), _json_safe(), _package_version(), Any, Path (+4 more)

### Community 112 - "Predictions Page"
Cohesion: 0.24
Nodes (10): Streamlit page for model testing – evaluate a trained model on real-world data., Show evaluation metrics when ground-truth labels are available., _render_evaluation_metrics(), render_model_testing_page(), Unified Predictions page — combines Score New Data and Test & Evaluate., Delegate to the existing prediction page, skipping the outer title., Delegate to the existing Model Testing page, skipping the outer title., _render_evaluate_tab() (+2 more)

### Community 113 - "Registry Tests"
Cohesion: 0.17
Nodes (7): List all registered models., BaseModel, Normalized MLflow registered model metadata., RegistryModelSummary, _read_registry_cache(), _write_registry_cache(), TestRegistrySchemas

### Community 114 - "Tracking Tests"
Cohesion: 0.29
Nodes (7): BaseModel, Sort specification for run history queries., RunHistorySort, _build_order_by(), Translate a :class:`RunHistorySort` into MLflow ``order_by`` clauses. Only…, Validates that history sorting is correct and not partial-dataset., TestHistoryServiceSorting

### Community 115 - "Autotabml Social Preview Documentation"
Cohesion: 0.17
Nodes (12): AutoTabML Studio Social Preview, Benchmarking, CLI workflows, Dashboard, Local-first tabular ML workbench, MLflow-backed history, Model Registry, Portfolio-ready local ML tooling (+4 more)

### Community 116 - "Integration Optional Deps Tests"
Cohesion: 0.27
Nodes (10): DataFrame, parametrize, Path, Optional integration checks for heavy dependency imports. These tests are…, _small_classification_df(), _sqlite_uri(), test_benchmark_executes_real_lazypredict_and_mlflow(), test_gx_validation_executes_real_expectations() (+2 more)

### Community 117 - "UA Analysis Artifacts"
Cohesion: 0.26
Nodes (11): complexity(), fileSummary(), fs, make(), memberSummary(), path, special, tagsFor() (+3 more)

### Community 118 - "Base Components"
Cohesion: 0.24
Nodes (5): BaseArtifacts, Path, Shared helpers for artifact path generation and persistence., Write artifacts for the current bundle and return the bundle of paths., ArtifactBundleT

### Community 119 - "Providers Ollama Provider"
Cohesion: 0.25
Nodes (3): OllamaProvider, Any, Ollama has no auth — just check reachability.

### Community 120 - "Code Of Conduct"
Cohesion: 0.18
Nodes (11): Community interest, Code of Conduct, Enforcement actions, Giving and accepting feedback gracefully, Harassment-free participation, Maintainer enforcement responsibility, Positive environment, Project spaces (+3 more)

### Community 121 - "Autotabml Studio Architecture Documentation"
Cohesion: 0.18
Nodes (11): AutoML Modeling, Background Jobs, Data Quality, Dataset Ingestion, Deployment Bundle, External Providers, Metadata Store, ML Practitioner (+3 more)

### Community 122 - "UA Analysis Artifacts"
Cohesion: 0.27
Nodes (10): comp(), emit(), fileSummary(), fs, make(), memberSummary(), path, purpose (+2 more)

### Community 123 - "Artifacts Manager"
Cohesion: 0.24
Nodes (7): Local artifact management utilities., ArtifactKind, datetime, Enum, str, Centralized local artifact path and lifecycle management. This manager only…, Supported local artifact directory kinds.

### Community 124 - "Observability Tracing"
Cohesion: 0.22
Nodes (6): _NoopSpan, Any, BaseException, Protocol, The minimal span surface used by AutoTabML Studio call sites., SpanLike

### Community 125 - "Providers Tests"
Cohesion: 0.33
Nodes (3): build_provider(), Instantiate the concrete provider. Falls back to environment variables for…, TestBuildProvider

### Community 126 - "UA Analysis Artifacts"
Cohesion: 0.20
Nodes (9): dir, fs, graph, ids, layers, output, path, scan (+1 more)

### Community 127 - "Deployment Components"
Cohesion: 0.36
Nodes (8): _build_current_wheel(), DeploymentBundle, export_deployment_bundle(), BaseModel, Path, Portable prediction bundle generation., Create API, Docker, and command-line deployment assets for one trusted model., _sha256()

### Community 128 - "Safe HTTP Tests"
Cohesion: 0.25
Nodes (6): fetch_url_bytes(), fetch_url_text(), Fetch a remote resource and return bytes, content-type, and final URL., Fetch a remote resource and return decoded text, content-type, and final URL., Confirm ingestion-facing wrappers translate guard errors into RemoteAccessError., TestIngestionIntegration

### Community 129 - "FLAML Setup Runner"
Cohesion: 0.25
Nodes (9): flaml_install_guidance(), is_flaml_available(), _probe_flaml_import_error(), Exception, Return the import-time failure when FLAML is unusable., Return True when FLAML is importable., Return a user-facing installation hint for environments without FLAML., Raise a clean dependency error when FLAML is unavailable. (+1 more)

### Community 130 - "Security Tests"
Cohesion: 0.39
Nodes (3): mask_secret(), Mask a secret string, keeping only *reveal* chars at each end visible.…, TestMaskSecret

### Community 131 - "Benchmark Leaderboard Documentation"
Cohesion: 0.25
Nodes (8): Benchmark, Benchmark configuration screen, Random seed: 42, Sample rows: 0 (full dataset), Target column: target, Task type: regression, Test size: 0.20, Top-k shortlist: 5

### Community 132 - "History View Documentation"
Cohesion: 0.29
Nodes (8): History navigation item, History view, Inspect run selector, Navigation sidebar, Run Detail, Run history table, Run metadata columns, Selected benchmark classification run

### Community 133 - "Profiling Report Documentation"
Cohesion: 0.36
Nodes (8): Column Types: 11 Numeric, 0 Categorical, Data Quality Summary: 0.0% Missing, 0 Duplicates, Dataset Dimensions: 442 Rows, 11 Columns, Active Diabetes Dataset, Profiling Complete, Profiling Artifacts, EDA / Profiling Dashboard, Standard Report Mode

### Community 134 - "Settings View Documentation"
Cohesion: 0.29
Nodes (8): Accelerators, colab_mcp backend, Execution backend, Local backend, Application navigation, Promote version, Register model, Settings view

### Community 135 - "Artifact Manager Tests"
Cohesion: 0.46
Nodes (4): _artifact_settings_for(), Path, Tests for centralized local artifact lifecycle management., TestLocalArtifactManager

### Community 137 - "Experiment Lab Documentation"
Cohesion: 0.29
Nodes (7): Benchmark configuration form, Experiment Lab, Experiment navigation item, Navigation sidebar, Run Benchmark button, Target column selector, Task type selector (regression)

### Community 138 - "Validation Summary Documentation"
Cohesion: 0.29
Nodes (7): Active Dataset: diabetes, Data Validation, Dataset Dimensions: 442 Rows, 11 Columns, Data Validation Screen, Target Column: target, Validation Complete, Validation Results: 5 Passed, 0 Warnings, 0 Failed

### Community 140 - "CLI Tests"
Cohesion: 0.29
Nodes (3): Regression tests for CLI output encoding on Windows cp1252., Ensure compare CLI output uses ASCII-safe arrows (->), not Unicode → (\u2192)., TestCliOutputEncoding

### Community 142 - "Observability Logging Setup"
Cohesion: 0.33
Nodes (5): CorrelationFilter, install_correlation_filter(), Logger, Attach :class:`CorrelationFilter` to ``logger`` (root by default). Idempotent:…, Inject the active correlation context onto every :class:`LogRecord`. Each key…

### Community 143 - "Security Safe HTTP"
Cohesion: 0.33
Nodes (5): ValueError, Fetch up to ``sample_size`` bytes for content sniffing. Implemented on top of…, Async counterpart of :func:`safe_stream_sample`., safe_stream_sample(), safe_stream_sample_async()

### Community 144 - "Compare View Documentation"
Cohesion: 0.33
Nodes (6): Artifact Availability, Saved comparison artifacts, Comparison status: not comparable, MLflow run comparison view, Compare view screenshot, Metadata verification warnings

### Community 145 - "Dashboard Overview Documentation"
Cohesion: 0.33
Nodes (6): dashboard-overview, Active dataset diabetes, Dashboard, Application navigation, Recent Local Jobs, Workspace metrics

### Community 146 - "Dataset Intake Documentation"
Cohesion: 0.33
Nodes (6): Benchmark, Loaded dataset 'diabetes', Experiment, Load Local Path, Profiling, Validation

### Community 147 - "Prediction Center Documentation"
Cohesion: 0.33
Nodes (6): Discovered Local Model, Load Model Action, Model Loading Guidance, Prediction Center Interface, Prediction Navigation, Task Type Hint

### Community 148 - "Registry View Documentation"
Cohesion: 0.33
Nodes (6): e2e-demo-classifier, Model Version Promotion, Model Registry, Model Versions, Registered Models, Registry View Screenshot

### Community 150 - "UA Analysis Artifacts"
Cohesion: 0.33
Nodes (5): cp, fs, path, projectRoot, uaDir

### Community 151 - "UA Analysis Artifacts"
Cohesion: 0.33
Nodes (5): cp, fs, path, projectRoot, uaDir

### Community 152 - "PyCaret Init"
Cohesion: 0.40
Nodes (3): PyCaret-backed experiment lab for AutoTabML Studio., Saved model metadata plus its persisted metadata sidecar path., SavedModelArtifact

### Community 153 - "Conftest Tests"
Cohesion: 0.50
Nodes (4): default_settings(), fixture, Shared test fixtures., runtime_state()

### Community 154 - "UA Analysis Artifacts"
Cohesion: 0.40
Nodes (4): fs, output, path, scan

### Community 155 - "Security Safe HTTP"
Cohesion: 0.50
Nodes (4): _is_blocked_ip(), Return a human-readable reason if ``addr`` is in a blocked range., IPv4Address, IPv6Address

### Community 156 - "Safe CSV Properties Tests"
Cohesion: 0.50
Nodes (4): composite, DrawFn, _dangerous_cell(), Generate a cell that begins with a known dangerous prefix.

### Community 157 - "Dataset Intake Documentation"
Cohesion: 0.50
Nodes (4): Dataset Intake page, Load Dataset Sources, Local dataset path: E:\Github\AutoTabML-Studio\datasets\sklearn\Diabetes\diabetes.csv, Local Path tab

### Community 158 - "Prediction Center Documentation"
Cohesion: 0.50
Nodes (4): Model Source, Prediction Job Mode, Prediction Job Status, Recent Prediction Jobs

### Community 159 - "Release Notes V0 2 0"
Cohesion: 0.50
Nodes (4): 0.1.x to 0.2.0 migration guide, v0.2.0 release notes, AutoTabML Studio v0.2.0, 0.1.x to 0.2.0 upgrade summary

### Community 161 - "PyCaret Compare Runner"
Cohesion: 0.67
Nodes (3): create_model(), Any, Create one concrete model from a model id.

### Community 162 - "PyCaret Tune Runner"
Cohesion: 0.67
Nodes (3): Any, Execute tune_model and optionally capture the tuner object., run_tune_model()

### Community 163 - "Safe HTTP Tests"
Cohesion: 0.67
Nodes (3): public_dns(), fixture, Force hostname resolution to a *public* IP regardless of the host string. This…

## Knowledge Gaps
- **198 isolated node(s):** `fs`, `path`, `ua`, `purpose`, `fs` (+193 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **25 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `DatasetInputSpec` connect `UCI Loader Tests` to `PyCaret Experiment Tests`, `CLI Application`, `Prediction Schemas`, `Dataset Workspace`, `Ingestion URL Loader`, `Notebook Page`, `UCI Real Datasets Tests`, `Prediction Tests`, `Ingestion Base`, `Prediction Base`, `Prediction Selectors`, `Batch UCI Runner Scripts`, `UI Cache`, `Ingestion UCI Loader`, `Prediction Page`, `Ingestion Metadata`, `Ingestion Tests`, `Verify Optional Deps Scripts`, `CLI Tests`, `Ingestion Kaggle Loader`, `Ingestion HTML Table Loader`, `Ingestion Async Tests`, `E2E Local Smoke Tests`, `UI Cache`?**
  _High betweenness centrality (0.105) - this node is a cross-community bridge._
- **Why does `ExecutionBackend` connect `PyCaret Experiment Tests` to `Benchmark Schemas`, `CLI Application`, `FLAML Automl Tests`, `FLAML Schemas`, `Providers Base`, `Benchmark Tests`, `Config Models`, `Modeling Architecture Tests`, `Security Trusted Artifacts`, `CLI Tests`, `CLI Tests`, `Notebook Page`, `Colab MCP Backend Tests`, `PyCaret Init`, `CLI Tests`, `Backends Colab MCP Backend`, `PyCaret Service`, `PyCaret Selectors`, `PyCaret Summary`, `CLI Tests`, `Startup Components`, `Providers Tests`, `Settings Page`, `Prediction Tests`, `CLI Tests`, `CLI Tests`, `Config Tests`, `Tracking Tests`, `State Session`, `Providers Anthropic Provider`, `Providers Tests`, `Modeling Architecture Tests`, `Providers Tests`?**
  _High betweenness centrality (0.089) - this node is a cross-community bridge._
- **Why does `AppSettings` connect `CLI Tests` to `PyCaret Experiment Tests`, `CLI Application`, `FLAML Automl Tests`, `Prediction Schemas`, `Providers Base`, `Artifact Manager Tests`, `Config Models`, `CLI Tests`, `CLI Tests`, `History Page`, `Notebook Page`, `UCI Real Datasets Tests`, `Storage Models`, `Colab MCP Backend Tests`, `Prediction Tests`, `Conftest Tests`, `Backends Colab MCP Backend`, `Config Models`, `Batch UCI Runner Scripts`, `UI Cache`, `CLI Tests`, `Startup Components`, `Storage SQLite Connector`, `Validation Schemas`, `Prediction Tests`, `Experiment Page`, `Verify Optional Deps Scripts`, `Prediction Loader`, `CLI Tests`, `CLI Tests`, `Config Tests`, `Tracking Compare Service`, `Autorun Components`, `Tracking Tests`, `State Session`, `E2E Local Smoke Tests`, `UI Cache`?**
  _High betweenness centrality (0.072) - this node is a cross-community bridge._
- **Are the 167 inferred relationships involving `ExecutionBackend` (e.g. with `BaseExecutionBackend` and `ColabMCPExecutionBackend`) actually correct?**
  _`ExecutionBackend` has 167 INFERRED edges - model-reasoned connections that need verification._
- **Are the 81 inferred relationships involving `AppSettings` (e.g. with `ExecutionBackend` and `LLMProvider`) actually correct?**
  _`AppSettings` has 81 INFERRED edges - model-reasoned connections that need verification._
- **Are the 53 inferred relationships involving `DatasetInputSpec` (e.g. with `BaseLoader` and `CSVLoader`) actually correct?**
  _`DatasetInputSpec` has 53 INFERRED edges - model-reasoned connections that need verification._
- **Are the 129 inferred relationships involving `WorkspaceMode` (e.g. with `AppSettings` and `ArtifactSettings`) actually correct?**
  _`WorkspaceMode` has 129 INFERRED edges - model-reasoned connections that need verification._