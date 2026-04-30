"""Domain-scoped repositories for the local app metadata store.

Each repository encapsulates the SQL and row-mapping logic for a single domain
(projects, datasets, jobs, saved local models, batch runs). Repositories share a
single :class:`SQLiteConnector` and lazy-initialization guard via
:class:`BaseRepository`.

The previous monolithic ``AppMetadataStore`` is preserved as a thin facade that
composes these repositories so existing call sites keep working.
"""

from app.storage.repositories.base import BaseRepository, RepositoryContext
from app.storage.repositories.batch_runs import BatchRunRepository
from app.storage.repositories.datasets import DatasetRepo, DatasetRepository
from app.storage.repositories.jobs import JobRepo, JobRepository
from app.storage.repositories.projects import ProjectRepo, ProjectRepository
from app.storage.repositories.saved_models import SavedModelRepository

__all__ = [
    "BaseRepository",
    "BatchRunRepository",
    "DatasetRepo",
    "DatasetRepository",
    "JobRepo",
    "JobRepository",
    "ProjectRepo",
    "ProjectRepository",
    "RepositoryContext",
    "SavedModelRepository",
]
