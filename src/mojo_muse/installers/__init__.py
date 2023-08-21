# flake8: noqa
from __future__ import annotations

from resolvelib.resolvers import Resolver

from ..models.requirements import BaseMuseRequirement
from ..models.specifiers import PySpecSet
from ..project import Project
from ..resolver import get_provider, get_reporter, resolve_python
from .manager import InstallManager

# from pdm.environments import BaseEnvironment
from .synchronizers import BaseSynchronizer, Synchronizer


def install_requirements(
    reqs: list[BaseMuseRequirement],
    project: Project,
    clean: bool = False,
    use_install_cache: bool = False,
) -> None:  # pragma: no cover
    """Resolve and install the given requirements into the environment."""
    # Rewrite the python requires to only resolve for the current python version.
    project.requires_python = PySpecSet(f"=={project.interpreter.version}")
    provider = get_provider(ignore_compatibility=False)
    reporter = get_reporter(reqs)
    resolver = Resolver(provider, reporter)
    resolve_max_rounds = int(project.config["strategy.resolve_max_rounds"])
    backend = project.backend
    for req in reqs:
        if req.is_file_or_url:
            req.relocate(backend)  # type: ignore[attr-defined]
    resolved, _ = resolve_python(
        resolver=resolver,
        requirements=reqs,
        requires_python=project.requires_python,
        project=project,
        max_rounds=resolve_max_rounds,
        keep_self=True,
    )
    syncer = BaseSynchronizer(
        candidates=resolved,
        project=project,
        clean=clean,
        retry_times=0,
        use_install_cache=use_install_cache,
    )
    syncer.synchronize()
