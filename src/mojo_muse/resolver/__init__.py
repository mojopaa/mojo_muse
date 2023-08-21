from __future__ import annotations

from typing import TYPE_CHECKING, Dict, cast

from mups import normalize_name
from resolvelib.resolvers import Resolver

from ..models.candidates import Candidate
from ..models.repositories import BaseRepository
from ..models.requirements import BaseMuseRequirement, strip_extras
from ..models.specifiers import PySpecSet
from ..project import Project
from .providers import BaseProvider, get_provider
from .python import PythonRequirement
from .reporters import get_reporter


def resolve_python(
    resolver: Resolver,
    requirements: list[BaseMuseRequirement],
    requires_python: PySpecSet,
    project: Project,
    max_rounds: int = 10000,
    keep_self: bool = False,
) -> tuple[
    dict[str, Candidate], dict[tuple[str, str | None], list[BaseMuseRequirement]]
]:
    """Core function to perform the actual resolve process.
    Return a tuple containing 2 items:

        1. A map of pinned candidates
        2. A map of resolved dependencies for each dependency group
    """
    requirements.append(PythonRequirement.from_pyspec_set(requires_python))
    provider = cast(BaseProvider, resolver.provider)

    result = resolver.resolve(requirements, max_rounds)

    mapping = cast(Dict[str, Candidate], result.mapping)
    mapping.pop("python", None)

    local_name = normalize_name(project.name) if project.name else None
    for key, candidate in list(result.mapping.items()):
        if key is None:
            continue

        # For source distribution whose name can only be determined after it is built,
        # the key in the resolution map should be updated.
        if key.startswith(":empty:"):
            new_key = provider.identify(candidate)
            mapping[new_key] = mapping.pop(key)
            key = new_key

        if not keep_self and strip_extras(key)[0] == local_name:
            del mapping[key]

    return mapping, provider.fetched_dependencies
