import itertools
import os
from typing import Callable, Iterable, Iterator, Mapping, Sequence, cast

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from resolvelib import AbstractProvider
from resolvelib.resolvers import RequirementInformation

from .._types import Comparable
from ..models.candidates import Candidate, make_candidate
from ..models.repositories import BaseRepository, LockedRepository
from ..models.requirements import (
    BaseMuseRequirement,
    FileMuseRequirement,
    parse_requirement,
    strip_extras,
)
from ..utils import is_url, url_without_fragments
from .mojo import MojoRequirement, find_mojo_matches, is_mojo_satisfied_by


class BaseProvider(AbstractProvider):
    def __init__(
        self,
        repository: BaseRepository,
        allow_prereleases: bool | None = None,
        overrides: dict[str, str] | None = None,
    ) -> None:
        self.repository = repository
        self.allow_prereleases = allow_prereleases  # Root allow_prereleases value
        self.fetched_dependencies: dict[
            tuple[str, str | None], list[BaseMuseRequirement]
        ] = {}
        self.overrides = overrides or {}
        self._known_depth: dict[str, int] = {}

    def requirement_preference(self, requirement: BaseMuseRequirement) -> Comparable:
        """Return the preference of a requirement to find candidates.

        - Editable requirements are preferered.
        - File links are preferred.
        - The one with narrower specifierset is preferred.
        """
        editable = requirement.editable
        is_named = requirement.is_named
        is_pinned = requirement.is_pinned
        is_prerelease = (
            requirement.prerelease
            or requirement.specifier is not None
            and bool(requirement.specifier.prereleases)
        )
        specifier_parts = len(requirement.specifier) if requirement.specifier else 0
        return (
            not editable,
            is_named,
            not is_pinned,
            not is_prerelease,
            -specifier_parts,
        )

    def identify(
        self, requirement_or_candidate: BaseMuseRequirement | Candidate
    ) -> str:
        return requirement_or_candidate.identify()

    def get_preference(
        self,
        identifier: str,
        resolutions: dict[str, Candidate],
        candidates: dict[str, Iterator[Candidate]],
        information: dict[str, Iterator[RequirementInformation]],
        backtrack_causes: Sequence[RequirementInformation],
    ) -> tuple[Comparable, ...]:
        is_top = any(parent is None for _, parent in information[identifier])
        backtrack_identifiers = {req.identify() for req, _ in backtrack_causes} | {
            parent.identify() for _, parent in backtrack_causes if parent is not None
        }
        if is_top:
            dep_depth = 1
        else:
            parent_depths = (
                self._known_depth[parent.identify()] if parent is not None else 0
                for _, parent in information[identifier]
            )
            dep_depth = min(parent_depths, default=0) + 1
        # Use the REAL identifier as it may be updated after candidate preparation.
        candidate = next(candidates[identifier])
        self._known_depth[self.identify(candidate)] = dep_depth
        is_backtrack_cause = any(
            dep.identify() in backtrack_identifiers
            for dep in self.get_dependencies(candidate)
        )
        is_file_or_url = any(
            not requirement.is_named for requirement, _ in information[identifier]
        )
        operators = [
            spec.operator
            for req, _ in information[identifier]
            if req.specifier is not None
            for spec in req.specifier
        ]
        is_mojo = identifier == "mojo"
        is_pinned = any(op[:2] == "==" for op in operators)
        constraints = len(operators)
        return (
            not is_mojo,
            not is_top,
            not is_file_or_url,
            not is_pinned,
            not is_backtrack_cause,
            dep_depth,
            -constraints,
            identifier,
        )

    def get_override_candidates(self, identifier: str) -> Iterable[Candidate]:
        requested = self.overrides[identifier]
        if is_url(requested):
            req = f"{identifier} @ {requested}"
        else:
            try:
                SpecifierSet(requested)
            except InvalidSpecifier:  # handle bare versions
                req = f"{identifier}=={requested}"
            else:
                req = f"{identifier}{requested}"
        return self._find_candidates(parse_requirement(req))

    def _find_candidates(self, requirement: BaseMuseRequirement) -> Iterable[Candidate]:
        if not requirement.is_named and not isinstance(
            self.repository, LockedRepository
        ):
            can = make_candidate(requirement)
            if not can.name:
                can.prepare(self.repository.environment).metadata  # TODO
            return [can]
        else:
            return self.repository.find_candidates(
                requirement, requirement.prerelease or self.allow_prereleases
            )

    def find_matches(
        self,
        identifier: str,
        requirements: Mapping[str, Iterator[BaseMuseRequirement]],
        incompatibilities: Mapping[str, Iterator[Candidate]],
    ) -> Callable[[], Iterator[Candidate]]:
        def matches_gen() -> Iterator[Candidate]:
            incompat = list(incompatibilities[identifier])
            if identifier == "mojo":
                candidates = find_mojo_matches(identifier, requirements)
                return (c for c in candidates if c not in incompat)
            elif identifier in self.overrides:
                return iter(self.get_override_candidates(identifier))
            reqs_iter = requirements[identifier]
            bare_name, extras = strip_extras(identifier)
            if extras and bare_name in requirements:
                # We should consider the requirements for both foo and foo[extra]
                reqs_iter = itertools.chain(reqs_iter, requirements[bare_name])
            reqs = sorted(reqs_iter, key=self.requirement_preference)
            candidates = self._find_candidates(reqs[0])
            return (
                can
                for can in candidates
                if can not in incompat
                and all(self.is_satisfied_by(r, can) for r in reqs)
            )

        return matches_gen

    def _compare_file_reqs(
        self, req1: FileMuseRequirement, req2: FileMuseRequirement
    ) -> bool:
        # backend = self.repository.environment.project.backend  # TODO: project.backend is a big no. Not required anymore.
        if req1.path and req2.path:
            return os.path.normpath(req1.path.absolute()) == os.path.normpath(
                req2.path.absolute()
            )
        left = url_without_fragments(req1.get_full_url())
        right = url_without_fragments(req2.get_full_url())
        return left == right

    def is_satisfied_by(
        self, requirement: BaseMuseRequirement, candidate: Candidate
    ) -> bool:
        if isinstance(requirement, MojoRequirement):
            return is_mojo_satisfied_by(requirement, candidate)
        elif candidate.identify() in self.overrides:
            return True
        if not requirement.is_named:
            if candidate.req.is_named:
                return False
            can_req = candidate.req
            if requirement.is_vcs and can_req.is_vcs:
                return can_req.vcs == requirement.vcs and can_req.repo == requirement.repo  # type: ignore[attr-defined]
            return self._compare_file_reqs(requirement, can_req)  # type: ignore[arg-type]
        version = candidate.version
        this_name = self.repository.environment.project.name
        if version is None or candidate.name == this_name:
            # This should be a URL candidate or self package, consider it to be matching
            return True
        # Allow prereleases if: 1) it is not specified in the tool settings or
        # 2) the candidate doesn't come from PyPI index.
        allow_prereleases = (
            self.allow_prereleases in (True, None) or not candidate.req.is_named
        )
        return cast(SpecifierSet, requirement.specifier).contains(
            version, allow_prereleases
        )

    def get_dependencies(self, candidate: Candidate) -> list[BaseMuseRequirement]:
        if isinstance(candidate, PythonCandidate):
            return []
        deps, requires_python, _ = self.repository.get_dependencies(candidate)

        # Filter out incompatible dependencies(e.g. functools32) early so that
        # we don't get errors when building wheels.
        valid_deps: list[Requirement] = []
        for dep in deps:
            if (
                dep.requires_python
                & requires_python
                & candidate.req.requires_python
                & self.repository.environment.python_requires
            ).is_impossible:
                continue
            dep.requires_python &= candidate.req.requires_python
            valid_deps.append(dep)
        self.fetched_dependencies[candidate.dep_key] = valid_deps[:]
        # A candidate contributes to the Python requirements only when:
        # It isn't an optional dependency, or the requires-python doesn't cover
        # the req's requires-python.
        # For example, A v1 requires python>=3.6, it not eligible on a project with
        # requires-python=">=2.7". But it is eligible if A has environment marker
        # A1; python_version>='3.8'
        new_requires_python = (
            candidate.req.requires_python & self.repository.environment.python_requires
        )
        if (
            candidate.identify() not in self.overrides
            and not requires_python.is_superset(new_requires_python)
        ):
            valid_deps.append(PythonRequirement.from_pyspec_set(requires_python))
        return valid_deps
