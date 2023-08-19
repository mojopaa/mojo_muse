import itertools
import os
from typing import Callable, Iterable, Iterator, Mapping, Sequence, cast

from mups import normalize_name
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from resolvelib import AbstractProvider
from resolvelib.resolvers import RequirementInformation

from ..models.candidates import Candidate, make_candidate
from ..models.repositories import BaseRepository, LockedRepository
from ..models.requirements import (
    BaseMuseRequirement,
    FileMuseRequirement,
    parse_requirement,
    strip_extras,
)
from ..project import Project
from ..utils import Comparable, is_url, url_without_fragments
from .mojo import (
    MojoCandidate,
    MojoRequirement,
    find_mojo_matches,
    is_mojo_satisfied_by,
)


class BaseProvider(AbstractProvider):
    def __init__(
        self,
        repository: BaseRepository,
        project: Project,
        allow_prereleases: bool | None = None,
        overrides: dict[str, str] | None = None,
    ) -> None:
        self.repository = repository
        self.project = project
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
        this_name = self.repository.environment.project.name  # TODO
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

    def get_dependencies(
        self, candidate: Candidate
    ) -> list[BaseMuseRequirement]:  # TODO
        if isinstance(candidate, MojoCandidate):
            return []
        deps, requires_mojo, _ = self.repository.get_dependencies(
            candidate
        )  # TODO: get_dependencies need project.

        # Filter out incompatible dependencies(e.g. functools32) early so that
        # we don't get errors when building wheels.
        valid_deps: list[BaseMuseRequirement] = []
        for dep in deps:
            if (
                dep.requires_mojo
                & requires_mojo
                & candidate.req.requires_mojo
                & self.repository.environment.requires_mojo
            ).is_impossible:
                continue
            dep.requires_mojo &= candidate.req.requires_mojo
            valid_deps.append(dep)
        self.fetched_dependencies[candidate.dep_key] = valid_deps[:]
        # A candidate contributes to the Python requirements only when:
        # It isn't an optional dependency, or the requires-python doesn't cover
        # the req's requires-python.
        # For example, A v1 requires python>=3.6, it not eligible on a project with
        # requires-python=">=2.7". But it is eligible if A has environment marker
        # A1; python_version>='3.8'
        new_requires_mojo = (
            candidate.req.requires_mojo & self.repository.environment.requires_mojo
        )
        if candidate.identify() not in self.overrides and not requires_mojo.is_superset(
            new_requires_mojo
        ):
            valid_deps.append(MojoRequirement.from_pyspec_set(requires_mojo))
        return valid_deps


class ReusePinProvider(BaseProvider):
    """A provider that reuses preferred pins if possible.

    This is used to implement "add", "remove", and "reuse upgrade",
    where already-pinned candidates in lockfile should be preferred.
    """

    def __init__(
        self,
        preferred_pins: dict[str, Candidate],
        tracked_names: Iterable[str],
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.preferred_pins = preferred_pins
        self.tracked_names = set(tracked_names)

    def find_matches(
        self,
        identifier: str,
        requirements: Mapping[str, Iterator[BaseMuseRequirement]],
        incompatibilities: Mapping[str, Iterator[Candidate]],
    ) -> Callable[[], Iterator[Candidate]]:
        super_find = super().find_matches(identifier, requirements, incompatibilities)
        bare_name = strip_extras(identifier)[0]

        def matches_gen() -> Iterator[Candidate]:
            if (
                bare_name not in self.tracked_names
                and identifier in self.preferred_pins
            ):
                pin = self.preferred_pins[identifier]
                incompat = list(incompatibilities[identifier])
                demanded_req = next(requirements[identifier], None)
                if demanded_req and demanded_req.is_named:
                    pin.req = demanded_req
                pin._preferred = True  # type: ignore[attr-defined]
                if pin not in incompat and all(
                    self.is_satisfied_by(r, pin) for r in requirements[identifier]
                ):
                    yield pin
            yield from super_find()

        return matches_gen


class EagerUpdateProvider(ReusePinProvider):
    """A specialized provider to handle an "eager" upgrade strategy.

    An eager upgrade tries to upgrade not only packages specified, but also
    their dependencies (recursively). This contrasts to the "only-if-needed"
    default, which only promises to upgrade the specified package, and
    prevents touching anything else if at all possible.

    The provider is implemented as to keep track of all dependencies of the
    specified packages to upgrade, and free their pins when it has a chance.
    """

    def is_satisfied_by(
        self, requirement: BaseMuseRequirement, candidate: Candidate
    ) -> bool:
        # If this is a tracking package, tell the resolver out of using the
        # preferred pin, and into a "normal" candidate selection process.
        if requirement.key in self.tracked_names and getattr(
            candidate, "_preferred", False
        ):
            return False
        return super().is_satisfied_by(requirement, candidate)

    def get_dependencies(self, candidate: Candidate) -> list[BaseMuseRequirement]:
        # If this package is being tracked for upgrade, remove pins of its
        # dependencies, and start tracking these new packages.
        dependencies = super().get_dependencies(candidate)
        if self.identify(candidate) in self.tracked_names:
            for dependency in dependencies:
                if dependency.key:
                    self.tracked_names.add(dependency.key)
        return dependencies

    def get_preference(
        self,
        identifier: str,
        resolutions: dict[str, Candidate],
        candidates: dict[str, Iterator[Candidate]],
        information: dict[str, Iterator[RequirementInformation]],
        backtrack_causes: Sequence[RequirementInformation],
    ) -> tuple[Comparable, ...]:
        # Resolve tracking packages so we have a chance to unpin them first.
        (mojo, *others) = super().get_preference(
            identifier, resolutions, candidates, information, backtrack_causes
        )
        return (mojo, identifier not in self.tracked_names, *others)


def get_provider(
    project: Project,
    strategy: str = "all",
    tracked_names: Iterable[str] | None = None,
    for_install: bool = False,
    ignore_compatibility: bool = True,
) -> BaseProvider:
    """Builds a provider class for resolver.

    Args:
        strategy (str): The resolve strategy.
        tracked_names (Iterable[str] | None): The names of packages that need to be updated.
        for_install (bool): If the provider is for install.
        ignore_compatibility (bool): Whether to ignore compatibility.

    Returns:
        BaseProvider: The provider object.
    """

    repository = project.get_repository(ignore_compatibility=ignore_compatibility)
    allow_prereleases = project.allow_prereleases
    overrides = {
        normalize_name(k): v
        for k, v in project.mojoproject.resolution_overrides.items()
    }
    locked_repository: LockedRepository | None = None
    if strategy != "all" or for_install:
        try:
            locked_repository = project.locked_repository
        except Exception:
            if for_install:
                raise
            project.ui.echo(
                "Unable to reuse the lock file as it is not compatible with PDM",
                style="warning",
                err=True,
            )

    if locked_repository is None:
        return BaseProvider(repository, allow_prereleases, overrides)
    if for_install:
        return BaseProvider(locked_repository, allow_prereleases, overrides)
    provider_class = ReusePinProvider if strategy == "reuse" else EagerUpdateProvider
    tracked_names = [strip_extras(name)[0] for name in tracked_names or ()]
    return provider_class(
        locked_repository.all_candidates,
        tracked_names,
        repository,
        allow_prereleases,
        overrides,
    )
