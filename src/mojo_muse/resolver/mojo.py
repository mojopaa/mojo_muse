from __future__ import annotations

from typing import Iterable, Iterator, Mapping, cast

from packaging.specifiers import SpecifierSet

from ..models.candidates import Candidate
from ..models.requirements import BaseMuseRequirement, NamedMuseRequirement


class MojoCandidate(Candidate):
    def format(self) -> str:
        return f"[req]{self.name}[/][warning]{self.req.specifier!s}[/]"


class MojoRequirement(NamedMuseRequirement):
    @classmethod
    def from_mojospec_set(cls, spec: SpecifierSet) -> MojoRequirement:
        return cls(name="mojo", specifier=spec)

    def as_candidate(self) -> MojoCandidate:
        return MojoCandidate(self)


def find_mojo_matches(
    identifier: str,
    requirements: Mapping[str, Iterator[BaseMuseRequirement]],
) -> Iterable[Candidate]:
    """All requires-python except for the first one(must come from the project)
    must be superset of the first one.
    """
    mojo_reqs = cast(Iterator[MojoRequirement], iter(requirements[identifier]))
    project_req = next(mojo_reqs)
    mojo_specs = cast(Iterator[SpecifierSet], (req.specifier for req in mojo_reqs))
    if all(
        spec.is_superset(project_req.specifier or "") for spec in mojo_specs
    ):  # TODO: PySpec.is_superset()
        return [project_req.as_candidate()]
    else:
        # There is a conflict, no match is found.
        return []


def is_mojo_satisfied_by(
    requirement: BaseMuseRequirement, candidate: Candidate
) -> bool:
    return cast(SpecifierSet, requirement.specifier).is_superset(
        candidate.req.specifier
    )  # TODO: PySpec.is_superset()
