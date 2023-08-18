"""
Special requirement and candidate classes to describe a requires-python constraint
"""
from __future__ import annotations

from typing import Iterable, Iterator, Mapping, cast

from ..models.candidates import Candidate
from ..models.requirements import BaseMuseRequirement, NamedMuseRequirement
from ..models.specifiers import PySpecSet  # TODO


class PythonCandidate(Candidate):
    def format(self) -> str:
        return f"[req]{self.name}[/][warning]{self.req.specifier!s}[/]"


class PythonRequirement(NamedMuseRequirement):
    @classmethod
    def from_pyspec_set(cls, spec: PySpecSet) -> PythonRequirement:
        return cls(name="python", specifier=spec)

    def as_candidate(self) -> PythonCandidate:
        return PythonCandidate(self)


def find_python_matches(
    identifier: str,
    requirements: Mapping[str, Iterator[BaseMuseRequirement]],
) -> Iterable[Candidate]:
    """All requires-python except for the first one(must come from the project)
    must be superset of the first one.
    """
    python_reqs = cast(Iterator[PythonRequirement], iter(requirements[identifier]))
    project_req = next(python_reqs)
    python_specs = cast(Iterator[PySpecSet], (req.specifier for req in python_reqs))
    if all(spec.is_superset(project_req.specifier or "") for spec in python_specs):
        return [project_req.as_candidate()]
    else:
        # There is a conflict, no match is found.
        return []


def is_python_satisfied_by(
    requirement: BaseMuseRequirement, candidate: Candidate
) -> bool:
    return cast(PySpecSet, requirement.specifier).is_superset(candidate.req.specifier)
