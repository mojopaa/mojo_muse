from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Iterable, Mapping, Protocol, Union

from ..models.candidates import Candidate
from ..models.requirements import BaseMuseRequirement
from ..project import Project

ExportItems = Union[Iterable[Candidate], Iterable[BaseMuseRequirement]]


class _Format(Protocol):
    def check_fingerprint(self, project: Project | None, filename: str | Path) -> bool:
        ...

    def convert(
        self,
        project: Project | None,
        filename: str | Path,
        options: Namespace | None,  # TODO: change to click types
    ) -> tuple[Mapping, Mapping]:
        ...

    def export(
        self, project: Project, candidates: ExportItems, options: Namespace | None
    ) -> str:
        ...


# FORMATS: Mapping[str, _Format] = {
#     "pipfile": cast("_Format", pipfile),
#     "poetry": cast("_Format", poetry),
#     "flit": cast("_Format", flit),
#     "setuppy": cast("_Format", setup_py),
#     "requirements": cast("_Format", requirements),
# }
