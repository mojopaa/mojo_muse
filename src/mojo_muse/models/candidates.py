from __future__ import annotations

import dataclasses
import importlib.metadata as im
from abc import ABC, abstractmethod
from functools import lru_cache
from pathlib import Path
from typing import Any

from mups.utils import normalize_name
from packaging.specifiers import InvalidSpecifier, SpecifierSet

from .._types import FileHash, Package
from ..utils import cd
from .link import Link
from .requirements import BaseMuseRequirement


class Candidate:
    """A concrete candidate that can be downloaded and installed.
    A candidate comes from the MojoPI index of a package, or from the requirement itself
    (for file or VCS requirements). Each candidate has a name, version and several
    dependencies together with package metadata.
    """

    __slots__ = (
        "req",
        "name",
        "version",
        "link",
        "summary",
        "hashes",
        "_prepared",
        "_requires_mojo",  # TODO: move to subclass
        "_preferred",
    )

    def __init__(
        self,
        req: BaseMuseRequirement,
        name: str | None = None,
        version: str | None = None,
        link: Link | None = None,
    ):
        self.req = req
        self.name = name or self.req.project_name

        self.version = version
        if link is None and not req.is_named:
            # link = cast("Link", req.as_file_link())  # type: ignore[attr-defined]
            link = Link(req.as_file_link())  # type: ignore[attr-defined]
        self.link = link
        self.summary = ""
        self.hashes: list[FileHash] = []

        self._requires_mojo: str | None = None
        self._prepared: BasePreparedCandidate | None = None

    def identify(self) -> str:
        return self.req.identify()

    @property
    def dep_key(self) -> tuple[str, str | None]:
        """Key for retrieving and storing dependencies from the provider.

        Return a tuple of (name, version). For URL candidates, the version is None but
        there will be only one for the same name so it is also unique.
        """
        return (self.identify(), self.version)

    @property
    def prepared(self) -> BasePreparedCandidate | None:
        return self._prepared

    def __eq__(self, other) -> bool:
        if not isinstance(other, Candidate):
            return False
        if self.req.is_named:
            return self.name == other.name and self.version == other.version
        return self.name == other.name and self.link == other.link

    def get_revision(self) -> str:
        if not self.req.is_vcs:
            raise AttributeError("Non-VCS candidate doesn't have revision attribute")
        if self.req.revision:  # type: ignore[attr-defined]
            return self.req.revision  # type: ignore[attr-defined]
        return self._prepared.revision if self._prepared else "unknown"

    def __repr__(self) -> str:
        source = getattr(self.link, "comes_from", None)
        from_source = f" from {source}" if source else ""
        return f"<Candidate {self}{from_source}>"

    def __str__(self) -> str:
        if self.req.is_named:
            return f"{self.name}@{self.version}"
        assert self.link is not None
        return f"{self.name}@{self.link.url_without_fragment}"

    @classmethod
    def from_installation_candidate(
        cls, candidate: Package, req: BaseMuseRequirement
    ) -> Candidate:
        # TODO: evaluator.py
        # """Build a candidate from unearth's find result."""
        return cls(
            req,
            name=candidate.name,
            version=str(candidate.version),
            link=candidate.link,
        )

    @property
    def requires_mojo(self) -> str:
        """The Mojo version constraint of the candidate."""
        if self._requires_mojo is not None:
            return self._requires_mojo
        if self.link:
            requires_mojo = self.link.requires_mojo
            if requires_mojo is not None:
                if requires_mojo.isdigit():
                    requires_mojo = f">={requires_mojo},<{int(requires_mojo) + 1}"
                try:  # ensure the specifier is valid
                    SpecifierSet(requires_mojo)  # TODO: check effectiveness
                except InvalidSpecifier:
                    # pass
                    raise
                else:
                    self._requires_python = requires_mojo
        return self._requires_python or ""

    @requires_mojo.setter
    def requires_mojo(self, value: str) -> None:
        self._requires_python = value

    def as_lockfile_entry(self, project_root: Path) -> dict[str, Any]:
        """Build a lockfile entry dictionary for the candidate."""
        result = {
            "name": normalize_name(self.name),
            "version": str(self.version),
            "extras": sorted(self.req.extras or ()),
            "requires_mojo": str(self.requires_mojo),
            "editable": self.req.editable,
            "subdirectory": getattr(self.req, "subdirectory", None),
        }
        if self.req.is_vcs:
            result.update(
                {
                    self.req.vcs: self.req.repo,
                    "ref": self.req.ref,
                }
            )
            if not self.req.editable:
                result.update(revision=self.get_revision())
        elif not self.req.is_named:
            with cd(project_root):
                if self.req.is_file_or_url and self.req.is_local:
                    result.update(path=self.req.str_path)
                else:
                    result.update(url=self.req.url)
        return {k: v for k, v in result.items() if v}

    def format(self) -> str:
        """Format for output."""
        return f"[req]{self.name}[/] [warning]{self.version}[/]"

    # def prepare(self, environment: BaseEnvironment) -> BasePreparedCandidate:
    # TODO: move to project
    #     """Prepare the candidate for installation."""
    #     if self._prepared is None:
    #         self._prepared = BasePreparedCandidate(self, environment)
    #     return self._prepared

    # TODO: PreparedCandidate


def _filter_none(data: dict[str, Any]) -> dict[str, Any]:
    """Return a new dict without None values"""
    return {k: v for k, v in data.items() if v is not None}


class BasePreparedCandidate(ABC):
    def __init__(self, candidate: Candidate) -> None:
        self.candidate = candidate
        self.req = candidate.req

        self.ring: Path | None = None
        self.link = self._replace_url_vars(self.candidate.link)

        self._source_dir: Path | None = None
        self._unpacked_dir: Path | None = None
        self._metadata_dir: str | None = None
        self._metadata: im.Distribution | None = None

        if self.link is not None and self.link.is_file and self.link.file_path.is_dir():
            self._source_dir = self.link.file_path
            self._unpacked_dir = self._source_dir / (self.link.subdirectory or "")

    def _replace_url_vars(self, link: Link | None) -> Link | None:
        if link is None:
            return None
        url = link.normalized
        return dataclasses.replace(link, url=url)

    @abstractmethod
    def revision(self) -> str:
        pass

    @abstractmethod
    def direct_url(self) -> dict[str, Any] | None:
        pass

    @abstractmethod
    def build(self) -> Path:
        pass

    @abstractmethod
    def obtain(self, allow_all: bool = False, unpack: bool = True) -> None:
        """Fetches the link of the candidate and unpacks it locally if necessary.

        Args:
            allow_all (bool, optional): If True, don't validate the wheel tag nor hashes. Defaults to False.
            unpack (bool, optional): Whether to download and unpack the link if it's not local. Defaults to True.

        Returns:
            None
        """

    @abstractmethod
    def prepare_metadata(self, force_build: bool = False) -> im.Distribution:
        pass

    @property
    @abstractmethod
    def metadata(self) -> im.Distribution:
        pass

    @abstractmethod
    def get_dependencies_from_metadata(self) -> list[str]:
        pass

    @abstractmethod
    def should_cache(self) -> bool:
        pass


@lru_cache(maxsize=None)
def make_candidate(
    req: BaseMuseRequirement,  # TODO: change to BaseFileMuseRequirement
    name: str | None = None,
    version: str | None = None,
    link: Link | None = None,
) -> Candidate:
    """Construct a candidate and cache it in memory"""
    return Candidate(
        req, name, version, link
    )  # TODO: make mojo candidate or python candidate
