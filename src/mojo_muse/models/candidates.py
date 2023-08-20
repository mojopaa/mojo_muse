from __future__ import annotations

import base64
import dataclasses
import hashlib
import importlib.metadata as im
import os
import re
import warnings
from abc import ABC, abstractmethod
from functools import cached_property, lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast

from mups import normalize_name, parse_ring_filename
from packaging.specifiers import InvalidSpecifier, SpecifierSet

from ..exceptions import CandidateNotFound
from ..termui import UI, logger, ui
from ..utils import (
    CandidateInfo,
    FileHash,
    Package,
    cd,
    convert_hashes,
    create_tracked_tempdir,
    get_rev_from_url,
    path_to_url,
    url_without_fragments,
)
from .caches import JSONFileCache, ProjectCache
from .link import Link
from .requirements import BaseMuseRequirement, FileMuseRequirement, VcsMuseRequirement
from .vcs import vcs_support


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


def _find_best_match_link(
    finder: PackageFinder,  # TODO
    req: BaseMuseRequirement,
    files: list[FileHash],
    ignore_compatibility: bool = False,
) -> Link | None:
    """Get the best matching link for a requirement"""

    # This function is called when a lock file candidate is given or incompatible wheel
    # In this case, the requirement must be pinned, so no need to pass allow_prereleases
    # If links are not empty, find the best match from the links, otherwise find from
    # the package sources.

    links = [Link(f["url"]) for f in files if "url" in f]
    hashes = convert_hashes(files)

    def attempt_to_find() -> Link | None:
        if not links:
            best = finder.find_best_match(req.as_line(), hashes=hashes).best  # TODO
        else:
            # this branch won't be executed twice if ignore_compatibility is True
            evaluator = finder.build_evaluator(req.name)
            packages = finder._evaluate_links(links, evaluator)
            best = max(packages, key=finder._sort_key, default=None)
        return best.link if best is not None else None

    assert finder.ignore_compatibility is False
    found = attempt_to_find()
    if ignore_compatibility and (found is None or not found.is_ring):
        # try to find a ring for easy metadata extraction
        finder.ignore_compatibility = True
        new_found = attempt_to_find()
        if new_found is not None:
            found = new_found
        finder.ignore_compatibility = False
    return found


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


class PreparedCandidate(BasePreparedCandidate):
    ui: UI = ui
    project_cache: ProjectCache | None = None

    def __init__(
        self,
        candidate: Candidate,
        project_cache: ProjectCache | None = None,
        ui: UI = ui,
    ) -> None:
        super().__init__(candidate)
        self.project_cache = project_cache or ProjectCache()
        self.ui = ui

    @cached_property
    def revision(self) -> str:
        if not (self._source_dir and os.path.exists(self._source_dir)):
            # It happens because the cached wheel is hit and the source code isn't
            # pulled to local. In this case the link url must contain the full commit
            # hash which can be taken as the revision safely.
            # See more info at https://github.com/pdm-project/pdm/issues/349
            rev = get_rev_from_url(self.candidate.link.url)  # type: ignore[union-attr]
            if rev:
                return rev
        assert isinstance(self.req, VcsMuseRequirement)  # TODO
        return vcs_support.get_backend(self.req.vcs, self.ui.verbosity).get_revision(
            cast(Path, self._source_dir)
        )

    def direct_url(self) -> dict[str, Any] | None:
        """PEP 610 direct_url.json data"""
        req = self.req
        if isinstance(req, VcsMuseRequirement):
            if req.editable:
                assert self._source_dir
                return _filter_none(
                    {
                        "url": path_to_url(self._source_dir.as_posix()),
                        "dir_info": {"editable": True},
                        "subdirectory": req.subdirectory,
                    }
                )
            return _filter_none(
                {
                    "url": url_without_fragments(req.repo),
                    "vcs_info": _filter_none(
                        {
                            "vcs": req.vcs,
                            "requested_revision": req.ref,
                            "commit_id": self.revision,
                        }
                    ),
                    "subdirectory": req.subdirectory,
                }
            )
        elif isinstance(req, FileMuseRequirement):
            assert self.link is not None
            if self.link.is_file and self.link.file_path.is_dir():
                return _filter_none(
                    {
                        "url": self.link.url_without_fragment,
                        "dir_info": _filter_none({"editable": req.editable or None}),
                        "subdirectory": req.subdirectory,
                    }
                )
            with self.environment.get_finder() as finder:  # TODO: get_finder(), unearth hard work!
                hash_cache = self.project_cache.make_hash_cache()
                return _filter_none(
                    {
                        "url": self.link.url_without_fragment,
                        "archive_info": {
                            "hash": hash_cache.get_hash(
                                self.link, finder.session
                            ).replace(":", "=")
                        },
                        "subdirectory": req.subdirectory,
                    }
                )
        else:
            return None

    def _ring_compatible(self, ring_file: str, allow_all: bool = False) -> bool:
        if allow_all:
            return True
        # supported_tags = self.environment.target_python.supported_tags()  # TODO: make mups supported tags
        file_tags = parse_ring_filename(ring_file)[-1]
        return not file_tags.isdisjoint(supported_tags)

    def _get_cached_wheel(self) -> Path | None:
        ring_cache = self.project_cache.make_ring_cache()  # TODO
        assert self.candidate.link
        cache_entry = ring_cache.get(
            self.candidate.link, self.candidate.name, self.environment.target_python
        )  # TODO: investigate
        if cache_entry is not None:
            logger.info("Using cached wheel: %s", cache_entry)
        return cache_entry

    def _get_build_dir(self) -> str:
        original_link = self.candidate.link
        assert original_link
        if original_link.is_file and original_link.file_path.is_dir():
            # Local directories are built in tree
            return str(original_link.file_path)
        if self.req.editable:
            # In this branch the requirement must be an editable VCS requirement.
            # The repository will be unpacked into a *persistent* src directory.
            prefix: Path | None = None
            if self.environment.is_local:  # TODO
                prefix = self.environment.packages_path  # type: ignore[attr-defined]
            else:
                venv = self.environment.interpreter.get_venv()
                if venv is not None:
                    prefix = venv.root
            if prefix is not None:
                src_dir = prefix / "src"
            else:
                src_dir = Path("src")
            src_dir.mkdir(exist_ok=True, parents=True)
            dirname = self.candidate.name or self.req.name
            if not dirname:
                dirname, _ = os.path.splitext(original_link.filename)
            return str(src_dir / str(dirname))
        # Otherwise, for source dists, they will be unpacked into a *temp* directory.
        return create_tracked_tempdir(prefix="pdm-build-")

    def _unpack(self, validate_hashes: bool = False) -> None:
        hash_options = None
        if validate_hashes and self.candidate.hashes:
            hash_options = convert_hashes(self.candidate.hashes)
        assert self.link is not None
        with self.environment.get_finder() as finder:  # TODO
            with TemporaryDirectory(prefix="muse-download-") as tmpdir:
                build_dir = self._get_build_dir()
                if self.link.is_ring:
                    download_dir = build_dir
                else:
                    download_dir = tmpdir
                result = finder.download_and_unpack(
                    self.link, build_dir, download_dir, hash_options
                )  # TODO
                if self.link.is_ring:
                    self.ring = result
                else:
                    self._source_dir = Path(build_dir)
                    self._unpacked_dir = result

    def _get_cached_ring(self) -> Path | None:
        ring_cache = self.project.make_ring_cache()
        assert self.candidate.link
        cache_entry = ring_cache.get(
            self.candidate.link, self.candidate.name, self.environment.target_python
        )  # TODO
        if cache_entry is not None:
            logger.info("Using cached wheel: %s", cache_entry)
        return cache_entry

    def obtain(self, allow_all: bool = False, unpack: bool = True) -> None:
        """Fetch the link of the candidate and unpack to local if necessary.

        :param allow_all: If true, don't validate the wheel tag nor hashes
        :param unpack: Whether to download and unpack the link if it's not local
        """
        if self.ring:
            if self._ring_compatible(
                self.ring.name, allow_all
            ):  # TODO: use .stem instead of name
                return
        elif self._source_dir and self._source_dir.exists():
            return

        with self.environment.get_finder() as finder:  # TODO
            if (
                not self.link
                or self.link.is_ring
                and not self._ring_compatible(self.link.filename, allow_all)
            ):
                if self.req.is_file_or_url:
                    raise CandidateNotFound(
                        f"The URL requirement {self.req.as_line()} is a wheel but incompatible"
                    )
                self.link = self.ring = None  # reset the incompatible wheel
                self.link = _find_best_match_link(
                    finder,
                    self.req.as_pinned_version(self.candidate.version),
                    self.candidate.hashes,
                    ignore_compatibility=allow_all,
                )
                if not self.link:
                    raise CandidateNotFound(
                        f"No candidate is found for `{self.req.project_name}` that matches the environment or hashes"
                    )
                if not self.candidate.link:
                    self.candidate.link = self.link
            if allow_all and not self.req.editable:
                cached = self._get_cached_ring()  # TODO
                if cached:
                    self.ring = cached
                    return
            if unpack:
                self._unpack(validate_hashes=not allow_all)

    def _get_metadata_from_metadata_link(
        self, link: Link, medata_hash: bool | dict[str, str] | None
    ) -> im.Distribution | None:  # TODO: use mups.RingInfo
        with self.environment.get_finder() as finder:  # TODO
            resp = finder.session.get(
                link.normalized, headers={"Cache-Control": "max-age=0"}
            )
            if isinstance(medata_hash, dict):
                hash_name, hash_value = next(iter(medata_hash.items()))
                if hashlib.new(hash_name, resp.content).hexdigest() != hash_value:
                    logger.warning(
                        "Metadata hash mismatch for %s, ignoring the metadata", link
                    )
                    return None
            return MetadataDistribution(resp.text)

    def _get_metadata_from_build(  # TODO: it's a mess.
        self, source_dir: Path, metadata_parent: str
    ) -> im.Distribution:
        builder = EditableBuilder if self.req.editable else WheelBuilder
        try:
            logger.info("Running PEP 517 backend to get metadata for %s", self.link)
            self._metadata_dir = builder(source_dir, self.environment).prepare_metadata(
                metadata_parent
            )
        except BuildError:
            logger.warning("Failed to build package, try parsing project files.")
            try:
                setup = Setup.from_directory(source_dir)
            except Exception:
                message = (
                    "Failed to parse the project files, dependencies may be missing"
                )
                logger.warning(message)
                warnings.warn(message, RuntimeWarning, stacklevel=1)
                setup = Setup()
            return setup.as_dist()
        else:
            return im.PathDistribution(Path(cast(str, self._metadata_dir)))

    def _get_metadata_from_ring(
        self, ring: Path, metadata_parent: str
    ) -> im.Distribution:
        # Get metadata from METADATA inside the ring
        # TODO: change to ring
        self._metadata_dir = _get_ring_metadata_from_ring(ring, metadata_parent)  # TODO
        return im.PathDistribution(Path(self._metadata_dir))  # TODO: use tomlkit

    def prepare_metadata(
        self, force_build: bool = False
    ) -> im.Distribution:  # TODO: use mups
        self.obtain(allow_all=True, unpack=False)

        metadata_parent = create_tracked_tempdir(prefix="muse-meta-")
        if self.ring:
            return self._get_metadata_from_ring(self.ring, metadata_parent)

        # TODO: move to mups and confirm the usage of Distribution and RingInfo is compatible
        # def _get_wheel_metadata_from_wheel(whl_file: Path, metadata_directory: str) -> str:
        #     """Extract the metadata from a wheel.
        #     Fallback for when the build backend does not
        #     define the 'get_wheel_metadata' hook.
        #     """
        #     with ZipFile(whl_file) as zipf:
        #         dist_info = _dist_info_files(zipf)
        #         zipf.extractall(path=metadata_directory, members=dist_info)
        #     return os.path.join(metadata_directory, dist_info[0].split("/")[0])

        assert self.link is not None
        if self.link.dist_info_metadata:
            assert self.link.dist_info_link
            dist = self._get_metadata_from_metadata_link(
                self.link.dist_info_link, self.link.dist_info_metadata
            )
            if dist is not None:
                return dist

        self._unpack(validate_hashes=False)
        if self.ring:  # check again if the wheel is downloaded to local
            return self._get_metadata_from_ring(self.ring, metadata_parent)

        assert self._unpacked_dir, "Source directory isn't ready yet"
        pyproject_toml = self._unpacked_dir / "pyproject.toml"
        if not force_build and pyproject_toml.exists():
            dist = self._get_metadata_from_project(pyproject_toml)
            if dist is not None:
                return dist

        # If all fail, try building the source to get the metadata
        return self._get_metadata_from_build(self._unpacked_dir, metadata_parent)

    @property
    def metadata(self) -> im.Distribution:
        if self._metadata is None:
            result = self.prepare_metadata()  # TODO: use mups.RingInfo
            if not self.candidate.name:
                self.req.name = self.candidate.name = cast(str, result.metadata["Name"])
            if not self.candidate.version:
                self.candidate.version = result.version
            if not self.candidate.requires_mojo:
                self.candidate.requires_mojo = cast(
                    str, result.metadata["Requires-Mojo"] or ""
                )
            self._metadata = result
        return self._metadata

    def get_dependencies_from_metadata(self) -> list[str]:
        """Get the dependencies of a candidate from metadata."""
        extras = self.req.extras or ()
        return filter_requirements_with_extras(
            self.req.project_name, self.metadata.requires or [], extras  # type: ignore[arg-type]
        )

    def should_cache(self) -> bool:
        """Determine whether to cache the dependencies and built ring."""
        link, source_dir = self.candidate.link, self._source_dir
        _egg_info_re = re.compile(r"([a-z0-9_.]+)-([a-z0-9_.!+-]+)", re.IGNORECASE)
        if self.req.editable:
            return False
        if self.req.is_named:
            return True
        if self.req.is_vcs:
            if not source_dir:
                # If the candidate isn't prepared, we can't cache it
                return False
            assert link
            vcs_backend = vcs_support.get_backend(link.vcs, self.ui.verbosity)
            return vcs_backend.is_immutable_revision(source_dir, link)
        if link and not (link.is_file and link.file_path.is_dir()):
            # Cache if the link contains egg-info like 'foo-1.0'
            return _egg_info_re.search(link.filename) is not None
        return False

    def _get_ring_dir(self) -> str:
        assert self.candidate.link
        ring_cache = self.project.make_ring_cache()
        if self.should_cache():
            logger.info("Saving wheel to cache: %s", self.candidate.link)
            return ring_cache.get_path_for_link(
                self.candidate.link, self.environment.target_python
            ).as_posix()
        else:
            return ring_cache.get_ephemeral_path_for_link(  # TODO
                self.candidate.link, self.environment.target_python
            ).as_posix()

    def _get_cached_ring(self) -> Path | None:
        ring_cache = self.project.make_ring_cache()
        assert self.candidate.link
        cache_entry = ring_cache.get(
            self.candidate.link, self.candidate.name, self.environment.target_python
        )  # TODO
        if cache_entry is not None:
            logger.info("Using cached wheel: %s", cache_entry)
        return cache_entry

    def build(self) -> Path:
        """Call PEP 517 build hook to build the candidate into a wheel"""
        self.obtain(allow_all=False)
        if self.ring:
            return self.ring
        if not self.req.editable:
            cached = self._get_cached_ring()
            if cached:
                self.ring = cached
                return self.ring
        assert self._source_dir, "Source directory isn't ready yet"
        builder_cls = (
            EditableBuilder if self.req.editable else WheelBuilder
        )  # TODO: RingBuilder
        builder = builder_cls(str(self._unpacked_dir), self.environment)  # TODO
        build_dir = self._get_ring_dir()
        os.makedirs(build_dir, exist_ok=True)
        logger.info("Running PEP 517 backend to build a wheel for %s", self.link)
        self.ring = Path(
            builder.build(build_dir, metadata_directory=self._metadata_dir)  # TODO
        )
        return self.ring


class CandidateInfoCache(JSONFileCache[Candidate, CandidateInfo]):
    """A cache manager that stores the
    candidate -> (dependencies, requires_python, summary) mapping.
    """

    @staticmethod
    def get_url_part(link: Link) -> str:
        url = url_without_fragments(link.split_auth()[1])
        return base64.urlsafe_b64encode(url.encode()).decode()

    @classmethod
    def _get_key(cls, obj: Candidate) -> str:
        # Name and version are set when dependencies are resolved,
        # so use them for cache key. Local directories won't be cached.
        if not obj.name or not obj.version:
            raise KeyError("The package is missing a name or version")
        extras = (
            "[{}]".format(",".join(sorted(obj.req.extras))) if obj.req.extras else ""
        )
        version = obj.version
        if not obj.req.is_named:
            assert obj.link is not None
            version = cls.get_url_part(obj.link)
        return f"{obj.name}{extras}-{version}"


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
