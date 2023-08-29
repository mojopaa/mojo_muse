"""pdm PreparedCandidate classes and Candidate's prepare function."""
import hashlib
import importlib.metadata as im
import os
import re
import warnings
from functools import cached_property
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast

from mups import parse_ring_filename
from packaging.utils import parse_wheel_filename

from ..exceptions import BuildError, CandidateNotFound
from ..finders import PyPackageFinder
from ..models.caches import ProjectCache
from ..models.candidates import (
    BasePreparedCandidate,
    Candidate,
    MetadataDistribution,
    make_candidate,
)
from ..models.link import Link
from ..models.requirements import (
    BaseMuseRequirement,
    FileMuseRequirement,
    VcsMuseRequirement,
    filter_requirements_with_extras,
    parse_requirement,
)
from ..models.setup import Setup
from ..models.vcs import vcs_support
from ..project import BaseEnvironment, MojoEnvironment, Project, PythonEnvironment
from ..termui import UI, logger
from ..utils import (
    DEFAULT_MOJOPROJECT_FILENAME,
    FileHash,
    convert_hashes,
    create_tracked_tempdir,
    get_rev_from_url,
    path_to_url,
    url_without_fragments,
)
from .editable import EditableBuilder
from .wheel import WheelBuilder

# from candidates.py


def _filter_none(data: dict[str, Any]) -> dict[str, Any]:
    """Return a new dict without None values"""
    return {k: v for k, v in data.items() if v is not None}


def _find_best_match_link_pypi(
    finder: PyPackageFinder,
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


def _find_best_match_link(
    finder: PyPackageFinder,  # TODO: mojo
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
            best = finder.find_best_match(req.as_line(), hashes=hashes).best
        else:
            # this branch won't be executed twice if ignore_compatibility is True
            evaluator = finder.build_evaluator(req.name)
            packages = finder._evaluate_links(links, evaluator)
            best = max(packages, key=finder._sort_key, default=None)
        return best.link if best is not None else None

    assert finder.ignore_compatibility is False
    found = attempt_to_find()
    if ignore_compatibility and (found is None or not found.is_wheel):
        # try to find a wheel for easy metadata extraction
        finder.ignore_compatibility = True
        new_found = attempt_to_find()
        if new_found is not None:
            found = new_found
        finder.ignore_compatibility = False
    return found


class PreparedPythonCandidate(BasePreparedCandidate):
    ui: UI = ui
    environment: PythonEnvironment | None = None
    project_cache: ProjectCache | None = None

    def __init__(
        self,
        candidate: Candidate,
        environment: PythonEnvironment | None = None,
        project_cache: ProjectCache | None = None,
        ui: UI = ui,
    ) -> None:
        if environment is None:
            self.environment = PythonEnvironment(project=Project())  # TODO: assure
        self.project = self.environment.project
        project_cache = project_cache or self.project.project_cache
        super().__init__(candidate=candidate, project_cache=project_cache)

        self.wheel: Path | None = None

        if self.link is not None and self.link.is_file and self.link.file_path.is_dir():
            self._source_dir = self.link.file_path
            self._unpacked_dir = self._source_dir / (self.link.subdirectory or "")

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

    def _wheel_compatible(self, wheel_file: str, allow_all: bool = False) -> bool:
        if allow_all:
            return True
        supported_tags = (
            self.project.target_python.supported_tags()
        )  # TODO: make mups supported tags
        file_tags = parse_wheel_filename(wheel_file)[-1]
        return not file_tags.isdisjoint(supported_tags)

    def _get_cached_wheel(self) -> Path | None:
        ring_cache = self.project_cache.make_wheel_cache()  # TODO
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
            if self.environment.is_local:  # TODO: deprecate: rejected PEP582 path
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
        return create_tracked_tempdir(prefix="muse-build-")

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
            self.candidate.link, self.candidate.name, self.project.target_python
        )  # TODO
        if cache_entry is not None:
            logger.info("Using cached wheel: %s", cache_entry)
        return cache_entry

    def obtain(self, allow_all: bool = False, unpack: bool = True) -> None:
        """Fetch the link of the candidate and unpack to local if necessary.

        :param allow_all: If true, don't validate the wheel tag nor hashes
        :param unpack: Whether to download and unpack the link if it's not local
        """
        if self.wheel:
            if self._wheel_compatible(
                self.ring.name, allow_all
            ):  # TODO: use .stem instead of name
                return
        elif self._source_dir and self._source_dir.exists():
            return

        with self.environment.get_finder() as finder:  # TODO
            if (
                not self.link
                or self.link.is_wheel  # TODO
                and not self._wheel_compatible(self.link.filename, allow_all)
            ):
                if self.req.is_file_or_url:
                    raise CandidateNotFound(
                        f"The URL requirement {self.req.as_line()} is a wheel but incompatible"
                    )
                self.link = self.wheel = None  # reset the incompatible wheel
                self.link = _find_best_match_link_pypi(
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
                cached = self._get_cached_wheel()  # TODO
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

    def _get_wheel_dir(self) -> str:
        assert self.candidate.link
        wheel_cache = self.project.project_cache.make_wheel_cache()
        if self.should_cache():
            logger.info("Saving wheel to cache: %s", self.candidate.link)
            return wheel_cache.get_path_for_link(
                self.candidate.link, self.environment.target_python
            ).as_posix()
        else:
            return wheel_cache.get_ephemeral_path_for_link(  # TODO
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
        builder_cls = EditableBuilder if self.req.editable else WheelBuilder
        builder = builder_cls(
            str(self._unpacked_dir), self.environment
        )  # TODO: refine init, use kwargs.
        build_dir = self._get_wheel_dir()
        os.makedirs(build_dir, exist_ok=True)
        logger.info(
            "Running PEP 517 backend to build a wheel for %s", self.link
        )  # TODO: really?
        self.wheel = Path(
            builder.build(build_dir, metadata_directory=self._metadata_dir)  # TODO
        )
        return self.wheel


class PreparedMojoCandidate(BasePreparedCandidate):
    ui: UI = ui
    project_cache: ProjectCache | None = None

    def __init__(
        self,
        candidate: Candidate,
        environment: MojoEnvironment | None = None,
        project_cache: ProjectCache | None = None,
        ui: UI = ui,
    ) -> None:
        if environment is None:
            self.environment = MojoEnvironment(project=Project())
        self.project = self.environment.project
        project_cache = project_cache or self.project.project_cache
        super().__init__(candidate=candidate, project_cache=project_cache)

        self.ring: Path | None = None
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
            if self.link.is_file and self.link.file_path.is_dir():  # TODO: why white?
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
        supported_tags = (
            self.environment.target_mojo.supported_tags()
        )  # TODO: make mups supported tags
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
            if self.environment.is_local:  # TODO: remove, dead pep582 path
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
        ring_cache = self.project.project_cache.make_ring_cache()
        assert self.candidate.link
        cache_entry = ring_cache.get(
            self.candidate.link, self.candidate.name, self.environment.target_mojo
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
        Builder = EditableBuilder if self.req.editable else WheelBuilder
        try:
            logger.info("Running PEP 517 backend to get metadata for %s", self.link)
            self._metadata_dir = Builder(source_dir, self.environment).prepare_metadata(
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
        mojoproject_toml = self._unpacked_dir / DEFAULT_MOJOPROJECT_FILENAME
        if not force_build and mojoproject_toml.exists():
            dist = self._get_metadata_from_project(mojoproject_toml)  # TODO
            if dist is not None:
                return dist

        # If all fail, try building the source to get the metadata
        return self._get_metadata_from_build(self._unpacked_dir, metadata_parent)

    @property
    def metadata(self) -> im.Distribution:  # TODO: use toml as return value
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
        ring_cache = self.project.project_cache.make_ring_cache()
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
        ring_cache = self.project.project_cache.make_ring_cache()
        assert self.candidate.link
        cache_entry = ring_cache.get(
            self.candidate.link, self.candidate.name, self.environment.target_python
        )  # TODO
        if cache_entry is not None:
            logger.info("Using cached ring: %s", cache_entry)
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
            EditableBuilder if self.req.editable else RingBuilder
        )  # TODO: RingBuilder
        builder = builder_cls(str(self._unpacked_dir), self.environment)  # TODO
        build_dir = self._get_ring_dir()
        os.makedirs(build_dir, exist_ok=True)
        logger.info("Running PEP 517 backend to build a wheel for %s", self.link)
        self.ring = Path(
            builder.build(build_dir, metadata_directory=self._metadata_dir)  # TODO
        )
        return self.ring


def prepare(
    candidate: Candidate,
    environment: BaseEnvironment | None = None,
    is_mojo: bool = True,
) -> BasePreparedCandidate:
    """Prepare the candidate for installation."""
    if candidate._prepared is None:
        if is_mojo:
            candidate._prepared = PreparedMojoCandidate(
                candidate=candidate, environment=environment
            )
        else:
            candidate._prepared = PreparedPythonCandidate(
                candidate=candidate, environment=environment
            )
    return candidate._prepared


def make_project_a_candidate(
    environment: BaseEnvironment, project: Project | None = None, editable: bool = True
) -> Candidate:
    project = project or environment.project
    req = parse_requirement(path_to_url(project.root.as_posix()), editable)
    assert project.name
    req.name = project.name
    can = make_candidate(req, name=project.name, link=Link.from_path(project.root))
    prepare(candidate=can, environment=environment).metadata
    return can
