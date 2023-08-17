from __future__ import annotations

import collections
import contextlib
import hashlib
import os
import re
import shutil
import sys
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping, cast

import platformdirs
import tomlkit
from mups import parse_ring_filename
from packaging.specifiers import SpecifierSet
from tomlkit.items import Array

from .._types import RepositoryConfig
from ..auth import MuseBasicAuth
from ..exceptions import MuseUsageError, deprecation_warning
from ..models.backends import BuildBackend, get_backend_by_spec
from ..models.caches import HashCache
from ..models.candidates import BasePreparedCandidate, Candidate, make_candidate
from ..models.link import Link
from ..models.repositories import BaseRepository, LockedRepository, MojoPIRepository
from ..models.requirements import (
    BaseMuseRequirement,
    FileMuseRequirement,
    VcsMuseRequirement,
    parse_requirement,
    strip_extras,
)
from ..models.vcs import vcs_support
from ..termui import UI, SilentSpinner, Spinner, ui
from ..utils import (
    cd,
    expand_env_vars_in_auth,
    find_project_root,
    get_rev_from_url,
    path_to_url,
    url_without_fragments,
)
from .config import Config
from .lockfile import Lockfile
from .project_file import MojoProject


class Project:
    """Core project class.

    Args:
        core: The core instance.
        root_path: The root path of the project.
        is_global: Whether the project is global.
        global_config: The path to the global config file.
    """

    MOJOPROJECT_FILENAME = "mojoproject.toml"
    LOCKFILE_FILENAME = "muse.lock"
    DEPENDENCIES_RE = re.compile(r"(?:(.+?)-)?dependencies")

    def __init__(
        self,
        root_path: str | Path | None,
        ui: UI = ui,
        is_global: bool = False,
        global_config: str | Path | None = None,
    ) -> None:
        self.ui = ui
        self._lockfile: Lockfile | None = None
        # self._python: PythonInfo | None = None
        self.auth = MuseBasicAuth(self.ui, self.sources)

        if global_config is None:
            global_config = platformdirs.user_config_path("muse") / "config.toml"
        self.global_config = Config(Path(global_config), is_global=True)
        global_project = Path(self.global_config["global_project.path"])

        if root_path is None:
            root_path = (
                find_project_root(max_depth=self.global_config["project_max_depth"])
                if not is_global
                else global_project
            )
        if (
            not is_global
            and root_path is None
            and self.global_config["global_project.fallback"]
        ):
            root_path = global_project
            is_global = True
            if self.global_config["global_project.fallback_verbose"]:
                self.ui.echo(
                    "Project is not found, fallback to the global project",
                    style="warning",
                    err=True,
                )

        self.root: Path = Path(root_path or "").absolute()
        self.is_global = is_global
        self.init_global_project()

    def init_global_project(self) -> None:
        if not self.is_global or not self.mojoproject.empty():
            return
        self.root.mkdir(parents=True, exist_ok=True)
        self.mojoproject.set_data(
            {"project": {"dependencies": ["pip", "setuptools", "wheel"]}}
        )
        self.mojoproject.write()

    @cached_property
    def mojoproject(self) -> MojoProject:
        return MojoProject(self.root / self.MOJOPROJECT_FILENAME, ui=self.ui)

    @cached_property
    def config(self) -> Mapping[str, Any]:
        """A read-only dict configuration"""
        return collections.ChainMap(self.project_config, self.global_config)

    @property
    def default_source(self) -> RepositoryConfig:
        """Get the default source from the mojopi setting"""
        return RepositoryConfig(
            config_prefix="mojopi",
            name="mojopi",
            url=self.config["mojopi.url"],
            verify_ssl=self.config["mojopi.verify_ssl"],
            username=self.config.get("mojopi.username"),
            password=self.config.get("mojopi.password"),
        )

    @cached_property
    def project_config(self) -> Config:
        """Read-and-writable configuration dict for project settings"""
        return Config(self.root / "muse.toml")

    @property
    def sources(self) -> list[RepositoryConfig]:
        result: dict[str, RepositoryConfig] = {}
        for source in self.mojoproject.settings.get("source", []):
            result[source["name"]] = RepositoryConfig(**source, config_prefix="mojopi")

        def merge_sources(other_sources: Iterable[RepositoryConfig]) -> None:
            for source in other_sources:
                name = source.name
                if name in result:
                    result[name].passive_update(source)
                else:
                    result[name] = source

        if not self.config.get("mojopi.ignore_stored_index", False):
            if "mojopi" not in result:  # put mojopi source at the beginning
                result = {"mojopi": self.default_source, **result}
            else:
                result["mojopi"].passive_update(self.default_source)
            merge_sources(self.project_config.iter_sources())
            merge_sources(self.global_config.iter_sources())
        sources: list[RepositoryConfig] = []
        for source in result.values():
            if not source.url:
                continue
            source.url = expand_env_vars_in_auth(source.url)
            sources.append(source)
        return sources

    def __repr__(self) -> str:
        return f"<Project '{self.root.as_posix()}'>"

    @property
    def lockfile(self) -> Lockfile:
        if self._lockfile is None:
            self._lockfile = Lockfile(self.root / self.LOCKFILE_FILENAME, ui=self.ui)
        return self._lockfile

    def set_lockfile(self, path: str | Path) -> None:
        self._lockfile = Lockfile(path, ui=self.ui)

    @property
    def scripts(self) -> dict[str, str | dict[str, str]]:
        return self.mojoproject.settings.get("scripts", {})

    @property
    def name(self) -> str | None:
        return self.mojoproject.metadata.get("name")

    @property
    def requires_mojo(self) -> SpecifierSet:
        return SpecifierSet(self.mojoproject.metadata.get("requires-mojo", ""))

    def get_dependencies(
        self, group: str | None = None
    ) -> dict[str, BaseMuseRequirement]:
        metadata = self.mojoproject.metadata
        group = group or "default"
        optional_dependencies = metadata.get("optional-dependencies", {})
        dev_dependencies = self.mojoproject.settings.get("dev-dependencies", {})
        in_metadata = group == "default" or group in optional_dependencies
        if group == "default":
            deps: list[str] = metadata.get("dependencies", [])
        else:
            if group in optional_dependencies and group in dev_dependencies:
                self.ui.echo(
                    f"The {group} group exists in both [optional-dependencies] "
                    "and [dev-dependencies], the former is taken.",
                    err=True,
                    style="warning",
                )
            if group in optional_dependencies:
                deps: list[str] = optional_dependencies[group]
            elif group in dev_dependencies:
                deps: list[str] = dev_dependencies[group]
            else:
                raise MuseUsageError(f"Non-exist group {group}")
        result = {}
        with cd(self.root):
            for line in deps:
                if line.startswith("-e "):
                    if in_metadata:
                        self.ui.echo(
                            f"WARNING: Skipping editable dependency [b]{line}[/] in the"
                            r" [success]\[project][/] table. Please move it to the "
                            r"[success]\[tool.muse.dev-dependencies][/] table",
                            err=True,
                            style="warning",
                        )
                        continue
                    req = parse_requirement(line[3:].strip(), True)
                else:
                    req = parse_requirement(line)
                # make editable packages behind normal ones to override correctly.
                result[req.identify()] = req
        return result

    def iter_groups(self) -> Iterable[str]:
        groups = {"default"}
        if self.mojoproject.metadata.get("optional-dependencies"):
            groups.update(self.mojoproject.metadata["optional-dependencies"].keys())
        if self.mojoproject.settings.get("dev-dependencies"):
            groups.update(self.mojoproject.settings["dev-dependencies"].keys())
        return groups

    @property
    def all_dependencies(self) -> dict[str, dict[str, BaseMuseRequirement]]:
        return {group: self.get_dependencies(group) for group in self.iter_groups()}

    @property
    def allow_prereleases(self) -> bool | None:
        return self.mojoproject.settings.get("allow_prereleases")

    def get_repository(
        self, cls: type[BaseRepository] | None = None, ignore_compatibility: bool = True
    ) -> BaseRepository:
        """Get the repository object"""
        if cls is None:
            # cls = self.core.repository_class  # TODO: need to investigate
            cls = MojoPIRepository
        sources = self.sources or []
        return cls(sources, ignore_compatibility=ignore_compatibility)

    @property
    def locked_repository(self) -> LockedRepository:
        lockfile = self.lockfile._data.unwrap()
        # except ProjectError:
        #     lockfile = {}

        return LockedRepository(lockfile, self.sources)

    def get_lock_metadata(self) -> dict[str, Any]:
        content_hash = "sha256:" + self.mojoproject.content_hash("sha256")
        return {
            "lock_version": self.lockfile.spec_version,
            "content_hash": content_hash,
        }

    def write_lockfile(
        self,
        toml_data: dict,
        show_message: bool = True,
        write: bool = True,
        **_kwds: Any,
    ) -> None:
        """Write the lock file to disk."""
        if _kwds:
            deprecation_warning(
                "Extra arguments have been moved to `format_lockfile` function",
                stacklevel=2,
            )
        toml_data["metadata"].update(self.get_lock_metadata())
        self.lockfile.set_data(toml_data)

        if write:
            self.lockfile.write(show_message)

    def make_self_candidate(self, editable: bool = True) -> Candidate:
        req = parse_requirement(path_to_url(self.root.as_posix()), editable)
        assert self.name
        req.name = self.name
        can = make_candidate(req, name=self.name, link=Link.from_path(self.root))
        can.prepare(self.environment).metadata  # TODO
        return can

    def is_lockfile_hash_match(self) -> bool:
        hash_in_lockfile = str(self.lockfile.hash)
        if not hash_in_lockfile:
            return False
        algo, hash_value = hash_in_lockfile.split(":")
        content_hash = self.mojoproject.content_hash(algo)
        return content_hash == hash_value

    def use_mojoproject_dependencies(
        self, group: str, dev: bool = False
    ) -> tuple[list[str], Callable[[list[str]], None]]:
        """Get the dependencies array and setter in the mojoproject.toml
        Return a tuple of two elements, the first is the dependencies array,
        and the second value is a callable to set the dependencies array back.
        """

        def update_dev_dependencies(deps: list[str]) -> None:
            from tomlkit.container import OutOfOrderTableProxy

            settings.setdefault("dev-dependencies", {})[group] = deps
            if isinstance(self.mojoproject._data["tool"], OutOfOrderTableProxy):
                # In case of a separate table, we have to remove and re-add it to make the write correct.
                # This may change the order of tables in the TOML file, but it's the best we can do.
                # see bug pdm-project/pdm#2056 for details
                del self.mojoproject._data["tool"]["muse"]
                self.mojoproject._data["tool"]["muse"] = settings

        metadata, settings = self.mojoproject.metadata, self.mojoproject.settings
        if group == "default":
            return metadata.get(
                "dependencies", tomlkit.array()
            ), lambda x: metadata.__setitem__("dependencies", x)
        deps_setter = [
            (
                metadata.get("optional-dependencies", {}),
                lambda x: metadata.setdefault("optional-dependencies", {}).__setitem__(
                    group, x
                ),
            ),
            (settings.get("dev-dependencies", {}), update_dev_dependencies),
        ]
        for deps, setter in deps_setter:
            if group in deps:
                return deps[group], setter
        # If not found, return an empty list and a setter to add the group
        return tomlkit.array(), deps_setter[int(dev)][1]

    def add_dependencies(
        self,
        requirements: dict[str, BaseMuseRequirement],
        to_group: str = "default",
        dev: bool = False,
        show_message: bool = True,
    ) -> None:
        deps, setter = self.use_mojoproject_dependencies(to_group, dev)
        for _, dep in requirements.items():
            matched_index = next(
                (i for i, r in enumerate(deps) if dep.matches(r)),
                None,
            )
            req = dep.as_line()
            if matched_index is None:
                deps.append(req)
            else:
                deps[matched_index] = req
        setter(cast(Array, deps).multiline(True))
        self.mojoproject.write(show_message)

    @property
    def backend(self) -> BuildBackend:
        return get_backend_by_spec(self.mojoproject.build_system)(self.root)

    @property
    def cache_dir(self) -> Path:
        return Path(self.config.get("cache_dir", ""))

    def cache(self, name: str) -> Path:
        path = self.cache_dir / name
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError:
            # The path could be not accessible
            pass
        return path

    def make_hash_cache(self) -> HashCache:
        return HashCache(directory=self.cache("hashes"))


def _filter_none(data: dict[str, Any]) -> dict[str, Any]:
    """Return a new dict without None values"""
    return {k: v for k, v in data.items() if v is not None}


class PreparedCandidate(BasePreparedCandidate):
    ui: UI = ui
    project: Project

    def __init__(self, candidate: Candidate, project: Project, ui: UI = ui) -> None:
        super().__init__(candidate)
        self.project = project
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
                hash_cache = self.project.make_hash_cache()
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

    def _ring_compatible(self, wheel_file: str, allow_all: bool = False) -> bool:
        if allow_all:
            return True
        # supported_tags = self.environment.target_python.supported_tags()
        file_tags = parse_ring_filename(wheel_file)[-1]
        return not file_tags.isdisjoint(supported_tags)

    def obtain(self, allow_all: bool = False, unpack: bool = True) -> None:
        """Fetch the link of the candidate and unpack to local if necessary.

        :param allow_all: If true, don't validate the wheel tag nor hashes
        :param unpack: Whether to download and unpack the link if it's not local
        """
        if self.wheel:
            if self._wheel_compatible(self.wheel.name, allow_all):
                return
        elif self._source_dir and self._source_dir.exists():
            return

        with self.environment.get_finder() as finder:
            if (
                not self.link
                or self.link.is_wheel
                and not self._wheel_compatible(self.link.filename, allow_all)
            ):
                if self.req.is_file_or_url:
                    raise CandidateNotFound(
                        f"The URL requirement {self.req.as_line()} is a wheel but incompatible"
                    )
                self.link = self.wheel = None  # reset the incompatible wheel
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
                cached = self._get_cached_wheel()
                if cached:
                    self.wheel = cached
                    return
            if unpack:
                self._unpack(validate_hashes=not allow_all)

    def build(self) -> Path:
        """Call PEP 517 build hook to build the candidate into a wheel"""
        self.obtain(allow_all=False)
        if self.wheel:
            return self.wheel
        if not self.req.editable:
            cached = self._get_cached_wheel()
            if cached:
                self.wheel = cached
                return self.wheel
        assert self._source_dir, "Source directory isn't ready yet"
        builder_cls = EditableBuilder if self.req.editable else WheelBuilder
        builder = builder_cls(str(self._unpacked_dir), self.environment)
        build_dir = self._get_wheel_dir()
        os.makedirs(build_dir, exist_ok=True)
        termui.logger.info("Running PEP 517 backend to build a wheel for %s", self.link)
        self.wheel = Path(
            builder.build(build_dir, metadata_directory=self._metadata_dir)
        )
        return self.wheel


def prepare(candidate: Candidate) -> PreparedCandidate:
    """Prepare the candidate for installation."""
    if candidate._prepared is None:
        candidate._prepared = PreparedCandidate(candidate)
    return candidate._prepared
