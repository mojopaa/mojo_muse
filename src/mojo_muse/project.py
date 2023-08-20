from __future__ import annotations

import base64
import collections
import contextlib
import hashlib
import os
import re
import shutil
import sys
import warnings
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping, Self, cast

import platformdirs
import tomlkit
from findpython import BaseProvider, Finder, PythonVersion
from mups import parse_ring_filename
from packaging.specifiers import SpecifierSet
from tomlkit.items import Array

from .auth import MuseBasicAuth, RepositoryConfigWithPassword
from .exceptions import (
    BuildError,
    CandidateNotFound,
    MuseUsageError,
    NoPythonVersion,
    deprecation_warning,
)
from .models.backends import BuildBackend, get_backend_by_spec
from .models.caches import HashCache, ProjectCache
from .models.candidates import (
    BasePreparedCandidate,
    Candidate,
    PreparedCandidate,
    make_candidate,
)
from .models.config import Config
from .models.link import Link
from .models.lockfile import Lockfile
from .models.project_file import MojoProjectFile
from .models.python import PythonInfo
from .models.repositories import BaseRepository, LockedRepository, MojoPIRepository
from .models.requirements import (
    BaseMuseRequirement,
    FileMuseRequirement,
    VcsMuseRequirement,
    filter_requirements_with_extras,
    parse_requirement,
    strip_extras,
)
from .models.vcs import vcs_support
from .models.venv import VirtualEnv, get_in_project_venv, get_venv_python
from .termui import UI, SilentSpinner, Spinner, logger, ui
from .utils import (
    DEFAULT_CONFIG_FILENAME,
    DEFAULT_MOJOPROJECT_FILENAME,
    FileHash,
    RepositoryConfig,
    cd,
    convert_hashes,
    expand_env_vars_in_auth,
    find_project_root,
    find_python_in_path,
    get_rev_from_url,
    path_to_url,
    url_without_fragments,
)

PYENV_ROOT = os.path.expanduser(os.getenv("PYENV_ROOT", "~/.pyenv"))


def hash_path(path: str) -> str:
    """Generate a hash for the given path."""
    return base64.urlsafe_b64encode(
        hashlib.new("md5", path.encode(), usedforsecurity=False).digest()
    ).decode()[:8]


class Project:
    """Core project class.

    Args:
        root_path: The root path of the project.
        ui: The UI instance.
        is_global: Whether the project is global.
        global_config: The path to the global config file.
    """

    MOJOPROJECT_FILENAME = DEFAULT_MOJOPROJECT_FILENAME
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
        self._python: PythonInfo | None = None

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

        self.project_cache = ProjectCache(
            root_path=self.root, global_config=self.global_config
        )
        self.is_global = is_global
        self.auth = MuseBasicAuth(self.ui, self.sources)
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
    def mojoproject(self) -> MojoProjectFile:
        return MojoProjectFile(self.root / self.MOJOPROJECT_FILENAME, ui=self.ui)

    @cached_property
    def config(self) -> Mapping[str, Any]:
        """A read-only dict configuration"""
        return collections.ChainMap(self.project_config, self.global_config)

    @property
    def default_source(self) -> RepositoryConfig:
        """Get the default source from the mojopi setting"""
        return RepositoryConfigWithPassword(
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
        return Config(self.root / DEFAULT_CONFIG_FILENAME)

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

    def get_venv_prefix(self) -> str:
        """Get the venv prefix for the project"""
        path = self.root
        name_hash = hash_path(path.as_posix())
        return f"{path.name}-{name_hash}-"

    def iter_venvs(self) -> Iterable[tuple[str, VirtualEnv]]:
        """Return an iterable of venv paths associated with the project"""
        in_project_venv = get_in_project_venv(self.root)
        if in_project_venv is not None:
            yield "in-project", in_project_venv
        venv_prefix = self.get_venv_prefix()
        venv_parent = Path(self.config["venv.location"])
        for path in venv_parent.glob(f"{venv_prefix}*"):
            ident = path.name[len(venv_prefix) :]
            venv = VirtualEnv.get(path)
            if venv is not None:
                yield ident, venv

    def _get_python_finder(self) -> Finder:
        class VenvProvider(BaseProvider):
            """A Python provider for project venv pythons"""

            def __init__(self, project: Project) -> None:
                self.project = project

            @classmethod
            def create(cls) -> Self | None:
                return None

            def find_pythons(self) -> Iterable[PythonVersion]:
                for _, venv in self.project.iter_venvs():
                    yield PythonVersion(
                        venv.interpreter,
                        _interpreter=venv.interpreter,
                        keep_symlink=True,
                    )

        providers: list[str] = self.config["python.providers"]
        finder = Finder(resolve_symlinks=True, selected_providers=providers or None)
        if self.config["python.use_venv"] and (not providers or "venv" in providers):
            venv_pos = providers.index("venv") if providers else 0
            finder.add_provider(VenvProvider(project=self), venv_pos)
        return finder

    def find_interpreters(self, python_spec: str | None = None) -> Iterable[PythonInfo]:
        """Return an iterable of interpreter paths that matches the given specifier,
        which can be:
            1. a version specifier like 3.7
            2. an absolute path
            3. a short name like python3
            4. None that returns all possible interpreters
        """
        config = self.config
        python: str | Path | None = None

        if not python_spec:
            if config.get("python.use_pyenv", True) and os.path.exists(PYENV_ROOT):
                pyenv_shim = os.path.join(PYENV_ROOT, "shims", "python3")
                if os.name == "nt":
                    pyenv_shim += ".bat"
                if os.path.exists(pyenv_shim):
                    yield PythonInfo.from_path(pyenv_shim)
                elif os.path.exists(pyenv_shim.replace("python3", "python")):
                    yield PythonInfo.from_path(pyenv_shim.replace("python3", "python"))
            python = shutil.which("python") or shutil.which("python3")
            if python:
                yield PythonInfo.from_path(python)
            args = []
        else:
            if not all(c.isdigit() for c in python_spec.split(".")):
                path = Path(python_spec)
                if path.exists():
                    python = find_python_in_path(python_spec)
                    if python:
                        yield PythonInfo.from_path(python)
                if len(path.parts) == 1:  # only check for spec with only one part
                    python = shutil.which(python_spec)
                    if python:
                        yield PythonInfo.from_path(python)
                return
            args = [int(v) for v in python_spec.split(".") if v != ""]
        finder = self._get_python_finder()
        for entry in finder.find_all(*args):
            yield PythonInfo(entry)
        if not python_spec:
            # Lastly, return the host Python as well
            this_python = getattr(sys, "_base_executable", sys.executable)
            yield PythonInfo.from_path(this_python)

    def resolve_interpreter(self) -> PythonInfo:
        """Get the Python interpreter path."""

        def match_version(python: PythonInfo) -> bool:
            return python.valid and self.requires_python.contains(python.version, True)

        def note(message: str) -> None:
            if not self.is_global:
                self.ui.echo(message, style="info", err=True)

        config = self.config
        saved_path = self._saved_python
        if saved_path and not os.getenv("MUSE_IGNORE_SAVED_PYTHON"):
            python = PythonInfo.from_path(saved_path)
            if match_version(python):
                return python
            else:
                note(
                    "The saved Python interpreter doesn't match the project's requirement. "
                    "Trying to find another one."
                )
            self._saved_python = None  # Clear the saved path if it doesn't match

        if (
            config.get("python.use_venv")
            and not self.is_global
            and not os.getenv("PDM_IGNORE_ACTIVE_VENV")
        ):
            # Resolve virtual environments from env-vars
            venv_in_env = os.getenv("VIRTUAL_ENV", os.getenv("CONDA_PREFIX"))
            if venv_in_env:
                python = PythonInfo.from_path(get_venv_python(Path(venv_in_env)))
                if match_version(python):
                    note(
                        f"Inside an active virtualenv [success]{venv_in_env}[/], reusing it.\n"
                        "Set env var [success]PDM_IGNORE_ACTIVE_VENV[/] to ignore it."
                    )
                    return python
            # otherwise, get a venv associated with the project
            for _, venv in self.iter_venvs():
                python = PythonInfo.from_path(venv.interpreter)
                if match_version(python):
                    note(f"Virtualenv [success]{venv.root}[/] is reused.")
                    self.python = python
                    return python

            if not self.root.joinpath("__pypackages__").exists():
                note("python.use_venv is on, creating a virtualenv for this project...")
                venv_path = self._create_virtualenv()
                self.python = PythonInfo.from_path(get_venv_python(venv_path))
                return self.python

        for py_version in self.find_interpreters():
            if match_version(py_version):
                if config.get("python.use_venv"):
                    note(
                        "[success]__pypackages__[/] is detected, using the PEP 582 mode"
                    )
                self.python = py_version
                return py_version

        raise NoPythonVersion(
            f"No Python that satisfies {self.python_requires} is found on the system."
        )

    @property
    def python(self) -> PythonInfo:
        if not self._python:
            self._python = self.resolve_interpreter()
            if self._python.major < 3:
                raise MuseUsageError(
                    "Python 2.7 has reached EOL and PDM no longer supports it. "
                    "Please upgrade your Python to 3.6 or later.",
                )
        return self._python

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

    @property
    def requires_python(self) -> SpecifierSet:
        return SpecifierSet(
            self.mojoproject.metadata.get("requires-python", "")
        )  # TODO: double check

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
                    req = parse_requirement(line[3:].strip(), editable=True)
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

    def make_hash_cache(self) -> HashCache:
        return self.project_cache.make_hash_cache()


def prepare(candidate: Candidate, project: Project) -> PreparedCandidate:
    """Prepare the candidate for installation."""
    if candidate._prepared is None:
        candidate._prepared = PreparedCandidate(candidate=candidate, project=project)
    return candidate._prepared
