from __future__ import annotations

import base64
import collections
import contextlib
import hashlib
import importlib.metadata as im
import os
import re
import shutil
import sys
import warnings
from contextlib import contextmanager
from functools import cached_property
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Callable, Generator, Iterable, Mapping, cast

import packaging
import platformdirs
import tomlkit
from findpython import BaseProvider, Finder, PythonVersion
from mups import parse_ring_filename
from packaging.specifiers import SpecifierSet
from packaging.tags import _32_BIT_INTERPRETER
from tomlkit.items import Array

from .auth import MuseBasicAuth, RepositoryConfigWithPassword
from .evaluator import TargetPython
from .exceptions import (
    BuildError,
    CandidateNotFound,
    MuseUsageError,
    NoPythonVersion,
    deprecation_warning,
)
from .finders import PyPackageFinder
from .in_process import get_python_abi_tag, get_uname
from .models.backends import BuildBackend, get_backend_by_spec
from .models.caches import HashCache, ProjectCache
from .models.candidates import (
    BasePreparedCandidate,
    Candidate,
    MetadataDistribution,
    make_candidate,
)
from .models.config import DEFAULT_CONFIG_FILENAME, Config
from .models.info import MojoInfo, PythonInfo
from .models.link import Link
from .models.lockfile import Lockfile
from .models.project_file import MojoProjectFile, PyProjectFile
from .models.repositories import BaseRepository, LockedRepository, MojoPIRepository
from .models.requirements import (
    BaseMuseRequirement,
    FileMuseRequirement,
    VcsMuseRequirement,
    filter_requirements_with_extras,
    parse_requirement,
    strip_extras,
)
from .models.setup import Setup
from .models.vcs import vcs_support
from .models.venv import VirtualEnv, get_in_project_venv, get_venv_python
from .session import MuseSession, PyPISession
from .termui import UI, SilentSpinner, Spinner, logger, ui
from .utils import (
    DEFAULT_MOJOPROJECT_FILENAME,
    DEFAULT_PYPROJECT_FILENAME,
    FileHash,
    RepositoryConfig,
    cd,
    convert_hashes,
    create_tracked_tempdir,
    expand_env_vars_in_auth,
    find_project_root,
    find_python_in_path,
    get_rev_from_url,
    get_trusted_hosts,
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
    PYPROJECT_FILENAME = DEFAULT_PYPROJECT_FILENAME
    LOCKFILE_FILENAME = "muse.lock"
    DEPENDENCIES_RE = re.compile(r"(?:(.+?)-)?dependencies")

    def __init__(
        self,
        root_path: str | Path | None = None,
        python: str | None = None,
        ui: UI = ui,
        is_global: bool = False,
        global_config: str | Path | None = None,
    ) -> None:
        self.ui = ui
        self._lockfile: Lockfile | None = None
        self._python: PythonInfo | None = None
        self._mojo: MojoInfo | None = None

        if global_config is None:
            global_config = platformdirs.user_config_path("muse") / "config.toml"
        self.global_config_path = Path(global_config)
        self.global_config = Config(self.global_config_path, is_global=True)
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

        self.project_cache = ProjectCache(
            root_path=self.root, global_config=self.global_config_path
        )

        if python is None:
            self._interpreter = self.python
        else:
            self._interpreter = PythonInfo.from_path(python)

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
    def pyproject(self) -> PyProjectFile:
        return PyProjectFile(self.root / self.PYPROJECT_FILENAME, ui=self.ui)

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
    def sources(self) -> list[RepositoryConfig]:  # TODO: sources_python, sources_mojo?
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
            def create(cls):  # -> Self | None
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
                venv_path = self._create_virtualenv()  # TODO
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
            f"No Python that satisfies {self.requires_python} is found on the system."
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

    @property
    def mojo(self) -> MojoInfo:
        if not self._mojo:
            pass  # TODO: self.resolve_mojo_compiler()
        return self._mojo

    @property
    def interpreter(self) -> PythonInfo:
        return self._interpreter

    @cached_property
    def target_python(self) -> TargetPython:
        python_version = self.interpreter.version_tuple
        python_abi_tag = get_python_abi_tag(str(self.interpreter.executable))
        return TargetPython(python_version, [python_abi_tag])

    @property
    def _saved_python(self) -> str | None:
        if os.getenv("PDM_PYTHON"):
            return os.getenv("PDM_PYTHON")
        with contextlib.suppress(FileNotFoundError):
            return self.root.joinpath(".pdm-python").read_text("utf-8").strip()
        with contextlib.suppress(FileNotFoundError):
            # TODO: remove this in the future
            with self.root.joinpath(".pdm.toml").open("rb") as fp:
                data = tomlkit.load(fp)
                if data.get("python", {}).get("path"):
                    return data["python"]["path"]
        return None

    @_saved_python.setter
    def _saved_python(self, value: str | None) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        python_file = self.root.joinpath(".pdm-python")
        if value is None:
            with contextlib.suppress(FileNotFoundError):
                python_file.unlink()
            return
        python_file.write_text(value, "utf-8")

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


def prepare(
    candidate: Candidate, project: Project | None = None, is_mojo: bool = True
) -> BasePreparedCandidate:
    """Prepare the candidate for installation."""
    if candidate._prepared is None:
        if is_mojo:
            candidate._prepared = PreparedMojoCandidate(
                candidate=candidate, project=project
            )
        else:
            candidate._prepared = PreparedPythonCandidate(
                candidate=candidate, project=project
            )
    return candidate._prepared


# original finders.py


def _build_pypi_session(
    project: Project, trusted_hosts: list[str], auth: MuseBasicAuth | None
) -> PyPISession:
    if auth is None:
        auth = MuseBasicAuth(ui=ui, sources=project.sources)
    ca_certs = project.config.get("pypi.ca_certs")
    session = PyPISession(
        cache_dir=project.project_cache.cache("http"),
        trusted_hosts=trusted_hosts,
        ca_certificates=Path(ca_certs) if ca_certs is not None else None,
    )
    certfn = project.config.get("pypi.client_cert")
    if certfn:
        keyfn = project.config.get("pypi.client_key")
        session.cert = (Path(certfn), Path(keyfn) if keyfn else None)

    session.auth = auth
    return session


@contextmanager
def _patch_target_python(
    project: Project, python: str | None = None
) -> Generator[None, None, None]:
    """Patch the packaging modules to respect the arch of target python."""
    if python is None:
        interpreter = project.python
    else:
        interpreter = PythonInfo.from_path(python)

    old_32bit = _32_BIT_INTERPRETER
    old_os_uname = getattr(os, "uname", None)

    if old_os_uname is not None:

        def uname() -> os.uname_result:
            return get_uname(str(interpreter.executable))

        os.uname = uname
    _32_BIT_INTERPRETER = interpreter.is_32bit
    try:
        yield
    finally:
        _32_BIT_INTERPRETER = old_32bit
        if old_os_uname is not None:
            os.uname = old_os_uname


@contextmanager
def get_pypi_finder(
    project: Project,
    sources: list[RepositoryConfig] | None = None,
    ignore_compatibility: bool = False,
) -> Generator[PyPackageFinder, None, None]:
    """Return the package finder of given index sources.

    :param sources: a list of sources the finder should search in.
    :param ignore_compatibility: whether to ignore the python version
        and wheel tags.
    """

    if sources is None:
        sources = project.sources
    if not sources:
        raise MuseUsageError(
            "You must specify at least one index in pyproject.toml or config.\n"
            "The 'pypi.ignore_stored_index' config value is "
            f"{project.config['pypi.ignore_stored_index']}"
        )

    trusted_hosts = get_trusted_hosts(sources)

    session = _build_pypi_session(trusted_hosts)
    with _patch_target_python():
        finder = PyPackageFinder(
            session=session,
            target_python=project.target_python,
            ignore_compatibility=ignore_compatibility,
            no_binary=os.getenv("PDM_NO_BINARY", "").split(","),
            only_binary=os.getenv("PDM_ONLY_BINARY", "").split(","),
            prefer_binary=os.getenv("PDM_PREFER_BINARY", "").split(","),
            respect_source_order=project.pyproject.settings.get("resolution", {}).get(
                "respect-source-order", False
            ),
            verbosity=project.ui.verbosity,
        )
        for source in sources:
            assert source.url
            if source.type == "find_links":
                finder.add_find_links(source.url)
            else:
                finder.add_index_url(source.url)
        try:
            yield finder
        finally:
            session.close()


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


class PreparedPythonCandidate(BasePreparedCandidate):
    ui: UI = ui
    project: Project | None = (None,)
    project_cache: ProjectCache | None = (None,)

    def __init__(
        self,
        candidate: Candidate,
        project: Project | None = None,
        project_cache: ProjectCache | None = None,
        ui: UI = ui,
    ) -> None:
        if project is None:
            project = Project()
        project_cache = project_cache or project.project_cache
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

    def _ring_compatible(self, ring_file: str, allow_all: bool = False) -> bool:
        if allow_all:
            return True
        supported_tags = (
            self.project.target_python.supported_tags()
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


class PreparedMojoCandidate(BasePreparedCandidate):
    ui: UI = ui
    project_cache: ProjectCache | None = None

    def __init__(
        self,
        candidate: Candidate,
        project: Project | None = None,
        project_cache: ProjectCache | None = None,
        ui: UI = ui,
    ) -> None:
        if project is None:
            project = Project()
        project_cache = project_cache or project.project_cache
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
