import base64
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, Mapping, cast

from packaging.version import Version
from resolvelib import ResolutionImpossible, Resolver
from resolvelib.resolvers import RequirementInformation

from .. import termui
from ..exceptions import MuseUsageError, NoPythonVersion
from ..models.caches import JSONFileCache
from ..models.candidates import Candidate
from ..models.python import PythonInfo
from ..models.requirements import BaseMuseRequirement
from ..models.specifiers import PySpecSet, get_specifier
from ..models.venv import VirtualEnv, get_venv_python
from ..project import BaseEnvironment, BaseRepository, Project, create_venv
from ..resolver import resolve_python
from ..resolver.providers import get_provider
from ..resolver.python import PythonRequirement
from ..resolver.reporters import BaseReporter
from ..utils import comparable_version


def get_in_project_venv(root: Path) -> VirtualEnv | None:
    """Get the python interpreter path of venv-in-project"""
    for possible_dir in (".venv", "venv", "env"):
        venv = VirtualEnv.get(root / possible_dir)
        if venv is not None:
            return venv
    return None


def hash_path(path: str) -> str:
    """Generate a hash for the given path."""
    return base64.urlsafe_b64encode(
        hashlib.new("md5", path.encode(), usedforsecurity=False).digest()
    ).decode()[:8]


def get_venv_prefix(project: Project) -> str:
    """Get the venv prefix for the project"""
    path = project.root
    name_hash = hash_path(path.as_posix())
    return f"{path.name}-{name_hash}-"


def iter_venvs(project: Project) -> Iterable[tuple[str, VirtualEnv]]:
    """Return an iterable of venv paths associated with the project"""
    in_project_venv = get_in_project_venv(project.root)
    if in_project_venv is not None:
        yield "in-project", in_project_venv
    venv_prefix = get_venv_prefix(project)
    venv_parent = Path(project.config["venv.location"])
    for path in venv_parent.glob(f"{venv_prefix}*"):
        ident = path.name[len(venv_prefix) :]
        venv = VirtualEnv.get(path)
        if venv is not None:
            yield ident, venv


def get_venv_with_name(project: Project, name: str) -> VirtualEnv:
    all_venvs = dict(iter_venvs(project))
    try:
        return all_venvs[name]
    except KeyError:
        raise MuseUsageError(
            f"No virtualenv with key '{name}' is found, must be one of {list(all_venvs)}.\n"
            "You can create one with 'pdm venv create'.",
        ) from None


def select_python(
    project: Project,
    python: str,
    *,
    ignore_remembered: bool,
    ignore_requires_python: bool,
    venv: str | None,
    first: bool,
) -> PythonInfo:
    def version_matcher(py_version: PythonInfo) -> bool:
        return py_version.valid and (
            ignore_requires_python
            or project.requires_python.contains(py_version.version, True)
        )

    if venv:
        virtualenv = get_venv_with_name(project, venv)
        return PythonInfo.from_path(virtualenv.interpreter)

    if not project.project_cache.cache_dir.exists():
        project.project_cache.cache_dir.mkdir(parents=True)
    use_cache: JSONFileCache[str, str] = JSONFileCache(
        project.project_cache.cache_dir / "use_cache.json"
    )
    python = python.strip()
    if python and not ignore_remembered and python in use_cache:
        path = use_cache.get(python)
        cached_python = PythonInfo.from_path(path)
        if not cached_python.valid:
            project.ui.echo(
                f"The last selection is corrupted. {path!r}",
                style="error",
                err=True,
            )
        elif version_matcher(cached_python):
            project.ui.echo(
                "Using the last selection, add '-i' to ignore it.",
                style="warning",
                err=True,
            )
            return cached_python

    found_interpreters = list(dict.fromkeys(project.find_interpreters(python)))
    matching_interpreters = list(filter(version_matcher, found_interpreters))
    if not found_interpreters:
        raise NoPythonVersion(
            f"No Python interpreter matching [success]{python}[/] is found."
        )
    if not matching_interpreters:
        project.ui.echo("Interpreters found but not matching:", err=True)
        for py in found_interpreters:
            info = py.identifier if py.valid else "Invalid"
            project.ui.echo(f"  - {py.path} ({info})", err=True)
        raise NoPythonVersion(
            f"No python is found meeting the requirement [success]python {project.requires_python!s}[/]"
        )
    if first or len(matching_interpreters) == 1:
        return matching_interpreters[0]

    project.ui.echo("Please enter the Python interpreter to use")
    for i, py_version in enumerate(matching_interpreters):
        project.ui.echo(
            f"{i}. [success]{py_version.path!s}[/] ({py_version.identifier})"
        )
    selection = termui.ask(
        "Please select",
        default="0",
        prompt_type=int,
        choices=[str(i) for i in range(len(matching_interpreters))],
        show_choices=False,
    )
    return matching_interpreters[int(selection)]


def do_use(
    environment: BaseEnvironment,
    project: Project | None = None,
    python: str = "",
    first: bool = False,
    ignore_remembered: bool = False,
    ignore_requires_python: bool = False,
    save: bool = True,
    venv: str | None = None,
    # hooks: HookManager | None = None,
) -> PythonInfo:
    """Use the specified python version and save in project config.
    The python can be a version string or interpreter path.
    """
    project = project or environment.project

    selected_python = select_python(
        project,
        python,
        ignore_remembered=ignore_remembered,
        first=first,
        venv=venv,
        ignore_requires_python=ignore_requires_python,
    )
    if python:
        use_cache: JSONFileCache[str, str] = JSONFileCache(
            project.project_cache.cache_dir / "use_cache.json"
        )
        use_cache.set(python, selected_python.path.as_posix())

    if not save:
        return selected_python

    saved_python = project._saved_python
    old_python = PythonInfo.from_path(saved_python) if saved_python else None
    project.ui.echo(
        f"Using Python interpreter: [success]{selected_python.path!s}[/] ({selected_python.identifier})"
    )
    project.python = selected_python
    # if environment.is_local:
    #     project.ui.echo(
    #         "Using __pypackages__ because non-venv Python is used.",
    #         style="primary",
    #         err=True,
    #     )
    # if (
    #     old_python
    #     and old_python.executable != selected_python.executable
    #     and isinstance(environment, PythonLocalEnvironment)
    # ):
    #     project.ui.echo("Updating executable scripts...", style="primary")
    #     environment.update_shebangs(selected_python.executable.as_posix())

    # hooks = hooks or HookManager(project)
    # hooks.try_emit("post_use", python=selected_python)
    return selected_python


def set_python(
    environment: BaseEnvironment,
    project: Project | None = None,
    python: str | None = None,
    interactive: bool = True,
):
    project = project or environment.project

    if interactive:
        python_info = do_use(
            environment,
            project,
            python or "",
            first=bool(python),
            ignore_remembered=True,
            ignore_requires_python=True,
            save=False,
        )
    else:
        python_info = do_use(
            environment,
            project,
            python or "3",
            first=True,
            ignore_remembered=True,
            ignore_requires_python=True,
            save=False,
        )
    if project.config["python.use_venv"] and python_info.get_venv() is None:
        if not interactive or termui.confirm(
            f"Would you like to create a virtualenv with [success]{python_info.executable}[/]?",
            default=True,
        ):
            project._python = python_info
            try:
                path = create_venv(project=project)  # TODO: check
                python_info = PythonInfo.from_path(get_venv_python(path))
            except Exception as e:  # pragma: no cover
                project.ui.echo(
                    f"Error occurred when creating virtualenv: {e}\nPlease fix it and create later.",
                    style="error",
                    err=True,
                )
    if python_info.get_venv() is None:
        project.ui.echo(
            "You are using the PEP 582 mode, no virtualenv is created.\n"
            "For more info, please visit https://peps.python.org/pep-0582/",
            style="success",
        )
    project.python = python_info


def format_resolution_impossible(err: ResolutionImpossible) -> str:
    causes: list[RequirementInformation] = err.causes
    info_lines: set[str] = set()
    if all(isinstance(cause.requirement, PythonRequirement) for cause in causes):
        project_requires: PythonRequirement = next(
            cause.requirement for cause in causes if cause.parent is None
        )
        pyspec = cast(PySpecSet, project_requires.specifier)
        conflicting = [
            cause
            for cause in causes
            if cause.parent is not None
            and not cause.requirement.specifier.is_superset(
                pyspec
            )  # TODO: use PySpec.is_superset in specifiers.py
        ]
        result = [
            "Unable to find a resolution because the following dependencies don't work "
            "on all Python versions in the range of the project's `requires-python`: "
            f"[success]{pyspec}[/]."
        ]
        for req, parent in conflicting:
            pyspec &= req.specifier
            info_lines.add(f"  {req.as_line()} (from {parent!r})")
        result.extend(sorted(info_lines))
        if pyspec.is_impossible:
            result.append(
                "Consider changing the version specifiers of the dependencies to be compatible"
            )
        else:
            result.append(
                "A possible solution is to change the value of `requires-python` "
                f"in pyproject.toml to [success]{pyspec}[/]."
            )
        return "\n".join(result)

    if len(causes) == 1:
        return (
            "Unable to find a resolution for "
            f"[success]{causes[0].requirement.identify()}[/]\n"
            "Please make sure the package name is correct."
        )

    result = [
        "Unable to find a resolution for "
        f"[success]{causes[0].requirement.identify()}[/]\n"
        "because of the following conflicts:"
    ]
    for req, parent in causes:
        info_lines.add(f"  {req.as_line()} (from {parent if parent else 'project'})")
    result.extend(sorted(info_lines))
    result.append(
        "To fix this, you could loosen the dependency version constraints in "
        "pyproject.toml. See https://pdm.fming.dev/latest/usage/dependency/"
        "#solve-the-locking-failure for more details."
    )
    return "\n".join(result)


def save_version_specifiers(
    requirements: dict[str, dict[str, BaseMuseRequirement]],
    resolved: dict[str, Candidate],
    save_strategy: str,
) -> None:
    """Rewrite the version specifiers according to the resolved result and save strategy

    :param requirements: the requirements to be updated
    :param resolved: the resolved mapping
    :param save_strategy: compatible/wildcard/exact
    """

    def candidate_version(c: Candidate) -> Version:
        assert c.version is not None
        return comparable_version(c.version)

    for reqs in requirements.values():
        for name, r in reqs.items():
            if r.is_named and not r.specifier:
                if save_strategy == "exact":
                    r.specifier = get_specifier(
                        f"=={candidate_version(resolved[name])}"
                    )
                elif save_strategy == "compatible":
                    version = candidate_version(resolved[name])
                    if version.is_prerelease or version.is_devrelease:
                        r.specifier = get_specifier(f">={version},<{version.major + 1}")
                    else:
                        r.specifier = get_specifier(
                            f"~={version.major}.{version.minor}"
                        )
                elif save_strategy == "minimum":
                    r.specifier = get_specifier(
                        f">={candidate_version(resolved[name])}"
                    )


def populate_requirement_names(req_mapping: dict[str, BaseMuseRequirement]) -> None:
    # Update the requirement key if the name changed.
    for key, req in list(req_mapping.items()):
        if key and key.startswith(":empty:"):
            req_mapping[req.identify()] = req
            del req_mapping[key]


def fetch_hashes(repository: BaseRepository, mapping: Mapping[str, Candidate]) -> None:
    """Fetch hashes for candidates in parallel"""

    def do_fetch(candidate: Candidate) -> None:
        candidate.hashes = repository.get_hashes(candidate)

    with ThreadPoolExecutor() as executor:
        executor.map(do_fetch, mapping.values())


def resolve_candidates_from_lockfile(
    environment: BaseEnvironment,
    requirements: Iterable[BaseMuseRequirement],
    project: Project | None = None,
) -> dict[str, Candidate]:
    project = project or environment.project
    ui = project.ui
    resolve_max_rounds = int(project.config["strategy.resolve_max_rounds"])
    reqs = [
        req
        for req in requirements
        if not req.marker or req.marker.evaluate(environment.marker_environment)
    ]
    with ui.logging("install-resolve"):
        with ui.open_spinner("Resolving packages from lockfile...") as spinner:
            reporter = BaseReporter()
            provider = get_provider(for_install=True)
            resolver: Resolver = Resolver(provider, reporter)
            mapping, *_ = resolve_python(
                resolver,
                reqs,
                environment.requires_python,
                resolve_max_rounds,
            )
            spinner.update("Fetching hashes for resolved packages...")
            fetch_hashes(provider.repository, mapping)
    return mapping
