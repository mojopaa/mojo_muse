from concurrent.futures import ThreadPoolExecutor
from typing import Collection, Iterable, Mapping, cast

import tomlkit
from mups import get_username_email_from_git, normalize_name
from resolvelib import ResolutionImpossible, ResolutionTooDeep, Resolver

from .. import termui
from ..exceptions import MuseUsageError
from ..formats import array_of_inline_tables, make_array, make_inline_table
from ..formats.base import make_array, make_inline_table
from ..models.backends import _BACKENDS, DEFAULT_BACKEND, get_backend
from ..models.candidates import Candidate
from ..models.requirements import BaseMuseRequirement, parse_requirement
from ..models.specifiers import get_specifier
from ..project import BaseEnvironment, Project, check_project_file
from ..project.filters import GroupSelection
from ..project.repositories import BaseRepository, get_locked_repository, get_repository
from ..resolver import resolve_python
from ..resolver.providers import get_provider
from ..templates import MojoProjectTemplate, PyProjectTemplate
from ..termui import ask
from .plumbing import (
    do_use,
    format_resolution_impossible,
    populate_requirement_names,
    save_version_specifiers,
    set_python,
)


# init
def get_metadata_from_input(
    project: Project,
    backend: str | None = None,
    is_interactive: bool = True,
    use_pyproject: bool = False,
):
    name = ask("Project name", default=project.root.name)
    version = ask("Project version", default="0.1.0")
    description = ask("Project description", default="")
    if backend:
        build_backend = get_backend(backend)
    elif is_interactive:
        all_backends = list(_BACKENDS)
        project.ui.echo("Which build backend to use?")
        for i, backend in enumerate(all_backends):
            project.ui.echo(f"{i}. [success]{backend}[/]")
        selected_backend = ask(
            "Please select",
            prompt_type=int,
            choices=[str(i) for i in range(len(all_backends))],
            show_choices=False,
            default=0,
        )
        build_backend = get_backend(all_backends[int(selected_backend)])
    else:
        build_backend = DEFAULT_BACKEND

    license = ask("License(SPDX name)", default="MIT")
    git_user, git_email = get_username_email_from_git()

    author = ask("Author name", default=git_user)
    email = ask("Author email", default=git_email)
    python = project.python
    python_version = f"{python.major}.{python.minor}"
    requires_python = ask("Python requires", default=f">={python_version}")
    requires_mojo = ">=0.1.0"  # TODO: project.mojo, mojo_version
    if not use_pyproject:
        # TODO mojo version
        mojo = project.mojo
        if mojo is not None:
            mojo_version = f"{mojo.major}.{mojo.minor}.{mojo.micro}"
            requires_mojo = ask("Mojo requires", default=f">={mojo_version}")
        else:
            requires_mojo = ask("Mojo requires", default=">=0.1.0")

    data = {
        "project": {
            "name": name,
            "version": version,
            "description": description,
            "authors": array_of_inline_tables([{"name": author, "email": email}]),
            "license": make_inline_table({"text": license}),
            "dependencies": make_array([], True),
        },
    }
    if requires_python and requires_python != "*":
        get_specifier(requires_python)
        data["project"]["requires-python"] = requires_python

    if requires_mojo and requires_mojo != "*":
        get_specifier(requires_mojo)
        data["project"]["requires-mojo"] = requires_mojo

    if build_backend is not None:
        data["build-system"] = cast(dict, build_backend.build_system())

    return data


def _init_builtin_pyproject(
    project: Project, template_path: str | None = None, overwrite: bool = False
):  # TODO template type
    metadata = get_metadata_from_input(project=project, use_pyproject=True)
    with PyProjectTemplate(template_path) as template:
        template.generate(
            target_path=project.root, metadata=metadata, overwrite=overwrite
        )
    project.pyproject.reload()


def _init_builtin_mojoproject(
    project: Project, template_path: str | None = None, overwrite: bool = False
):
    metadata = get_metadata_from_input(project=project)
    with MojoProjectTemplate(template_path) as template:
        template.generate(
            target_path=project.root, metadata=metadata, overwrite=overwrite
        )
    project.mojoproject.reload()


# def set_python(self, project: Project, python: str | None):


def do_init(
    environment: BaseEnvironment,
    project: Project | None = None,
    use_pyproject: bool = False,
):
    # TODO: post_init hook
    project = project or environment.project
    set_python(environment=environment)
    if use_pyproject:
        _init_builtin_pyproject(project=project)
    else:
        _init_builtin_mojoproject(project=project)


# Add
def format_lockfile(
    project: Project,
    mapping: dict[str, Candidate],
    fetched_dependencies: dict[tuple[str, str | None], list[BaseMuseRequirement]],
    groups: list[str] | None = None,
    cross_platform: bool | None = None,
    static_urls: bool | None = None,
) -> dict:
    """Format lock file from a dict of resolved candidates, a mapping of dependencies
    and a collection of package summaries.
    """

    packages = tomlkit.aot()
    for _k, v in sorted(mapping.items()):
        base = tomlkit.table()
        base.update(v.as_lockfile_entry(project.root))
        base.add("summary", v.summary or "")
        deps = make_array(
            sorted(r.as_line() for r in fetched_dependencies[v.dep_key]), True
        )
        if len(deps) > 0:
            base.add("dependencies", deps)
        if v.hashes:
            collected = {}
            for item in v.hashes:
                if static_urls:
                    row = {"url": item["url"], "hash": item["hash"]}
                else:
                    row = {"file": item["file"], "hash": item["hash"]}
                inline = make_inline_table(row)
                # deduplicate and sort
                collected[tuple(row.values())] = inline
            if collected:
                base.add(
                    "files", make_array([collected[k] for k in sorted(collected)], True)
                )
        packages.append(base)
    doc = tomlkit.document()
    metadata = tomlkit.table()
    if groups is None:
        groups = list(project.iter_groups())
    metadata.update(
        {
            "groups": sorted(groups, key=lambda k: k != "default"),
            "cross_platform": cross_platform
            if cross_platform is not None
            else project.lockfile.cross_platform,
            "static_urls": static_urls
            if static_urls is not None
            else project.lockfile.static_urls,
        }
    )
    doc.add("metadata", metadata)
    doc.add("package", packages)

    return cast(dict, doc)


def fetch_hashes(repository: BaseRepository, mapping: Mapping[str, Candidate]) -> None:
    """Fetch hashes for candidates in parallel"""

    def do_fetch(candidate: Candidate) -> None:
        candidate.hashes = repository.get_hashes(candidate)

    with ThreadPoolExecutor() as executor:
        executor.map(do_fetch, mapping.values())


def do_lock(
    environment: BaseEnvironment,
    strategy: str = "all",
    tracked_names: Iterable[str] | None = None,
    requirements: list[BaseMuseRequirement] | None = None,
    dry_run: bool = False,
    refresh: bool = False,
    groups: list[str] | None = None,
    cross_platform: bool | None = None,
    static_urls: bool | None = None,
    # hooks: HookManager | None = None,
) -> dict[str, Candidate]:
    """Performs the locking process and update lockfile."""
    project = environment.project
    # hooks = hooks or HookManager(project)
    check_project_file(project)
    if static_urls is None:
        static_urls = project.lockfile.static_urls
    if refresh:
        locked_repo = get_locked_repository(project)
        repo = get_repository(project)
        mapping: dict[str, Candidate] = {}
        dependencies: dict[tuple[str, str | None], list[BaseMuseRequirement]] = {}
        with project.ui.open_spinner("Re-calculating hashes..."):
            for key, candidate in locked_repo.packages.items():
                reqs, python_requires, summary = locked_repo.candidate_info[key]
                candidate.summary = summary
                candidate.requires_python = python_requires  # TODO
                mapping[candidate.identify()] = candidate
                dependencies[candidate.dep_key] = list(map(parse_requirement, reqs))
            with project.ui.logging("lock"):
                for c in mapping.values():
                    c.hashes.clear()
                fetch_hashes(repo, mapping)
            lockfile = format_lockfile(
                project,
                mapping,
                dependencies,
                groups=project.lockfile.groups,
                static_urls=static_urls,
            )
        project.write_lockfile(lockfile)
        return mapping
    # TODO: multiple dependency definitions for the same package.
    if cross_platform is None:
        cross_platform = project.lockfile.cross_platform
    provider = get_provider(
        strategy, tracked_names, ignore_compatibility=cross_platform  # TODO: check
    )
    if not requirements:
        requirements = [
            r
            for g, deps in project.all_dependencies.items()
            if groups is None or g in groups
            for r in deps.values()
        ]
    if not cross_platform:
        this_env = environment.marker_environment
        requirements = [
            req
            for req in requirements
            if not req.marker or req.marker.evaluate(this_env)
        ]
    resolve_max_rounds = int(project.config["strategy.resolve_max_rounds"])
    ui = project.ui
    with ui.logging("lock"):
        # The context managers are nested to ensure the spinner is stopped before
        # any message is thrown to the output.
        try:
            with ui.open_spinner(title="Resolving dependencies") as spin:
                reporter = project.get_reporter(requirements, tracked_names, spin)
                resolver: Resolver = project.core.resolver_class(
                    provider, reporter
                )  # TODO
                # hooks.try_emit("pre_lock", requirements=requirements, dry_run=dry_run)
                mapping, dependencies = resolve_python(
                    resolver=resolver,
                    requirements=requirements,
                    requires_python=project.requires_python,
                    project=project,
                    max_rounds=resolve_max_rounds,
                )
                spin.update("Fetching hashes for resolved packages...")
                fetch_hashes(provider.repository, mapping)
        except ResolutionTooDeep:
            ui.echo(f"{termui.Emoji.LOCK} Lock failed", err=True)
            ui.echo(
                "The dependency resolution exceeds the maximum loop depth of "
                f"{resolve_max_rounds}, there may be some circular dependencies "
                "in your project. Try to solve them or increase the "
                f"[success]`strategy.resolve_max_rounds`[/] config.",
                err=True,
            )
            raise
        except ResolutionImpossible as err:
            ui.echo(f"{termui.Emoji.LOCK} Lock failed", err=True)
            ui.echo(format_resolution_impossible(err), err=True)
            raise ResolutionImpossible("Unable to find a resolution") from None
        else:
            data = format_lockfile(
                project,
                mapping,
                dependencies,
                groups=groups,
                cross_platform=cross_platform,
                static_urls=static_urls,
            )
            ui.echo(f"{termui.Emoji.LOCK} Lock successful")
            project.write_lockfile(data, write=not dry_run)
            # hooks.try_emit("post_lock", resolution=mapping, dry_run=dry_run)

    return mapping


def do_padd(
    environment: BaseEnvironment,
    project: Project | None = None,
    *,
    selection: GroupSelection,
    sync: bool = True,
    save: str = "compatible",
    strategy: str = "reuse",
    editables: Collection[str] = (),
    packages: Collection[str] = (),
    no_editable: bool = False,
    no_self: bool = False,
    dry_run: bool = False,
    prerelease: bool = False,
    fail_fast: bool = False,
    unconstrained: bool = False,
):
    """Add packages and install"""
    # hooks = hooks or HookManager(project)
    project = project or environment.project
    check_project_file(project)
    if editables and no_editable:
        raise MuseUsageError("Cannot use --no-editable with editable packages given.")
    group = selection.one()
    tracked_names: set[str] = set()
    requirements: dict[str, BaseMuseRequirement] = {}
    lock_groups = ["default"] if project.lockfile.empty() else project.lockfile.groups
    if lock_groups is not None and group not in lock_groups:
        project.ui.echo(
            f"Adding group [success]{group}[/] to lockfile", err=True, style="info"
        )
        lock_groups.append(group)
    if (
        group == "default"
        or not selection.dev
        and group not in project.pyproject.settings.get("dev-dependencies", {})
    ):
        if editables:
            raise MuseUsageError(
                "Cannot add editables to the default or optional dependency group"
            )

    for r in [parse_requirement(line, True) for line in editables] + [
        parse_requirement(line) for line in packages
    ]:
        if project.name and normalize_name(project.name) == r.key and not r.extras:
            project.ui.echo(
                f"Package [req]{project.name}[/] is the project itself.",
                err=True,
                style="warning",
            )
            continue
        if r.is_file_or_url:
            r.relocate(project.backend)  # type: ignore[attr-defined]
        key = r.identify()
        r.prerelease = prerelease
        tracked_names.add(key)
        requirements[key] = r

    if requirements:
        project.ui.echo(
            f"Adding packages to [primary]{group}[/] "
            f"{'dev-' if selection.dev else ''}dependencies: "
            + ", ".join(f"[req]{r.as_line()}[/]" for r in requirements.values())
        )
    all_dependencies = project.all_dependencies
    group_deps = all_dependencies.setdefault(group, {})
    if unconstrained:
        if not requirements:
            raise MuseUsageError("--unconstrained requires at least one package")
        for req in group_deps.values():
            req.specifier = get_specifier("")
    group_deps.update(requirements)
    reqs = [
        r
        for g, deps in all_dependencies.items()
        if lock_groups is None or g in lock_groups
        for r in deps.values()
    ]
    # with hooks.skipping("post_lock"):
    resolved = do_lock(
        environment,
        strategy,
        tracked_names,
        reqs,
        dry_run=True,
        # hooks=hooks,
        groups=lock_groups,
    )

    # Update dependency specifiers and lockfile hash.
    deps_to_update = group_deps if unconstrained else requirements
    save_version_specifiers({group: deps_to_update}, resolved, save)
    if not dry_run:
        project.add_dependencies(deps_to_update, group, selection.dev or False)
        project.write_lockfile(project.lockfile._data, False)
        # hooks.try_emit("post_lock", resolution=resolved, dry_run=dry_run)
    populate_requirement_names(group_deps)
    if sync:
        do_sync(
            project,
            selection=GroupSelection(project, groups=[group], default=False),
            no_editable=no_editable and tracked_names,
            no_self=no_self or group != "default",
            requirements=list(group_deps.values()),
            dry_run=dry_run,
            fail_fast=fail_fast,
            # hooks=hooks,
        )
