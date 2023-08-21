from typing import cast

import click
from mups import get_username_email_from_git

from .formats import array_of_inline_tables, make_array, make_inline_table
from .models.backends import _BACKENDS, DEFAULT_BACKEND, get_backend
from .models.specifiers import get_specifier
from .project import Project
from .templates import MojoProjectTemplate, PyProjectTemplate
from .termui import ask


@click.group()
def main():  # subcommand uses main as decorator. main.command().
    pass


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
def set_python():
    pass


def do_init(project: Project, use_pyproject: bool = False):
    # TODO: post_init hook
    set_python()
    if use_pyproject:
        _init_builtin_pyproject(project=project)
    else:
        _init_builtin_mojoproject(project=project)


def handle_init(project: Project, is_interactive: bool = True):
    use_pyproject: bool = False
    if project.mojoproject.exists():
        project.ui.echo(
            "mojoproject.toml already exists, update it now.", style="primary"
        )
    elif project.pyproject.exists():
        question = (
            "pyproject.toml already exists, do you want to use pyproject.toml instead?"
        )
        use_pyproject: bool = ask(question, default="Y").lower() == "y"
    else:
        project.ui.echo("Creating a mojoproject.toml for Muse...", style="primary")

    do_init(project=project, use_pyproject=use_pyproject)
    project.ui.echo("Project is initialized successfully", style="primary")


@main.command()
def init():
    project = Project(root_path=".")
    handle_init(project)


if __name__ == "__main__":
    main()
