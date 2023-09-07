from typing import cast

import click

from .core import do_init, do_padd
from .project import BaseEnvironment, Project, PythonEnvironment
from .project.filters import GroupSelection  # TODO: export to project
from .termui import ask


@click.group()
def main():  # subcommand uses main as decorator. main.command().
    pass


def handle_init(
    environment: BaseEnvironment,
    project: Project | None = None,
    is_interactive: bool = True,
):
    project = project or environment.project

    use_pyproject: bool = False
    if project.mojoproject.exists():
        project.ui.echo(
            "mojoproject.toml already exists, update it now.", style="primary"
        )
    elif project.pyproject.exists():
        # TODO: it writes requires_mojo to pyproject.toml
        question = (
            "pyproject.toml already exists, do you want to use pyproject.toml instead?"
        )
        use_pyproject: bool = ask(question, default="Y").lower() == "y"
    else:
        project.ui.echo("Creating a mojoproject.toml for Muse...", style="primary")

    do_init(environment=environment, use_pyproject=use_pyproject)
    project.ui.echo("Project is initialized successfully", style="primary")


@main.command()
def init():
    project = Project(root_path=".")
    environment = PythonEnvironment(project=project)  # TODO: refine this.
    handle_init(environment=environment, project=project)


@main.command()
@main.option("-d", "--dev", is_flag=True)
@main.option("-G", "--group", help="Specify the target dependency group to add into")
@main.option("-s", "--sync", help="Write pyproject.toml and sync the working set")
def padd(dev: bool, group: str | None = None):
    project = Project()
    environment = PythonEnvironment(project=project)
    if group:
        selection = GroupSelection(project=project, group=group, dev=dev)
    else:
        selection = GroupSelection(
            project=project, default=default, dev=dev, groups=groups
        )
    do_padd(
        environment=environment,
        selection=selection,
    )


if __name__ == "__main__":
    main()
