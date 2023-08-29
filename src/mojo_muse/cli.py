from typing import cast

import click

from .core import do_init
from .project import Project
from .termui import ask


@click.group()
def main():  # subcommand uses main as decorator. main.command().
    pass


def handle_init(project: Project, is_interactive: bool = True):
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

    do_init(project=project, use_pyproject=use_pyproject)
    project.ui.echo("Project is initialized successfully", style="primary")


@main.command()
def init():
    project = Project(root_path=".")
    handle_init(project)


if __name__ == "__main__":
    main()
