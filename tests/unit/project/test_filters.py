"""A inspiring test file. This contains technique about how to init a Project under a foler, and how to set up deps for it."""

from pathlib import Path
import shutil
import os

import pytest

from mojo_muse.project.filters import GroupSelection
from mojo_muse.project import Project
from mojo_muse.utils import cd


def setup_dependencies(project):  # TODO: project.pyproject?
    project.pyproject.metadata.update(
        {
            "dependencies": ["requests"],
            "optional-dependencies": {"web": ["flask"], "auth": ["passlib"]},
        }
    )
    project.pyproject.settings.update({"dev-dependencies": {"test": ["pytest"], "doc": ["mkdocs"]}})
    project.pyproject.write()  # TODO: Project(root_path=".") will write to mojo_muse's pyproject.toml? Why?
@pytest.mark.parametrize(
    "args,golden",
    [
        ({"default": True, "dev": None, "groups": ()}, ["default", "test", "doc"]),
        (
            {"default": True, "dev": None, "groups": [":all"]},
            ["default", "web", "auth", "test", "doc"],
        ),
        (
            {"default": True, "dev": True, "groups": ["web"]},
            ["default", "web", "test", "doc"],
        ),
        (
            {"default": True, "dev": None, "groups": ["web"]},
            ["default", "web", "test", "doc"],
        ),
        ({"default": True, "dev": None, "groups": ["test"]}, ["default", "test"]),
        (
            {"default": True, "dev": None, "groups": ["test", "web"]},
            ["default", "test", "web"],
        ),
        ({"default": True, "dev": False, "groups": ["web"]}, ["default", "web"]),
        ({"default": False, "dev": None, "groups": ()}, ["test", "doc"]),
    ],
)
def dependency_group_selection(args, golden):
# def test_dependency_group_selection(args, golden):  # TODO: test failed.
    test_dir = Path(__file__).parent.parent.parent / "g"  # test/g
    os.makedirs(test_dir, exist_ok=True)

    try:
        with cd(test_dir):
            project = Project(root_path=".")
            setup_dependencies(project)
            selection = GroupSelection(project, **args)
            assert sorted(golden) == sorted(selection)
    finally:
        shutil.rmtree(test_dir)
        # pass