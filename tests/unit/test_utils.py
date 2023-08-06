import os
from pathlib import Path

from mojo_muse.utils import cd, find_project_root


def test_cd():
    cwd = os.getcwd()
    with cd("../"):
        assert Path(os.getcwd()) == Path(cwd).parent
    assert os.getcwd() == cwd


def test_find_project_root():
    assert find_project_root() == Path(__file__).parent.parent.parent  # Any better idea?