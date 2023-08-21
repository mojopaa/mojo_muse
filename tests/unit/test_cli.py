import os
import shutil
from pathlib import Path

from click.testing import CliRunner

from mojo_muse.cli import init, main
from mojo_muse.utils import cd


def test_main():
    runner = CliRunner()
    result = runner.invoke(main)
    assert result.exit_code == 0


def test_init():
    runner = CliRunner()
    test_path = Path(__file__).parent.parent / "t"  # tests/t

    os.makedirs(test_path, exist_ok=True)
    with cd(test_path):
        result = runner.invoke(init, input="\n\n\n\n\n\n\n\n\n")  # --yes
        assert result.exit_code == 0

    shutil.rmtree(test_path)
