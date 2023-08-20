import pytest

from mojo_muse.models.config import Config
from mojo_muse.utils import DEFAULT_CONFIG_FILENAME, find_project_root


def test_project_config_items():
    root = find_project_root()
    config = Config(root / DEFAULT_CONFIG_FILENAME, is_global=True)

    for item in ("python.use_pyenv", "pypi.url", "cache_dir"):
        assert item in config


def test_project_config_set_invalid_key():
    root = find_project_root()
    config = Config(root / DEFAULT_CONFIG_FILENAME)

    with pytest.raises(KeyError):
        config["foo"] = "bar"
