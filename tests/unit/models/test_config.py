import pytest

from mojo_muse.models.config import Config


def test_project_config_items():
    config = Config(is_global=True)

    for item in ("python.use_pyenv", "pypi.url", "cache_dir"):
        assert item in config


def test_project_config_set_invalid_key():
    config = Config()

    with pytest.raises(KeyError):
        config["foo"] = "bar"
