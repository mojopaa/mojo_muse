import os
from pathlib import Path

from packaging.version import Version

from mojo_muse.utils import (
    add_ssh_scheme_to_git_uri,
    build_url_from_netloc,
    cd,
    comparable_version,
    compare_urls,
    expand_env_vars_in_auth,
    find_project_root,
    get_rev_from_url,
    is_url,
    join_list_with,
    parse_netloc,
    parse_query,
    path_to_url,
    split_auth_from_netloc,
    split_auth_from_url,
    url_to_path,
    url_without_fragments,
)


def test_cd():
    cwd = os.getcwd()
    with cd("../"):
        assert Path(os.getcwd()) == Path(cwd).parent
    assert os.getcwd() == cwd


def test_find_project_root():
    assert (
        find_project_root() == Path(__file__).parent.parent.parent
    )  # Any better idea?


def test_path_to_url():
    path = r"C:\Users\drunk\projects\mojo_muse\src\mojo_muse\utils.py"
    assert (
        path_to_url(path)
        == "file:///C:/Users/drunk/projects/mojo_muse/src/mojo_muse/utils.py"
    )


def test_url_to_path():
    url = "file:///C:/Users/drunk/projects/mojo_muse/src/mojo_muse/utils.py"
    assert (
        url_to_path(url) == r"C:\Users\drunk\projects\mojo_muse\src\mojo_muse\utils.py"
    )


def test_add_ssh_scheme_to_git_uri():
    uri = "git@github.com:drunkwcodes/mups.git"
    assert add_ssh_scheme_to_git_uri(uri) == "ssh://git@github.com/drunkwcodes/mups.git"


def test_parse_query():
    url = "https://pypi.org/simple#egg=pip&subdirectory=src"
    assert parse_query(url) == {
        "https://pypi.org/simple#egg": "pip",
        "subdirectory": "src",
    }


def test_split_auth_from_netloc():
    netloc = "abc@pypi.org"
    assert split_auth_from_netloc(netloc) == (("abc", None), "pypi.org")

    netloc = "abc:pass@pypi.org"
    assert split_auth_from_netloc(netloc) == (("abc", "pass"), "pypi.org")


def test_split_auth_from_url():
    url = "https://abc@pypi.org/simple"
    assert split_auth_from_url(url) == (("abc", None), "https://pypi.org/simple")

    url = "https://abc:pass@pypi.org/simple"
    assert split_auth_from_url(url) == (("abc", "pass"), "https://pypi.org/simple")


def test_build_url_from_netloc():
    assert build_url_from_netloc("github.com") == "https://github.com"


def test_parse_netloc():
    assert parse_netloc("github.com") == ("github.com", None)


def test_comparable_version():
    assert comparable_version("1.2.3") == Version("1.2.3")
    assert comparable_version("1.2.3a1+local1") == Version("1.2.3a1")


def test_join_list_with():
    assert join_list_with(["a", "b", "c"], "and") == ["a", "and", "b", "and", "c"]


def test_url_without_fragments():
    assert (
        url_without_fragments("http://www.example.org/foo.html#bar")
        == "http://www.example.org/foo.html"
    )


def test_expand_env_vars_in_auth(monkeypatch):
    envs = {"HELLO": "world"}
    monkeypatch.setattr(os, "environ", envs)

    assert expand_env_vars_in_auth("http://${HELLO}@world") == "http://world@world"


def test_compare_urls():
    assert compare_urls("https://github.com", "https://github.com")


def test_get_rev_from_url():
    assert (
        get_rev_from_url(
            "git+https://github.com/python/mypy@effd970ad1e8bb09fd2a18634339e1d043a83400#egg=mypy"
        )
        == "effd970ad1e8bb09fd2a18634339e1d043a83400"
    )


def test_is_url():
    assert is_url("http://github.com")
    assert is_url("asdf") is False
