import contextlib
import functools
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import IO, Iterator
from urllib import parse
from urllib.request import pathname2url, url2pathname

WINDOWS = sys.platform == "win32"


@contextlib.contextmanager
def cd(path: str | Path) -> Iterator:
    """Can use like this: with cd(path): ..."""
    _old_cwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(_old_cwd)


def find_project_root(cwd: str = ".", max_depth: int = 10) -> str | None:
    """Recursively find a `pyproject.toml` at given path or current working directory.
    If none if found, go to the parent directory, at most `max_depth` levels will be
    looked for.
    """
    original_path = Path(cwd).absolute()
    path = original_path
    for _ in range(max_depth):
        if path.joinpath("pyproject.toml").exists():
            # return path.as_posix()
            return path
        if path.parent == path:
            # Root path is reached
            break
        path = path.parent
    return None


def path_to_url(path: str) -> str:
    """
    Convert a path to a file: URL.  The path will be made absolute and have
    quoted path parts.
    """
    path = os.path.normpath(os.path.abspath(path))
    url = parse.urljoin("file:", pathname2url(path))
    return url


def url_to_path(url: str) -> str:
    """
    Convert a file: URL to a path.
    """
    assert url.startswith(
        "file:"
    ), f"You can only turn file: urls into filenames (not {url!r})"

    _, netloc, path, _, _ = parse.urlsplit(url)

    if not netloc or netloc == "localhost":
        # According to RFC 8089, same as empty authority.
        netloc = ""
    elif WINDOWS:
        # If we have a UNC path, prepend UNC share notation.
        netloc = "\\\\" + netloc
    else:
        raise ValueError(
            f"non-local file URIs are not supported on this platform: {url!r}"
        )

    path = url2pathname(netloc + path)

    # On Windows, urlsplit parses the path as something like "/C:/Users/foo".
    # This creates issues for path-related functions like io.open(), so we try
    # to detect and strip the leading slash.
    if (
        WINDOWS
        and not netloc  # Not UNC.
        and len(path) >= 3
        and path[0] == "/"  # Leading slash to strip.
        and path[1].isalpha()  # Drive letter.
        and path[2:4] in (":", ":/")  # Colon + end of string, or colon + absolute path.
    ):
        path = path[1:]

    return path


def add_ssh_scheme_to_git_uri(uri: str) -> str:
    """Cleans VCS uris from pip format"""
    # Add scheme for parsing purposes, this is also what pip does
    if "://" not in uri:
        uri = "ssh://" + uri
        parsed = parse.urlparse(uri)
        if ":" in parsed.netloc:
            netloc, _, path_start = parsed.netloc.rpartition(":")
            path = "/{0}{1}".format(path_start, parsed.path)
            uri = parse.urlunparse(parsed._replace(netloc=netloc, path=path))
    return uri


def parse_query(query: str) -> dict[str, str]:
    """Parse the query string of a url."""
    return {k: v[0] for k, v in parse.parse_qs(query).items()}


def split_auth_from_netloc(netloc: str) -> tuple[tuple[str, str | None] | None, str]:
    auth, has_auth, host = netloc.rpartition("@")
    if not has_auth:
        return None, host
    user, has_pass, password = auth.partition(":")
    return (parse.unquote(user), parse.unquote(password) if has_pass else None), host


@functools.lru_cache()
def split_auth_from_url(url: str) -> tuple[tuple[str, str | None] | None, str]:
    """Return a tuple of ((username, password), url_without_auth)"""
    parsed = parse.urlparse(url)
    auth, netloc = split_auth_from_netloc(parsed.netloc)
    if auth is None:
        return None, url
    return auth, parse.urlunparse(parsed._replace(netloc=netloc))


def build_url_from_netloc(netloc: str, scheme: str = "https") -> str:
    """
    Build a full URL from a netloc.
    """
    if netloc.count(":") >= 2 and "@" not in netloc and "[" not in netloc:
        # It must be a bare IPv6 address, so wrap it with brackets.
        netloc = f"[{netloc}]"
    return f"{scheme}://{netloc}"


def parse_netloc(netloc: str) -> tuple[str, int | None]:
    """
    Return the host-port pair from a netloc.
    """
    url = build_url_from_netloc(netloc)
    parsed = parse.urlparse(url)
    return parsed.hostname or "", parsed.port


@contextlib.contextmanager
def atomic_open_for_write(
    filename: str | Path, *, mode: str = "w", encoding: str = "utf-8"
) -> Iterator[IO]:
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    fd, name = tempfile.mkstemp(prefix="atomic-write-", dir=dirname)
    fp = open(fd, mode, encoding=encoding if "b" not in mode else None)
    try:
        yield fp
    except Exception:
        fp.close()
        raise
    else:
        fp.close()
        with contextlib.suppress(OSError):
            os.unlink(filename)
        # The tempfile is created with mode 600, we need to restore the default mode
        # with copyfile() instead of move().
        # See: https://github.com/pdm-project/pdm/issues/542
        shutil.copyfile(name, str(filename))
    finally:
        os.unlink(name)
