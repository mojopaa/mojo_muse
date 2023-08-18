import atexit
import contextlib
import functools
import importlib.resources
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from re import Match
from typing import IO, Any, BinaryIO, Iterator
from urllib import parse
from urllib.request import pathname2url, url2pathname

from packaging.version import Version, _cmpkey

from ._types import FileHash

WINDOWS = sys.platform == "win32"

WHEEL_EXTENSION = ".whl"
BZ2_EXTENSIONS = (".tar.bz2", ".tbz")
XZ_EXTENSIONS = (
    ".tar.xz",
    ".txz",
    ".tlz",
    ".tar.lz",
    ".tar.lzma",
)
ZIP_EXTENSIONS = (".zip", WHEEL_EXTENSION)
TAR_EXTENSIONS = (".tar.gz", ".tgz", ".tar")
ARCHIVE_EXTENSIONS = ZIP_EXTENSIONS + BZ2_EXTENSIONS + TAR_EXTENSIONS + XZ_EXTENSIONS


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


def comparable_version(version: str) -> Version:
    """Normalize a version to make it valid in a specifier."""
    parsed = Version(version)
    if parsed.local is not None:
        # strip the local part
        parsed._version = parsed._version._replace(local=None)

        parsed._key = _cmpkey(
            parsed._version.epoch,
            parsed._version.release,
            parsed._version.pre,
            parsed._version.post,
            parsed._version.dev,
            parsed._version.local,
        )
    return parsed


def join_list_with(items: list[Any], sep: Any) -> list[Any]:
    new_items = []
    for item in items:
        new_items.extend([item, sep])
    return new_items[:-1]  # final sep is stripped off.


def url_without_fragments(url: str) -> str:
    return parse.urlunparse(parse.urlparse(url)._replace(fragment=""))


def expand_env_vars(credential: str, quote: bool = False) -> str:
    """A safe implementation of env var substitution.
    It only supports the following forms:

        ${ENV_VAR}

    Neither $ENV_VAR and %ENV_VAR is not supported.
    """

    def replace_func(match: Match) -> str:
        rv = os.getenv(match.group(1), match.group(0))
        return parse.quote(rv) if quote else rv

    return re.sub(r"\$\{(.+?)\}", replace_func, credential)


def expand_env_vars_in_auth(url: str) -> str:
    """In-place expand the auth in url"""
    scheme, netloc, path, params, query, fragment = parse.urlparse(url)
    if "@" in netloc:
        auth, rest = netloc.split("@", 1)
        auth = expand_env_vars(auth, True)
        netloc = "@".join([auth, rest])
    return parse.urlunparse((scheme, netloc, path, params, query, fragment))


@functools.lru_cache()
def compare_urls(left: str, right: str) -> bool:
    """
    Compare two urls, ignoring the ending slash.
    """
    return parse.unquote(left).rstrip("/") == parse.unquote(right).rstrip("/")


def get_rev_from_url(url: str) -> str:
    """Get the rev part from the VCS URL."""
    path = parse.urlparse(url).path
    if "@" in path:
        _, rev = path.rsplit("@", 1)
        return rev
    return ""


def is_url(url: str) -> bool:
    """Check if the given string is a URL"""
    return bool(parse.urlparse(url).scheme)


def get_relative_path(url: str) -> str | None:
    if url.startswith("file:///${PROJECT_ROOT}"):
        return parse.unquote(url[len("file:///${PROJECT_ROOT}/") :])
    if url.startswith("{root:uri}"):
        return parse.unquote(url[len("{root:uri}/") :])
    return None


def path_without_fragments(path: str) -> Path:
    """Remove egg fragment from path"""
    _egg_fragment_re = re.compile(r"(.*)[#&]egg=[^&]*")
    match = _egg_fragment_re.search(path)
    if not match:
        return Path(path)
    return Path(match.group(1))


def convert_hashes(files: list[FileHash]) -> dict[str, list[str]]:
    """Convert Pipfile.lock hash lines into InstallRequirement option format.

    The option format uses a str-list mapping. Keys are hash algorithms, and
    the list contains all values of that algorithm.
    """
    result: dict[str, list[str]] = {}
    for f in files:
        hash_value = f.get("hash", "")
        name, has_name, hash_value = hash_value.partition(":")
        if not has_name:
            name, hash_value = "sha256", name
        result.setdefault(name, []).append(hash_value)
    return result


def create_tracked_tempdir(
    suffix: str | None = None, prefix: str | None = None, dir: str | None = None
) -> str:
    name = tempfile.mkdtemp(suffix, prefix, dir)
    os.makedirs(name, mode=0o777, exist_ok=True)

    def clean_up() -> None:
        shutil.rmtree(name, ignore_errors=True)

    atexit.register(clean_up)
    return name


def splitext(path: str) -> tuple[str, str]:
    """Like os.path.splitext but also takes off the .tar part"""
    base, ext = os.path.splitext(path)
    if base.lower().endswith(".tar"):
        ext = base[-4:] + ext
        base = base[:-4]
    return base, ext


def resources_open_binary(package: str, resource: str) -> BinaryIO:
    return (importlib.resources.files(package) / resource).open("rb")
