from __future__ import annotations

import atexit
import contextlib
import functools
import importlib.resources
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from importlib.metadata import Distribution
from pathlib import Path
from re import Match
from typing import (
    IO,
    Any,
    BinaryIO,
    ContextManager,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    MutableMapping,
    NamedTuple,
    Protocol,
    Sequence,
    Tuple,
    TypedDict,
    TypeVar,
    Union,
)
from urllib import parse
from urllib.request import pathname2url, url2pathname

from packaging.specifiers import SpecifierSet
from packaging.version import Version, _cmpkey
from semver import Version as SemVer

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

DEFAULT_MOJOPROJECT_FILENAME = "mojoproject.toml"
DEFAULT_PYPROJECT_FILENAME = "pyproject.toml"


def is_archive_file(name: str) -> bool:
    """Return True if `name` is a considered as an archive file."""
    ext = splitext(name)[1].lower()
    return ext in ARCHIVE_EXTENSIONS


T = TypeVar("T", covariant=True)


class LazySequence(Sequence[T]):
    """A sequence that is lazily evaluated."""

    def __init__(self, data: Iterable[T]) -> None:
        self._inner = data

    def __iter__(self) -> Iterator[T]:
        self._inner, this = itertools.tee(self._inner)
        return this

    def __len__(self) -> int:
        i = 0
        for _ in self:
            i += 1
        return i

    def __bool__(self) -> bool:
        for _ in self:
            return True
        return False

    def __getitem__(self, index: int) -> T:  # type: ignore[override]
        if index < 0:
            raise IndexError("Negative indices are not supported")
        for i, item in enumerate(self):
            if i == index:
                return item
        raise IndexError("Index out of range")


@dataclass
class _RepositoryConfig:
    """Private dataclass to be subclassed"""

    config_prefix: str
    name: str

    url: str | None = None
    username: str | None = None
    _password: str | None = field(default=None, repr=False)
    verify_ssl: bool | None = None
    type: str | None = None
    ca_certs: str | None = None


class RepositoryConfig(_RepositoryConfig):
    """Removed auth.py deps. use auth.RepositoryConfigWithPassword instead if password is required"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def passive_update(
        self, other: RepositoryConfig | None = None, **kwargs: Any
    ) -> None:
        """An update method that prefers the existing value over the new one."""
        if other is not None:
            for k in other.__dataclass_fields__:
                v = getattr(other, k)
                if getattr(self, k) is None and v is not None:
                    setattr(self, k, v)
        for k, v in kwargs.items():
            if getattr(self, k) is None and v is not None:
                setattr(self, k, v)

    def __rich__(self) -> str:
        config_prefix = (
            f"{self.config_prefix}.{self.name}."
            if self.name
            else f"{self.config_prefix}."
        )
        lines: list[str] = []
        if self.url:
            lines.append(f"[primary]{config_prefix}url[/] = {self.url}")
        if self.username:
            lines.append(f"[primary]{config_prefix}username[/] = {self.username}")
        if self.password:
            lines.append(f"[primary]{config_prefix}password[/] = [i]<hidden>[/]")
        if self.verify_ssl is not None:
            lines.append(f"[primary]{config_prefix}verify_ssl[/] = {self.verify_ssl}")
        if self.type:
            lines.append(f"[primary]{config_prefix}type[/] = {self.type}")
        if self.ca_certs:
            lines.append(f"[primary]{config_prefix}ca_certs[/] = {self.ca_certs}")
        return "\n".join(lines)


RequirementDict = Union[str, Dict[str, Union[str, bool]]]
CandidateInfo = Tuple[List[str], str, str]


class Package(NamedTuple):
    name: str
    version: str
    summary: str


SearchResult = List[Package]


class Comparable(Protocol):
    def __lt__(self, __other: Any) -> bool:
        ...


SpinnerT = TypeVar("SpinnerT", bound="Spinner")


class Spinner(Protocol):
    def update(self, text: str) -> None:
        ...

    def __enter__(self: SpinnerT) -> SpinnerT:
        ...

    def __exit__(self, *args: Any) -> None:
        ...


class RichProtocol(Protocol):
    def __rich__(self) -> str:
        ...


class FileHash(TypedDict, total=False):
    url: str
    hash: str
    file: str


@contextlib.contextmanager
def cd(path: str | Path) -> Iterator:
    """Can use like this: with cd(path): ..."""
    _old_cwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(_old_cwd)


def find_project_root(cwd: str = ".", max_depth: int = 10, default: str = ".") -> str:
    """Recursively find a `pyproject.toml` at given path or current working directory.
    If none if found, go to the parent directory, at most `max_depth` levels will be
    looked for.

    If no `pyproject.toml` is found, return the given `default` path.
    """
    original_path = Path(cwd).absolute()
    path = original_path
    for _ in range(max_depth):
        # print(path)
        if (
            path.joinpath(DEFAULT_MOJOPROJECT_FILENAME).exists()
            or path.joinpath(DEFAULT_PYPROJECT_FILENAME).exists()
        ):
            # return path.as_posix()
            return path
        if path.parent == path:
            # Root path is reached
            break
        path = path.parent
    return default


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


def resources_path(package: str, resource: str) -> ContextManager[Path]:
    return importlib.resources.as_file(importlib.resources.files(package) / resource)


def is_subset(superset: SpecifierSet, subset: SpecifierSet) -> bool:
    # Rough impl.
    versions = []
    MAX_VER = Version("9999")
    MIN_VER = Version("0.0.1")
    for spec in subset:
        vstr = str(spec).split(spec.operator)[-1]
        try:
            version = SemVer.parse(vstr)
        except ValueError:  # semantic version is strict to have patch number.
            vstr += ".0"
            version = SemVer.parse(vstr)

        if spec.operator == ">":
            version = Version(str(version.bump_patch()))

        elif spec.operator == "<":
            version = SemVer(
                major=version.major, minor=version.minor - 1, patch=version.patch
            )
            version = Version(str(version))

        versions.append(Version(str(version)))

    if len(subset) == 1:
        for spec in subset:
            if ">" in spec.operator:
                versions.append(MAX_VER)
            elif "<" in spec.operator:
                versions.append(MIN_VER)
    # print([type(version) for version in versions])
    return all(version in superset for version in versions)


def display_path(path: Path) -> str:
    """Show the path relative to cwd if possible"""
    if not path.is_absolute():
        return str(path)
    try:
        relative = path.absolute().relative_to(Path.cwd())
    except ValueError:
        return str(path)
    else:
        return str(relative)


def get_trusted_hosts(sources: list[RepositoryConfig]) -> list[str]:
    """Parse the project sources and return the trusted hosts"""
    trusted_hosts = []
    for source in sources:
        assert source.url
        url = source.url
        netloc = parse.urlparse(url).netloc
        host = netloc.rsplit("@", 1)[-1]
        if host not in trusted_hosts and source.verify_ssl is False:
            trusted_hosts.append(host)
    return trusted_hosts


def is_path_relative_to(path: str | Path, other: str | Path) -> bool:
    try:
        Path(path).relative_to(other)
    except ValueError:
        return False
    return True


def get_venv_like_prefix(interpreter: str | Path) -> tuple[Path | None, bool]:
    """Check if the given interpreter path is from a virtualenv,
    and return two values: the root path and whether it's a conda env.
    """
    interpreter = Path(interpreter)
    prefix = interpreter.parent
    if prefix.joinpath("conda-meta").exists():
        return prefix, True

    prefix = prefix.parent
    if prefix.joinpath("pyvenv.cfg").exists():
        return prefix, False
    if prefix.joinpath("conda-meta").exists():
        return prefix, True

    virtual_env = os.getenv("VIRTUAL_ENV")
    if virtual_env and is_path_relative_to(interpreter, virtual_env):
        return Path(virtual_env), False
    virtual_env = os.getenv("CONDA_PREFIX")
    if virtual_env and is_path_relative_to(interpreter, virtual_env):
        return Path(virtual_env), True
    return None, False


def find_python_in_path(path: str | Path) -> Path | None:
    """Find a python interpreter from the given path, the input argument could be:

    - A valid path to the interpreter
    - A Python root directory that contains the interpreter
    """
    pathlib_path = Path(path).absolute()
    if pathlib_path.is_file():
        return pathlib_path

    if os.name == "nt":
        for root_dir in (pathlib_path, pathlib_path / "Scripts"):
            if root_dir.joinpath("python.exe").exists():
                return root_dir.joinpath("python.exe")
    else:
        executable_pattern = re.compile(r"python(?:\d(?:\.\d+m?)?)?$")

        for python in pathlib_path.joinpath("bin").glob("python*"):
            if executable_pattern.match(python.name):
                return python

    return None


def merge_dictionary(
    target: MutableMapping[Any, Any],
    input: Mapping[Any, Any],
    append_array: bool = True,
) -> None:
    """Merge the input dict with the target while preserving the existing values
    properly. This will update the target dictionary in place.
    List values will be extended, but only if the value is not already in the list.
    """
    for key, value in input.items():
        if key not in target:
            target[key] = value
        elif isinstance(value, dict):
            merge_dictionary(target[key], value, append_array=append_array)
        elif isinstance(value, list) and append_array:
            target[key].extend(x for x in value if x not in target[key])
            if hasattr(target[key], "multiline"):
                target[key].multiline(True)  # type: ignore[attr-defined]
        else:
            target[key] = value
    return target


def format_size(size: str) -> str:
    try:
        int_size = int(size)
    except (TypeError, ValueError):
        return "size unknown"
    if int_size > 1000 * 1000:
        return f"{int_size / 1000.0 / 1000:.1f} MB"
    elif int_size > 10 * 1000:
        return f"{int(int_size / 1000)} kB"
    elif int_size > 1000:
        return f"{int_size / 1000.0:.1f} kB"
    else:
        return f"{int(int_size)} bytes"


# TODO: make dist easier and test
def is_egg_link(dist: Distribution) -> bool:
    """Check if the distribution is an egg-link install"""
    return getattr(dist, "link_file", None) is not None


def is_editable(dist: Distribution) -> bool:
    """Check if the distribution is installed in editable mode"""
    if is_egg_link(dist):
        return True
    direct_url = dist.read_text("direct_url.json")
    if not direct_url:
        return False
    direct_url_data = json.loads(direct_url)
    return direct_url_data.get("dir_info", {}).get("editable", False)


@functools.lru_cache()
def fs_supports_symlink() -> bool:
    if not hasattr(os, "symlink"):
        return False
    if sys.platform == "win32":
        with tempfile.NamedTemporaryFile(prefix="TmP") as tmp_file:
            temp_dir = os.path.dirname(tmp_file.name)
            dest = os.path.join(temp_dir, "{}-{}".format(tmp_file.name, "b"))
            try:
                os.symlink(tmp_file.name, dest)
                return True
            except (OSError, NotImplementedError):
                return False
    else:
        return True
