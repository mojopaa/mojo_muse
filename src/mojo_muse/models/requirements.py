import functools
import posixpath
import re
import secrets
import warnings
from abc import ABC, abstractmethod
from pathlib import Path

from mups import normalize_name
from packaging.markers import Marker
from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import SpecifierSet

from ..exceptions import ExtrasWarning, RequirementError
from ..utils import comparable_version, path_to_url
from .markers import split_marker_extras
from .specifiers import get_specifier

VCS_SCHEMA = ("git", "hg", "svn", "bzr")
_vcs_req_re = re.compile(
    rf"(?P<url>(?P<vcs>{'|'.join(VCS_SCHEMA)})\+[^\s;]+)(?P<marker>[\t ]*;[^\n]+)?",
    flags=re.IGNORECASE,
)
_file_req_re = re.compile(
    r"(?:(?P<url>\S+://[^\s\[\];]+)|"
    r"(?P<path>(?:[^\s;\[\]]|\\ )*"
    r"|'(?:[^']|\\')*'"
    r"|\"(?:[^\"]|\\\")*\"))"
    r"(?P<extras>\[[^\[\]]+\])?(?P<marker>[\t ]*;[^\n]+)?"
)


def strip_extras(line: str) -> tuple[str, tuple[str, ...] | None]:
    match = re.match(r"^(.+?)(?:\[([^\]]+)\])?$", line)
    assert match is not None
    name, extras_str = match.groups()
    extras = tuple({e.strip() for e in extras_str.split(",")}) if extras_str else None
    return name, extras


class BaseMuseRequirement(ABC):
    name: str | None = None
    marker: Marker | str | None = None
    extras: list[str] | None = None
    specifier: SpecifierSet | None = None
    # editable: bool = False  # Not needed for now.
    prerelease: bool = False

    def __init__(
        self,
        name: str,
        marker: Marker | None = None,
        extras: list[str] | None = None,
        specifier: SpecifierSet | None = None,
        prerelease: bool = False,
    ) -> None:
        self.name = name
        if marker:
            if isinstance(marker, Marker):
                self.marker = marker
            elif isinstance(marker, str):
                self.marker = Marker(marker)
            elif marker is None:
                pass
            else:
                raise TypeError("marker must be a string or a Marker.")
        if extras:
            if isinstance(extras, list):
                self.extras = extras
            elif isinstance(extras, str):
                self.extras = list(e.strip() for e in extras[1:-1].split(","))
            elif extras is None:
                pass
            else:
                raise TypeError("extras must be a list or a string.")
        if specifier:
            if isinstance(specifier, SpecifierSet):
                self.specifier = specifier
            elif isinstance(specifier, str):
                self.specifier = get_specifier(specifier)
            elif specifier is None:
                pass
            else:
                raise TypeError("specifier must be a string or a SpecifierSet.")

        assert isinstance(prerelease, bool)
        self.prerelease = prerelease

    @abstractmethod
    def as_line(self) -> str:
        pass

    @property
    def is_named(self) -> bool:
        return isinstance(self, BaseNamedMuseRequirement)

    @property
    def is_vcs(self) -> bool:
        return isinstance(self, BaseVcsMuseRequirement)

    @property
    def is_file_or_url(self) -> bool:
        return isinstance(self, BaseFileMuseRequirement)

    @property
    def is_pinned(self) -> bool:
        if not self.specifier:
            return False

        if len(self.specifier) != 1:
            return False

        sp = next(iter(self.specifier))
        return sp.operator == "===" or sp.operator == "==" and "*" not in sp.version

    @property
    def project_name(self) -> str | None:
        return normalize_name(self.name, lowercase=False) if self.name else None

    @property
    def key(self) -> str | None:
        return self.project_name.lower() if self.project_name else None

    def identify(self) -> str:
        if not self.key:
            return _get_random_key(self)
        extras = "[{}]".format(",".join(sorted(self.extras))) if self.extras else ""
        return self.key + extras

    def _format_marker(self) -> str:
        if self.marker:
            return f"; {self.marker!s}"
        return ""

    def as_pinned_version(self, other_version: str | None):
        """Return a new requirement with the given pinned version."""
        if self.is_pinned or not other_version:
            return self
        normalized = comparable_version(other_version)
        return self.__class__(
            name=self.name,
            marker=self.marker,
            extras=self.extras,
            prerelease=self.prerelease,
            specifier=get_specifier(f"=={normalized}"),
        )

    def matches(self, line: str) -> bool:
        """Return whether the passed in PEP 508 string
        is the same requirement as this one.
        """
        if line.strip().startswith("-e "):
            req = parse_requirement(line.split("-e ", 1)[-1], True)
        else:
            req = parse_requirement(line, False)
        return self.key == req.key


class BaseNamedMuseRequirement(BaseMuseRequirement):
    pass


class BaseVcsMuseRequirement(BaseMuseRequirement):
    pass


class BaseFileMuseRequirement(BaseMuseRequirement):
    url: str = ""
    path: Path | None = None
    subdirectory: str | None = None

    def __init__(
        self,
        name: str,
        marker: Marker | None = None,
        extras: list[str] | None = None,
        specifier: SpecifierSet | None = None,
        prerelease: bool = False,
        url: str = "",
        path: Path | None = None,
        subdirectory: str | None = None,
    ) -> None:
        super().__init__(
            name=name,
            marker=marker,
            extras=extras,
            specifier=specifier,
            prerelease=prerelease,
        )
        self.url = url
        self.path = Path(path)
        self.subdirectory = subdirectory

    @property
    def is_local(self) -> bool:
        return self.path and self.path.exists() or False

    @property
    def is_local_dir(self) -> bool:
        return self.is_local and Path(self.path).is_dir()

    @property
    def str_path(self) -> str | None:
        if not self.path:
            return None
        if self.path.is_absolute():
            try:
                result = self.path.relative_to(Path.cwd()).as_posix()
            except ValueError:
                return self.path.as_posix()
        else:
            result = self.path.as_posix()
        result = posixpath.normpath(result)
        if not result.startswith(("./", "../")):
            result = "./" + result
        if result.startswith("./../"):
            result = result[2:]
        return result


#### ABC End, Concrete Classes Start #################################


class MuseRequirement(BaseMuseRequirement):
    @classmethod
    def from_requirement(cls, req: Requirement) -> BaseMuseRequirement:
        kwargs = {
            "name": req.name,
            "extras": req.extras,
            "specifier": req.specifier,
            "marker": req.marker,  # Do not use Marker() constructer for now.
        }

        if getattr(req, "url", None):
            # TODO: VcsMuseRequirement or FileMuseRequirement
            # link = Link(cast(str, req.url))
            # klass = VcsRequirement if link.is_vcs else FileRequirement
            # return klass(url=req.url, **kwargs)
            pass
        else:
            return NamedMuseRequirement(**kwargs)  # type: ignore[arg-type]

    def as_line(self) -> str:
        extras = f"[{','.join(sorted(self.extras))}]" if self.extras else ""
        return (
            f"{self.project_name}{extras}{self.specifier or ''}{self._format_marker()}"
        )


class NamedMuseRequirement(BaseNamedMuseRequirement):
    def as_line(self) -> str:
        extras = f"[{','.join(sorted(self.extras))}]" if self.extras else ""
        return (
            f"{self.project_name}{extras}{self.specifier or ''}{self._format_marker()}"
        )


class VcsMuseRequirement(BaseVcsMuseRequirement):
    pass


class FileMuseRequirement(BaseFileMuseRequirement):
    pass


@functools.lru_cache(maxsize=None)
def _get_random_key(req: Requirement | MuseRequirement) -> str:
    return f":empty:{secrets.token_urlsafe(8)}"


def parse_requirement(line: str, editable: bool = False) -> BaseMuseRequirement:
    # TODO: WIP
    m = _vcs_req_re.match(line)
    r: BaseMuseRequirement
    if m is not None:
        # TODO: VcsMuseRequirement
        # r = VcsMuseRequirement(**m.groupdict())
        pass
    else:
        # Special handling for hatch local references:
        # https://hatch.pypa.io/latest/config/dependency/#local
        # We replace the {root.uri} temporarily with a dummy URL header
        # to make it pass through the packaging.requirement parser
        # and then revert it.
        root_url = path_to_url(Path().as_posix())
        replaced = "{root:uri}" in line
        if replaced:
            line = line.replace("{root:uri}", root_url)
        try:
            pkg_req = Requirement(line)
        except InvalidRequirement as e:
            m = _file_req_re.match(line)
            if m is None:
                raise RequirementError(str(e)) from None
            # TODO: FileMuseRequirement
            # args = m.groupdict()
            # if (
            #     not line.startswith(".")
            #     and not args["url"]
            #     and args["path"]
            #     and not os.path.exists(args["path"])
            # ):
            #     raise RequirementError(str(e)) from None
            # r = FileRequirement.create(**args)
        else:
            r = MuseRequirement.from_requirement(pkg_req)
        if replaced:
            # assert isinstance(r, FileRequirement)
            # r.url = r.url.replace(root_url, "{root:uri}")
            # r.path = Path(get_relative_path(r.url) or "")
            pass

    if editable:
        # if r.is_vcs or r.is_file_or_url and r.is_local_dir:  # type: ignore[attr-defined]
        #     assert isinstance(r, FileRequirement)
        #     r.editable = True
        # else:
        #     raise RequirementError(
        #         "Editable requirement is only supported for VCS link or local directory."
        #     )
        pass
    return r


def filter_requirements_with_extras(
    project_name: str,
    requirement_lines: list[str],
    extras: list[str],
    include_default: bool = False,
) -> list[str]:
    """Filter the requirements with extras.
    If extras are given, return those with matching extra markers.
    Otherwise, return those without extra markers.
    """
    extras = [normalize_name(e) for e in extras]
    result: list[str] = []
    extras_in_meta: set[str] = set()
    for req in requirement_lines:
        _r = parse_requirement(req)
        if _r.marker:
            req_extras, rest = split_marker_extras(str(_r.marker))
            if req_extras:
                extras_in_meta.update(req_extras)
                _r.marker = Marker(rest) if rest else None
        else:
            req_extras = set()
        if (
            req_extras
            and not req_extras.isdisjoint(extras)
            or not req_extras
            and (include_default or not extras)
        ):
            result.append(_r.as_line())

    extras_not_found = [e for e in extras if e not in extras_in_meta]
    if extras_not_found:
        warnings.warn(ExtrasWarning(project_name, extras_not_found), stacklevel=2)

    return result
