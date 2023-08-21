import functools
import json
import os
import posixpath
import re
import secrets
import warnings
from abc import ABC, abstractmethod
from importlib.metadata import Distribution
from pathlib import Path
from typing import Any, cast
from urllib.parse import parse_qsl, unquote, urlparse, urlunparse

from mups import normalize_name, parse_ring_filename, parse_sdist_filename
from packaging.markers import Marker
from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import SpecifierSet

from ..exceptions import ExtrasWarning, RequirementError
from ..utils import (
    RequirementDict,
    add_ssh_scheme_to_git_uri,
    comparable_version,
    get_relative_path,
    path_to_url,
    path_without_fragments,
    url_to_path,
    url_without_fragments,
)
from .backends import BuildBackend
from .link import Link
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
    editable: bool = False  # Not needed for now.
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

    def _hash_key(self) -> tuple:
        return (
            self.key,
            frozenset(self.extras) if self.extras else None,
            str(self.marker) if self.marker else None,
        )

    def __hash__(self) -> int:
        return hash(self._hash_key())


class BaseNamedMuseRequirement(BaseMuseRequirement):
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
        self.path = Path(path)

        # if url:  # TODO: decide later.
        #     self.url = url
        self._parse_url()

        if self.is_local_dir:
            self._check_installable()
        self.subdirectory = subdirectory

    def _parse_url(self) -> None:
        if not self.url and self.path and self.path.is_absolute():
            self.url = path_to_url(self.path.as_posix())
        if not self.path:
            path = get_relative_path(self.url)
            if path is None:
                try:
                    self.path = path_without_fragments(url_to_path(self.url))
                except AssertionError:
                    pass
            else:
                self.path = path_without_fragments(path)
        if self.url:
            self._parse_name_from_url()

    def _check_installable(self) -> None:
        assert self.path
        if not (self.path.joinpath("mojoproject.toml").exists()):
            raise RequirementError(f"The local path '{self.path}' is not installable.")
        # TODO: Is this required? If so, move project_file.py & toml_file.py to models/.
        # result = Setup.from_directory(self.path.absolute())
        # if result.name:
        #     self.name = result.name

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

    def get_full_url(self) -> str:
        return url_without_fragments(self.url)

    def _hash_key(self) -> tuple:
        return (*super()._hash_key(), self.get_full_url(), self.editable)

    def guess_name(self) -> str | None:
        filename = os.path.basename(unquote(url_without_fragments(self.url))).rsplit(
            "@", 1
        )[0]
        if self.is_vcs:
            if self.vcs == "git":  # type: ignore[attr-defined]
                name = filename
                if name.endswith(".git"):
                    name = name[:-4]
                return name
            elif self.vcs == "hg":  # type: ignore[attr-defined]
                return filename
            else:  # svn and bzr
                name, in_branch, _ = filename.rpartition("/branches/")
                if not in_branch and name.endswith("/trunk"):
                    return name[:-6]
                return name
        elif filename.endswith(".ring"):
            return parse_ring_filename(filename)[0]
        else:
            try:
                return parse_sdist_filename(filename)[0]
            except ValueError:
                # match = _egg_info_re.match(filename)
                # # Filename is like `<name>-<version>.tar.gz`, where name will be
                # # extracted and version will be left to be determined from
                # # the metadata.
                # if match:
                #     return match.group(1)
                pass  # TODO: double check
        return None

    def _parse_name_from_url(self) -> None:
        parsed = urlparse(self.url)
        fragments = dict(parse_qsl(parsed.fragment))
        if "egg" in fragments:  # TODO: customize for mojopi
            egg_info = unquote(fragments["egg"])
            name, extras = strip_extras(egg_info)
            self.name = name
            if not self.extras:
                self.extras = extras
        if not self.name and not self.is_vcs:
            self.name = self.guess_name()

    def as_file_link(self) -> Link:
        url = self.get_full_url()
        # only subdirectory is useful in a file link
        if self.subdirectory:
            url += f"#subdirectory={self.subdirectory}"
        return Link(url)

    def relocate(self, backend: BuildBackend) -> None:
        """Change the project root to the given path"""  # TODO: reword? project root?
        if self.path is None or self.path.is_absolute():
            return
        # self.path is relative
        self.path = path_without_fragments(os.path.relpath(self.path, backend.root))
        path = self.path.as_posix()
        if path == ".":
            path = ""
        self.url = backend.relative_path_to_url(path)

    # @classmethod
    # def create(cls: type[T], **kwargs: Any) -> T:
    #     if kwargs.get("path"):
    #         kwargs["path"] = Path(kwargs["path"])
    #     return super().create(**kwargs)


class BaseVcsMuseRequirement(BaseFileMuseRequirement):
    vcs: str = ""
    ref: str | None = None
    revision: str | None = None

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
        vcs: str = "",
        ref: str | None = None,
        revision: str | None = None,
    ) -> None:
        super().__init__(
            name=name,
            marker=marker,
            extras=extras,
            specifier=specifier,
            prerelease=prerelease,
            url=url,
            path=path,
            subdirectory=subdirectory,
        )

        if not vcs:
            self.vcs = self.url.split("+", 1)[0]
        else:
            self.vcs = vcs

        if ref:
            self.ref = ref
        if revision:
            self.revision = revision

    def get_full_url(self) -> str:
        url = super().get_full_url()
        if self.revision and not self.editable:
            url += f"@{self.revision}"
        elif self.ref:
            url += f"@{self.ref}"
        return url

    def _parse_url(self) -> None:
        vcs, url_no_vcs = self.url.split("+", 1)
        if url_no_vcs.startswith("git@"):
            url_no_vcs = add_ssh_scheme_to_git_uri(url_no_vcs)
        if not self.name:
            self._parse_name_from_url()
        ref = self.ref
        parsed = urlparse(url_no_vcs)
        path = parsed.path
        fragments = dict(parse_qsl(parsed.fragment))
        if "subdirectory" in fragments:
            self.subdirectory = fragments["subdirectory"]
        if "@" in parsed.path:
            path, ref = parsed.path.split("@", 1)
        repo = urlunparse(parsed._replace(path=path, fragment=""))
        self.url = f"{vcs}+{repo}"
        self.repo, self.ref = repo, ref


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

    @classmethod
    def from_req_dict(cls, name: str, req_dict: RequirementDict) -> Requirement:
        if isinstance(req_dict, str):  # Version specifier only.
            return NamedMuseRequirement(name=name, specifier=get_specifier(req_dict))
        for vcs in VCS_SCHEMA:
            if vcs in req_dict:
                repo = cast(str, req_dict.pop(vcs, None))
                url = f"{vcs}+{repo}"
                return VcsMuseRequirement(name=name, vcs=vcs, url=url, **req_dict)
        if "path" in req_dict or "url" in req_dict:
            return FileMuseRequirement(name=name, **req_dict)
        return NamedMuseRequirement(name=name, **req_dict)

    @classmethod
    def from_dist(
        cls, dist: Distribution
    ) -> Requirement:  # TODO: change name if mojo uses distribution
        direct_url_json = dist.read_text("direct_url.json")
        if direct_url_json is not None:
            direct_url = json.loads(direct_url_json)
            data = {
                "name": dist.metadata["Name"],
                "url": direct_url.get("url"),
                "editable": direct_url.get("dir_info", {}).get("editable"),
                "subdirectory": direct_url.get("subdirectory"),
            }
            if "vcs_info" in direct_url:
                vcs_info = direct_url["vcs_info"]
                data.update(
                    url=f"{vcs_info['vcs']}+{direct_url['url']}",
                    ref=vcs_info.get("requested_revision"),
                    revision=vcs_info.get("commit_id"),
                )
                return VcsMuseRequirement(**data)  # TODO: check if valid
            return FileMuseRequirement(**data)  # TODO: check if valid
        return NamedMuseRequirement(
            name=dist.metadata["Name"], version=f"=={dist.version}"
        )  # TODO: check if valid

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


class FileMuseRequirement(BaseFileMuseRequirement):
    def as_line(self) -> str:
        project_name = f"{self.project_name}" if self.project_name else ""
        extras = (
            f"[{','.join(sorted(self.extras))}]"
            if self.extras and self.project_name
            else ""
        )
        marker = self._format_marker()
        if marker:
            marker = f" {marker}"
        url = self.get_full_url()
        fragments = []
        if self.subdirectory:
            fragments.append(f"subdirectory={self.subdirectory}")
        if self.editable:
            if project_name:
                fragments.insert(0, f"egg={project_name}{extras}")
            fragment_str = ("#" + "&".join(fragments)) if fragments else ""
            return f"-e {url}{fragment_str}{marker}"
        delimiter = " @ " if project_name else ""
        fragment_str = ("#" + "&".join(fragments)) if fragments else ""
        return f"{project_name}{extras}{delimiter}{url}{fragment_str}{marker}"


class VcsMuseRequirement(BaseVcsMuseRequirement):
    def as_line(self) -> str:
        project_name = f"{self.project_name}" if self.project_name else ""
        extras = (
            f"[{','.join(sorted(self.extras))}]"
            if self.extras and self.project_name
            else ""
        )
        marker = self._format_marker()
        if marker:
            marker = f" {marker}"
        url = self.get_full_url()
        fragments = []
        if self.subdirectory:
            fragments.append(f"subdirectory={self.subdirectory}")
        if self.editable:
            if project_name:
                fragments.insert(0, f"egg={project_name}{extras}")
            fragment_str = ("#" + "&".join(fragments)) if fragments else ""
            return f"-e {url}{fragment_str}{marker}"
        delimiter = " @ " if project_name else ""
        fragment_str = ("#" + "&".join(fragments)) if fragments else ""
        return f"{project_name}{extras}{delimiter}{url}{fragment_str}{marker}"


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
