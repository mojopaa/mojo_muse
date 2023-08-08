import functools
import re
import secrets
from abc import ABC, abstractmethod

from packaging.requirements import Requirement

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


class BaseNamedMuseRequirement(BaseMuseRequirement):
    pass


class BaseVcsMuseRequirement(BaseMuseRequirement):
    pass


class BaseFileMuseRequirement(BaseMuseRequirement):
    pass


class MuseRequirement:
    pass


@functools.lru_cache(maxsize=None)
def _get_random_key(req: Requirement | MuseRequirement) -> str:
    return f":empty:{secrets.token_urlsafe(8)}"
