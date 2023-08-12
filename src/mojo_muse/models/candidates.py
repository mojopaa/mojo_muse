from .requirements import BaseMuseRequirement


class Candidate:
    """A concrete candidate that can be downloaded and installed.
    A candidate comes from the PyPI index of a package, or from the requirement itself
    (for file or VCS requirements). Each candidate has a name, version and several
    dependencies together with package metadata.
    """

    __slots__ = (
        "req",
        "name",
        "version",
        "link",
        "summary",
        "hashes",
        "_prepared",
        "_requires_mojo",
        "_preferred",
    )

    def __init__(
        self,
        req: BaseMuseRequirement,
        name: str | None = None,
        version: str | None = None,
        link: Link | None = None,
    ):
        self.req = req
        self.name = name or self.req.project_name

    def identify(self) -> str:
        return self.req.identify()
