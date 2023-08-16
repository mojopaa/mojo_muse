import warnings


class MuseException(Exception):
    pass


class MuseUsageError(MuseException):
    pass


class NoConfigError(MuseUsageError, KeyError):
    def __str__(self) -> str:
        return f"Not such config key: {self.args[0]!r}"


class RequirementError(MuseException):
    pass


class CandidateInfoNotFound(MuseException):
    pass


class CandidateNotFound(MuseException):
    pass


class ExtrasWarning(UserWarning):
    def __init__(self, project_name: str, extras: list[str]) -> None:
        super().__init__(f"Extras not found for {project_name}: [{','.join(extras)}]")
        self.extras = tuple(extras)


def deprecation_warning(
    message: str, stacklevel: int = 1, raise_since: str | None = None
) -> None:
    """Show a deprecation warning with the given message and raise an error
    after a specified version.
    """
    from packaging.version import Version

    from . import __version__

    if raise_since is not None:
        if Version(__version__) >= Version(raise_since):
            raise FutureWarning(message)
    warnings.warn(message, FutureWarning, stacklevel=stacklevel + 1)


# vcs module
class URLError(ValueError):
    pass


class VCSBackendError(URLError):
    pass


class UnpackError(RuntimeError):
    pass
