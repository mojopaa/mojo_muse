import warnings

from .models.link import Link


class MuseException(Exception):
    pass


class InstallationError(MuseException):
    pass


class UninstallError(MuseException):
    pass


class MuseUsageError(MuseException):
    pass


class ProjectError(MuseUsageError):
    pass


class NoConfigError(MuseUsageError, KeyError):
    def __str__(self) -> str:
        return f"Not such config key: {self.args[0]!r}"


class InvalidPyVersion(MuseUsageError, ValueError):
    pass


class RequirementError(MuseException):
    pass


class CandidateInfoNotFound(MuseException):
    pass


class CandidateNotFound(MuseException):
    pass


class NoPythonVersion(MuseUsageError):
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


class BuildError(MuseException, RuntimeError):
    pass


# unearth


class HashMismatchError(UnpackError):
    def __init__(
        self, link: Link, expected: dict[str, list[str]], actual: dict[str, str]
    ) -> None:
        self.link = link
        self.expected = expected
        self.actual = actual

    def format_hash_item(self, name: str) -> str:
        expected = self.expected[name]
        actual = self.actual[name]
        expected_prefix = f"Expected({name}): "
        actual_prefix = f"  Actual({name}): "
        sep = "\n" + " " * len(expected_prefix)
        return f"{expected_prefix}{sep.join(expected)}\n{actual_prefix}{actual}"

    def __str__(self) -> str:
        return f"Hash mismatch for {self.link.redacted}:\n" + "\n".join(
            self.format_hash_item(name) for name in sorted(self.expected)
        )


class LinkCollectError(Exception):
    pass
