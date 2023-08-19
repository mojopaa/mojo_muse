import pytest

from mojo_muse import (
    BuildError,
    CandidateInfoNotFound,
    CandidateNotFound,
    ExtrasWarning,
    InvalidPyVersion,
    MuseException,
    MuseUsageError,
    NoConfigError,
    RequirementError,
    UnpackError,
    URLError,
    VCSBackendError,
    deprecation_warning,
)


def test_raise():
    with pytest.raises(BuildError):
        raise BuildError()

    with pytest.raises(CandidateInfoNotFound):
        raise CandidateInfoNotFound()

    with pytest.raises(CandidateNotFound):
        raise CandidateNotFound()

    with pytest.raises(ExtrasWarning):
        raise ExtrasWarning("", [])

    with pytest.raises(InvalidPyVersion):
        raise InvalidPyVersion()

    with pytest.raises(MuseException):
        raise MuseException()

    with pytest.raises(MuseUsageError):
        raise MuseUsageError()

    with pytest.raises(NoConfigError):
        raise NoConfigError()

    with pytest.raises(RequirementError):
        raise RequirementError()

    with pytest.raises(UnpackError):
        raise UnpackError()

    with pytest.raises(URLError):
        raise URLError()

    with pytest.raises(VCSBackendError):
        raise VCSBackendError()

    with pytest.warns(FutureWarning):
        deprecation_warning("")
