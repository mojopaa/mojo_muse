from packaging.requirements import Requirement

from mojo_muse.models.requirements import (
    FileMuseRequirement,
    MuseRequirement,
    VcsMuseRequirement,
    parse_requirement,
)


def test_MuseRequirement():
    req = MuseRequirement.from_requirement(Requirement("foo"))
    assert req
    assert req.name == "foo"
    assert req.extras is None
    assert req.marker is None
    assert req.as_line() == "foo"
    assert req.key == "foo"
    assert req.identify() == "foo"


def test_pare_requirement():
    req = parse_requirement("foo")
    assert req
    assert req.name == "foo"
    assert req.extras is None
    assert req.marker is None
    assert req.as_line() == "foo"
