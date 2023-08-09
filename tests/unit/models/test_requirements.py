from mojo_muse.models.requirements import (
    MuseRequirement,
    VcsMuseRequirement,
    FileMuseRequirement,
    parse_requirement,
)

from packaging.requirements import Requirement


def test_MuseRequirement():
    req = MuseRequirement.from_requirement(Requirement("foo"))
    assert req