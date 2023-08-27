from mojo_muse.models.candidates import Candidate
from mojo_muse.models.requirements import parse_requirement


class TestCandidate:
    r = parse_requirement("foo")
    c = Candidate(req=r)

    def test_init(self):
        assert self.c.__slots__ == (
            "req",
            "name",
            "version",
            "link",
            "summary",
            "hashes",
            "_prepared",
            "_requires_mojo",  # TODO: move to subclass
            "_requires_python",
            "_preferred",
        )

        assert hasattr(self.c, "req")
        assert hasattr(self.c, "name")
        assert hasattr(self.c, "version")
        assert hasattr(self.c, "link")

        assert hasattr(self.c, "identify")
        assert hasattr(self.c, "prepared")
