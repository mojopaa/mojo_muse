from mojo_muse.models.vcs.base import (
    HiddenText,
    VcsSupport,
    VersionControl,
    vcs_support,
)


class TestHiddenText:
    ht = HiddenText(secret="secret", redacted="redacted")

    def test___init__(self):
        assert self.ht.secret == "secret"
        assert self.ht.redacted == "redacted"

        assert str(self.ht) == self.ht.redacted


def test_vcs_support():
    assert hasattr(vcs_support, "get_backend")
    assert hasattr(vcs_support, "unregister_all")
    assert hasattr(vcs_support, "register")
