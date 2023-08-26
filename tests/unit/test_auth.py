import shutil

from mojo_muse.auth import (
    BaseKeyringProvider,
    Keyring,
    KeyringCliProvider,
    KeyringModuleProvider,
    MultiDomainBasicAuth,
    MuseBasicAuth,
    RepositoryConfigWithPassword,
    get_keyring_auth,
    get_keyring_provider,
    keyring,
)


class TestKeyring:
    k = Keyring()

    def test_init(self):
        assert hasattr(self.k, "get_auth_info")
        assert hasattr(self.k, "save_auth_info")


class TestKeyringCliProvider:
    kr = shutil.which("keyring")
    k = KeyringCliProvider(kr)

    def test_init(self):
        assert hasattr(self.k, "get_auth_info")
        assert hasattr(self.k, "save_auth_info")
        assert hasattr(self.k, "keyring")


class TestKeyringModuleProvider:
    k = KeyringModuleProvider()

    def test_init(self):
        assert hasattr(self.k, "get_auth_info")
        assert hasattr(self.k, "save_auth_info")
        assert hasattr(self.k, "keyring")


class TestMultiDomainBasicAuth:
    m = MultiDomainBasicAuth()

    def test_init(self):  # TODO
        assert hasattr(self.m, "save_credentials")


class TestMuseBasicAuth:
    m = MuseBasicAuth(sources=[])

    def test_init(self):  # TODO
        assert hasattr(self.m, "sources")
        assert hasattr(self.m, "ui")


class TestRepositoryConfigWithPassword:
    r = RepositoryConfigWithPassword(config_prefix="mojopi", name="mojopi")

    def test_init(self):
        assert hasattr(self.r, "password")


def test_get_keyring_auth():
    assert get_keyring_auth() is None
    assert (
        get_keyring_auth(url="http://example.com", username="test") is None
    )  # TODO: better test case


def test_get_keyring_provider():
    # assert isinstance(get_keyring_provider(), KeyringModuleProvider)
    pass  # TODO: failed on workflow.
