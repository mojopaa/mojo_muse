from mojo_muse.models.caches import (
    HashCache,
    JSONFileCache,
    ProjectCache,
    RingCache,
    SafeFileCache,
    WheelCache,
    get_ring_cache,
    get_wheel_cache,
)


class TestSafeFileCache:
    cache = SafeFileCache("test")

    def test_init(self):
        assert hasattr(self.cache, "directory")
        assert hasattr(self.cache, "get")
        assert hasattr(self.cache, "get_body")
        assert hasattr(self.cache, "set")
        assert hasattr(self.cache, "set_body")
        assert hasattr(self.cache, "delete")


class TestHasetCache:
    cache = HashCache("test")

    def test_init(self):
        assert hasattr(self.cache, "directory")
        assert hasattr(self.cache, "get_hash")
        assert hasattr(self.cache, "get")
        assert hasattr(self.cache, "set")


class TestWeelCache:
    cache = WheelCache("test")

    def test_init(self):
        assert hasattr(self.cache, "directory")
        assert hasattr(self.cache, "ephemeral_directory")
        assert hasattr(self.cache, "get_path_for_link")
        assert hasattr(self.cache, "get_ephemeral_path_for_link")
        assert hasattr(self.cache, "get")


class TestRingCache:
    cache = RingCache("test")

    def test_init(self):
        assert hasattr(self.cache, "directory")
        assert hasattr(self.cache, "ephemeral_directory")
        assert hasattr(self.cache, "get_path_for_link")
        assert hasattr(self.cache, "get_ephemeral_path_for_link")
        assert hasattr(self.cache, "get")


class TestJSONFileCache:
    cache = JSONFileCache("test")

    def test_init(self):
        assert hasattr(self.cache, "get")
        assert hasattr(self.cache, "set")
        assert hasattr(self.cache, "delete")
        assert hasattr(self.cache, "clear")


class TestProjectCache:
    cache = ProjectCache()

    def test_init(self):
        assert hasattr(self.cache, "root")
        assert hasattr(self.cache, "global_config")
        assert hasattr(self.cache, "project_config")
        assert hasattr(self.cache, "config")
        assert hasattr(self.cache, "cache_dir")
        assert hasattr(self.cache, "cache")
        assert hasattr(self.cache, "make_hash_cache")
        assert hasattr(self.cache, "make_ring_cache")


def test_get_ring_cache():  # TODO
    pass


def test_get_wheel_cache():  # TODO
    pass
