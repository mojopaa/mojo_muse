from mojo_muse.models.backends import MuseBackend, get_backend_by_spec


class TestMuseBackend:
    mb = MuseBackend(root=".")

    def test_init(self):
        assert hasattr(self.mb, "root")
        assert hasattr(self.mb, "expand_line")
        assert hasattr(self.mb, "relative_path_to_url")

    def test_build_system(self):
        assert hasattr(MuseBackend, "build_system")
        assert MuseBackend.build_system() == {
            "requires": ["muse-backend"],
            "build-backend": "muse.backend",
        }


def test_get_backend_by_spec():  # TODO
    pass
