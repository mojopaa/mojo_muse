from mojo_muse.project import Project


class TestProject:
    project = Project(root_path=".")

    def test_init(self):
        assert hasattr(self.project, "is_global")
