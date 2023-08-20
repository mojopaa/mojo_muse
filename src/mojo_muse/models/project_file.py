import hashlib
import json
from typing import Mapping

from tomlkit import TOMLDocument, items

from ..exceptions import deprecation_warning
from .toml_file import TOMLBase


def _remove_empty_tables(doc: dict) -> None:
    for k, v in list(doc.items()):
        if isinstance(v, dict):
            _remove_empty_tables(v)
            if not v:
                del doc[k]


class MojoProjectFile(TOMLBase):
    """The data object representing th mojoproject.toml file"""

    def read(self) -> TOMLDocument:
        data = super().read()
        return data

    def write(self, show_message: bool = True) -> None:
        """Write the TOMLDocument to the file."""
        _remove_empty_tables(self._data)
        super().write()
        if show_message:
            self.ui.echo("Changes are written to [success]mojoproject.toml[/].")

    @property
    def is_valid(self) -> bool:
        return bool(self._data.get("project"))

    @property
    def metadata(self) -> items.Table:
        return self._data.setdefault("project", {})

    @property
    def settings(self) -> items.Table:
        return self._data.setdefault("tool", {}).setdefault("muse", {})

    @property
    def build_system(self) -> dict:
        return self._data.get("build-system", {})

    @property
    def resolution_overrides(self) -> Mapping[str, str]:
        """A compatible getter method for the resolution overrides
        in the mojoproject.toml file.
        """
        settings = self.settings
        if "overrides" in settings:
            deprecation_warning(
                "The 'tool.muse.overrides' table has been renamed to "
                "'tool.muse.resolution.overrides', please update the "
                "setting accordingly."
            )
            return settings["overrides"]
        return settings.get("resolution", {}).get("overrides", {})

    def content_hash(self, algo: str = "sha256") -> str:
        """Generate a hash of the sensible content of the mojoproject.toml file.
        When the hash changes, it means the project needs to be relocked.
        """
        dump_data = {
            "sources": self.settings.get("source", []),
            "dependencies": self.metadata.get("dependencies", []),
            "dev-dependencies": self.settings.get("dev-dependencies", {}),
            "optional-dependencies": self.metadata.get("optional-dependencies", {}),
            "requires-mojo": self.metadata.get("requires-mojo", ""),
            "overrides": self.resolution_overrides,
        }
        mojoproject_content = json.dumps(dump_data, sort_keys=True)
        hasher = hashlib.new(algo)
        hasher.update(mojoproject_content.encode("utf-8"))
        return hasher.hexdigest()

    @property
    def plugins(self) -> list[str]:
        return self.settings.get("plugins", [])


class PyProjectFile(TOMLBase):
    """The data object representing th pyproject.toml file"""

    def read(self) -> TOMLDocument:
        data = super().read()
        return data

    def write(self, show_message: bool = True) -> None:
        """Write the TOMLDocument to the file."""
        _remove_empty_tables(self._data)
        super().write()
        if show_message:
            self.ui.echo("Changes are written to [success]pyproject.toml[/].")

    @property
    def is_valid(self) -> bool:
        return bool(self._data.get("project"))

    @property
    def metadata(self) -> items.Table:
        return self._data.setdefault("project", {})

    @property
    def settings(self) -> items.Table:
        return self._data.setdefault("tool", {}).setdefault("muse", {})

    @property
    def build_system(self) -> dict:
        return self._data.get("build-system", {})

    @property
    def resolution_overrides(self) -> Mapping[str, str]:
        """A compatible getter method for the resolution overrides
        in the pyproject.toml file.
        """
        settings = self.settings
        if "overrides" in settings:
            deprecation_warning(
                "The 'tool.muse.overrides' table has been renamed to "
                "'tool.muse.resolution.overrides', please update the "
                "setting accordingly."
            )
            return settings["overrides"]
        return settings.get("resolution", {}).get("overrides", {})

    def content_hash(self, algo: str = "sha256") -> str:
        """Generate a hash of the sensible content of the pyproject.toml file.
        When the hash changes, it means the project needs to be relocked.
        """
        dump_data = {
            "sources": self.settings.get("source", []),
            "dependencies": self.metadata.get("dependencies", []),
            "dev-dependencies": self.settings.get("dev-dependencies", {}),
            "optional-dependencies": self.metadata.get("optional-dependencies", {}),
            "requires-python": self.metadata.get("requires-python", ""),
            "overrides": self.resolution_overrides,
        }
        pyproject_content = json.dumps(dump_data, sort_keys=True)
        hasher = hashlib.new(algo)
        hasher.update(pyproject_content.encode("utf-8"))
        return hasher.hexdigest()

    @property
    def plugins(self) -> list[str]:
        return self.settings.get("plugins", [])
