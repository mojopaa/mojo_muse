from __future__ import annotations

import contextlib
import hashlib
import os
import re
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping, cast

import platformdirs
import tomlkit
from tomlkit.items import Array

from ..termui import UI, ui
from .lockfile import Lockfile


class Project:
    """Core project class.

    Args:
        core: The core instance.
        root_path: The root path of the project.
        is_global: Whether the project is global.
        global_config: The path to the global config file.
    """

    MOJOPROJECT_FILENAME = "mojoproject.toml"
    LOCKFILE_FILENAME = "muse.lock"
    DEPENDENCIES_RE = re.compile(r"(?:(.+?)-)?dependencies")

    def __init__(
        self,
        root_path: str | Path | None,
        ui: UI = ui,
        is_global: bool = False,
        global_config: str | Path | None = None,
    ) -> None:
        self.ui = ui
        self._lockfile: Lockfile | None = None
        self._environment: BaseEnvironment | None = None
        self._python: PythonInfo | None = None
