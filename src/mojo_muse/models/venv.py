from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from ..utils import get_venv_like_prefix

IS_WIN = sys.platform == "win32"
BIN_DIR = "Scripts" if IS_WIN else "bin"


def get_venv_python(venv: Path) -> Path:
    """Get the interpreter path inside the given venv."""
    suffix = ".exe" if IS_WIN else ""
    result = venv / BIN_DIR / f"python{suffix}"
    if IS_WIN and not result.exists():
        result = venv / "bin" / f"python{suffix}"  # for mingw64/msys2
        if result.exists():
            return result
        else:
            return venv / "python.exe"  # for conda
    return result


def is_conda_venv(root: Path) -> bool:
    return (root / "conda-meta").exists()


@dataclass(frozen=True)
class VirtualEnv:
    root: Path
    is_conda: bool
    interpreter: Path

    @classmethod
    def get(cls, root: Path) -> VirtualEnv | None:
        path = get_venv_python(root)
        if not path.exists():
            return None
        return cls(root, is_conda_venv(root), path)

    @classmethod
    def from_interpreter(cls, interpreter: Path) -> VirtualEnv | None:
        root, is_conda = get_venv_like_prefix(interpreter)
        if root is not None:
            return cls(root, is_conda, interpreter)
        return None

    def env_vars(self) -> dict[str, str]:
        key = "CONDA_PREFIX" if self.is_conda else "VIRTUAL_ENV"
        return {key: str(self.root)}


def get_in_project_venv(root: Path) -> VirtualEnv | None:
    """Get the python interpreter path of venv-in-project"""
    for possible_dir in (".venv", "venv", "env"):
        venv = VirtualEnv.get(root / possible_dir)
        if venv is not None:
            return venv
    return None
