import contextlib
import os
from typing import BinaryIO, cast

from cachecontrol.cache import SeparateBodyBaseCache
from cachecontrol.caches import FileCache

from ..utils import atomic_open_for_write


class SafeFileCache(SeparateBodyBaseCache):
    """
    A file based cache which is safe to use even when the target directory may
    not be accessible or writable.
    """

    def __init__(self, directory: str) -> None:
        super().__init__()
        self.directory = directory

    def _get_cache_path(self, name: str) -> str:
        # From cachecontrol.caches.file_cache.FileCache._fn, brought into our
        # class for backwards-compatibility and to avoid using a non-public
        # method.
        hashed = FileCache.encode(name)
        parts = [*list(hashed[:5]), hashed]
        return os.path.join(self.directory, *parts)

    def get(self, key: str) -> bytes | None:
        path = self._get_cache_path(key)
        with contextlib.suppress(OSError):
            with open(path, "rb") as f:
                return f.read()

        return None

    def get_body(self, key: str) -> BinaryIO | None:
        path = self._get_cache_path(key)
        with contextlib.suppress(OSError):
            return cast(BinaryIO, open(f"{path}.body", "rb"))

        return None

    def set(self, key: str, value: bytes, expires: int | None = None) -> None:
        path = self._get_cache_path(key)
        with contextlib.suppress(OSError):
            with atomic_open_for_write(path, mode="wb") as f:
                cast(BinaryIO, f).write(value)

    def set_body(self, key: str, body: bytes) -> None:
        if body is None:
            return

        path = self._get_cache_path(key)
        with contextlib.suppress(OSError):
            with atomic_open_for_write(f"{path}.body", mode="wb") as f:
                cast(BinaryIO, f).write(body)

    def delete(self, key: str) -> None:
        path = self._get_cache_path(key)
        with contextlib.suppress(OSError):
            os.remove(path)
