import contextlib
import hashlib
import os
from pathlib import Path
from typing import BinaryIO, Iterable, cast

from cachecontrol.cache import SeparateBodyBaseCache
from cachecontrol.caches import FileCache
from requests import HTTPError

from ..exceptions import MuseException
from ..session import Session
from ..termui import logger
from ..utils import atomic_open_for_write
from .link import Link


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


class HashCache:

    """Caches hashes of MojoPI artifacts so we do not need to re-download them.

    Hashes are only cached when the URL appears to contain a hash in it and the
    cache key includes the hash value returned from the server). This ought to
    avoid issues where the location on the server changes.
    """

    FAVORITE_HASH = "sha256"
    STRONG_HASHES = ("sha256", "sha384", "sha512")

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def _read_from_link(self, link: Link, session: Session) -> Iterable[bytes]:
        if link.is_file:
            with open(link.file_path, "rb") as f:
                yield from f
        else:
            with session.get(link.normalized, stream=True) as resp:
                try:
                    resp.raise_for_status()
                except HTTPError as e:
                    raise MuseException(
                        f"Failed to read from {link.redacted}: {e}"
                    ) from e
                yield from resp.iter_content(chunk_size=8192)

    def _get_file_hash(self, link: Link, session: Session) -> str:
        h = hashlib.new(self.FAVORITE_HASH)
        logger.debug("Downloading link %s for calculating hash", link.redacted)
        for chunk in self._read_from_link(link, session):
            h.update(chunk)
        return ":".join([h.name, h.hexdigest()])

    def _should_cache(self, link: Link) -> bool:
        # For now, we only disable caching for local files.
        # We may add more when we know better about it.
        return not link.is_file

    def get_hash(self, link: Link, session: Session) -> str:
        # If there is no link hash (i.e., md5, sha256, etc.), we don't want
        # to store it.
        hash_value = self.get(link.url_without_fragment)
        if not hash_value:
            if link.hashes and link.hashes.keys() & self.STRONG_HASHES:
                logger.debug("Using hash in link for %s", link.redacted)
                hash_name = next(k for k in self.STRONG_HASHES if k in link.hashes)
                hash_value = f"{hash_name}:{link.hashes[hash_name]}"
            elif link.hash and link.hash_name in self.STRONG_HASHES:
                logger.debug("Using hash in link for %s", link.redacted)
                hash_value = f"{link.hash_name}:{link.hash}"
            else:
                hash_value = self._get_file_hash(link, session)
            if self._should_cache(link):
                self.set(link.url_without_fragment, hash_value)
        return hash_value

    def _get_path_for_key(self, key: str) -> Path:
        hashed = hashlib.sha224(key.encode("utf-8")).hexdigest()
        parts = (hashed[:2], hashed[2:4], hashed[4:6], hashed[6:8], hashed[8:])
        return self.directory.joinpath(*parts)

    def get(self, url: str) -> str | None:
        path = self._get_path_for_key(url)
        with contextlib.suppress(OSError, UnicodeError):
            return path.read_text("utf-8").strip()
        return None

    def set(self, url: str, hash: str) -> None:
        path = self._get_path_for_key(url)
        with contextlib.suppress(OSError, UnicodeError):
            path.parent.mkdir(parents=True, exist_ok=True)
            with atomic_open_for_write(path, encoding="utf-8") as fp:
                fp.write(hash)
