import collections
import contextlib
import dataclasses
import hashlib
import json
import os
from functools import cached_property, lru_cache
from pathlib import Path
from typing import Any, BinaryIO, Generic, Iterable, Mapping, TypeVar, cast

import platformdirs
from cachecontrol.cache import SeparateBodyBaseCache
from cachecontrol.caches import FileCache
from mups import parse_ring_filename
from packaging.tags import Tag
from packaging.utils import canonicalize_name, parse_wheel_filename
from requests import HTTPError

from ..evaluator import TargetMojo, TargetPython
from ..exceptions import MuseException
from ..session import Session
from ..termui import logger
from ..utils import (
    DEFAULT_MOJOPROJECT_FILENAME,
    atomic_open_for_write,
    convert_hashes,
    create_tracked_tempdir,
    find_project_root,
    url_without_fragments,
)
from .config import DEFAULT_CONFIG_FILENAME, Config
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

    def __init__(self, directory: Path | str) -> None:
        self.directory = Path(directory)

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


class WheelCache:
    """Caches wheels so we do not need to rebuild them.

    Wheels are only cached when the URL contains egg-info or is a VCS repository
    with an *immutable* revision. There might be more than one wheels built for
    one sdist, the one with most preferred tag will be returned.
    """

    def __init__(self, directory: Path | str) -> None:
        self.directory = Path(directory)
        self.ephemeral_directory = Path(
            create_tracked_tempdir(prefix="muse-wheel-cache-")
        )

    def _get_candidates(self, path: Path) -> Iterable[Path]:
        if not path.exists():
            return
        for candidate in path.iterdir():
            if candidate.name.endswith(".whl"):
                yield candidate

    def _get_path_parts(
        self, link: Link, target_python: TargetPython
    ) -> tuple[str, ...]:
        hash_key = {
            "url": link.url_without_fragment,
            # target python participates in the hash key to handle the some cases
            # where the sdist produces different wheels on different Pythons, and
            # the differences are not encoded in compatibility tags.
            "target_python": dataclasses.astuple(target_python),
        }
        if link.subdirectory:
            hash_key["subdirectory"] = link.subdirectory
        if link.hash:
            hash_key[link.hash_name] = link.hash
        hashed = hashlib.sha224(
            json.dumps(
                hash_key, sort_keys=True, separators=(",", ":"), ensure_ascii=True
            ).encode("utf-8")
        ).hexdigest()
        return (hashed[:2], hashed[2:4], hashed[4:6], hashed[6:])

    def get_path_for_link(self, link: Link, target_python: TargetPython) -> Path:
        parts = self._get_path_parts(link, target_python)
        return self.directory.joinpath(*parts)

    def get_ephemeral_path_for_link(
        self, link: Link, target_python: TargetPython
    ) -> Path:
        parts = self._get_path_parts(link, target_python)
        return self.ephemeral_directory.joinpath(*parts)

    def get(
        self, link: Link, project_name: str | None, target_python: TargetPython
    ) -> Path | None:
        if not project_name:
            return None
        canonical_name = canonicalize_name(project_name)
        tags_priorities = {
            tag: i for i, tag in enumerate(target_python.supported_tags())
        }

        candidate = self._get_from_path(
            self.get_path_for_link(link, target_python), canonical_name, tags_priorities
        )
        if candidate is not None:
            return candidate
        return self._get_from_path(
            self.get_ephemeral_path_for_link(link, target_python),
            canonical_name,
            tags_priorities,
        )

    def _get_from_path(
        self, path: Path, canonical_name: str, tags_priorities: dict[Tag, int]
    ) -> Path | None:
        candidates: list[tuple[int, Path]] = []
        for candidate in self._get_candidates(path):
            try:
                name, *_, tags = parse_wheel_filename(candidate.name)
            except ValueError:
                logger.debug("Ignoring invalid cached wheel %s", candidate.name)
                continue
            if canonical_name != canonicalize_name(name):
                logger.debug(
                    "Ignoring cached wheel %s with invalid project name %s, expected: %s",
                    candidate.name,
                    name,
                    canonical_name,
                )
                continue
            if tags.isdisjoint(tags_priorities):
                continue
            support_min = min(
                tags_priorities[tag] for tag in tags if tag in tags_priorities
            )
            candidates.append((support_min, candidate))
        if not candidates:
            return None
        return min(candidates, key=lambda x: x[0])[1]


@lru_cache()
def get_wheel_cache(directory: Path) -> WheelCache:
    return WheelCache(directory)


class RingCache:
    """Caches rings so we do not need to rebuild them.

    Rings are only cached when the URL contains egg-info or is a VCS repository
    with an *immutable* revision. There might be more than one rings built for
    one sdist, the one with most preferred tag will be returned.
    """

    def __init__(self, directory: Path | str) -> None:
        self.directory = Path(directory)
        self.ephemeral_directory = Path(
            create_tracked_tempdir(prefix="muse-ring-cache-")
        )

    def _get_candidates(self, path: Path) -> Iterable[Path]:
        if not path.exists():
            return
        for candidate in path.iterdir():
            if candidate.name.endswith(".ring"):
                yield candidate

    def _get_path_parts(
        self, link: Link, target_mojo  # TODO: TargetMojo
    ) -> tuple[str, ...]:
        hash_key = {
            "url": link.url_without_fragment,
            # target python participates in the hash key to handle the some cases
            # where the sdist produces different wheels on different Pythons, and
            # the differences are not encoded in compatibility tags.
            "target_mojo": dataclasses.astuple(target_mojo),
        }
        if link.subdirectory:
            hash_key["subdirectory"] = link.subdirectory
        if link.hash:
            hash_key[link.hash_name] = link.hash
        hashed = hashlib.sha224(
            json.dumps(
                hash_key, sort_keys=True, separators=(",", ":"), ensure_ascii=True
            ).encode("utf-8")
        ).hexdigest()
        return (hashed[:2], hashed[2:4], hashed[4:6], hashed[6:])

    def get_path_for_link(self, link: Link, target_mojo: TargetMojo) -> Path:
        parts = self._get_path_parts(link, target_mojo)
        return self.directory.joinpath(*parts)

    def get_ephemeral_path_for_link(self, link: Link, target_mojo: TargetMojo) -> Path:
        parts = self._get_path_parts(link, target_mojo)
        return self.ephemeral_directory.joinpath(*parts)

    def get(
        self, link: Link, project_name: str | None, target_mojo: TargetMojo
    ) -> Path | None:
        if not project_name:
            return None
        canonical_name = canonicalize_name(project_name)
        tags_priorities = {tag: i for i, tag in enumerate(target_mojo.supported_tags())}

        candidate = self._get_from_path(
            self.get_path_for_link(link, target_mojo), canonical_name, tags_priorities
        )
        if candidate is not None:
            return candidate
        return self._get_from_path(
            self.get_ephemeral_path_for_link(link, target_mojo),
            canonical_name,
            tags_priorities,
        )

    def _get_from_path(
        self, path: Path, canonical_name: str, tags_priorities: dict[Tag, int]
    ) -> Path | None:
        candidates: list[tuple[int, Path]] = []
        for candidate in self._get_candidates(path):
            try:
                name, *_, tags = parse_ring_filename(candidate.name)
            except ValueError:
                logger.debug("Ignoring invalid cached ring %s", candidate.name)
                continue
            if canonical_name != canonicalize_name(name):
                logger.debug(
                    "Ignoring cached ring %s with invalid project name %s, expected: %s",
                    candidate.name,
                    name,
                    canonical_name,
                )
                continue
            if tags.isdisjoint(tags_priorities):
                continue
            support_min = min(
                tags_priorities[tag] for tag in tags if tag in tags_priorities
            )
            candidates.append((support_min, candidate))
        if not candidates:
            return None
        return min(candidates, key=lambda x: x[0])[1]


KT = TypeVar("KT")
VT = TypeVar("VT")


class JSONFileCache(Generic[KT, VT]):
    """A file cache that stores key-value pairs in a json file."""

    def __init__(self, cache_file: Path | str) -> None:
        self.cache_file = Path(cache_file)
        self._cache: dict[str, VT] = {}
        self._read_cache()

    def _read_cache(self) -> None:
        if not self.cache_file.exists():
            self._cache = {}
            return
        with self.cache_file.open() as fp:
            try:
                self._cache = json.load(fp)
            except json.JSONDecodeError:
                return

    def _write_cache(self) -> None:
        with self.cache_file.open("w") as fp:
            json.dump(self._cache, fp)

    def __contains__(self, obj: KT) -> bool:
        return self._get_key(obj) in self._cache

    @classmethod
    def _get_key(cls, obj: KT) -> str:
        return str(obj)

    def get(self, obj: KT) -> VT:
        key = self._get_key(obj)
        return self._cache[key]

    def set(self, obj: KT, value: VT) -> None:
        key = self._get_key(obj)
        self._cache[key] = value
        self._write_cache()

    def delete(self, obj: KT) -> None:
        try:
            del self._cache[self._get_key(obj)]
        except KeyError:
            pass
        self._write_cache()

    def clear(self) -> None:
        self._cache.clear()
        self._write_cache()


@lru_cache()
def get_ring_cache(directory: Path) -> RingCache:
    return RingCache(directory)


class ProjectCache:
    """Caches project caches so we do not need to rebuild them.
    Used by both, candidates, repositories, and project modules.
    """

    def __init__(
        self,
        root_path: str | Path | None = None,
        global_config: str | Path | None = None,
    ) -> None:
        root_path = root_path or find_project_root()
        self.root = root_path

        if global_config is None:
            global_config = platformdirs.user_config_path("muse") / "config.toml"
        self.global_config = Config(Path(global_config), is_global=True)

    @cached_property
    def project_config(self) -> Config:
        """Read-and-writable configuration dict for project settings"""
        return Config(self.root / DEFAULT_CONFIG_FILENAME)

    @cached_property
    def config(self) -> Mapping[str, Any]:
        """A read-only dict configuration"""
        return collections.ChainMap(self.project_config, self.global_config)

    @property
    def cache_dir(self) -> Path:
        return Path(self.config.get("cache_dir", ""))

    def cache(self, name: str) -> Path:
        path = self.cache_dir / name
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError:
            # The path could be not accessible
            pass
        return path

    def make_hash_cache(self) -> HashCache:
        return HashCache(directory=self.cache("hashes"))

    def make_ring_cache(self) -> RingCache:  # TODO
        return get_ring_cache(self.cache("rings"))  # TODO

    def make_wheel_cache(self) -> WheelCache:  # TODO
        return get_wheel_cache(self.cache("wheels"))  # TODO
