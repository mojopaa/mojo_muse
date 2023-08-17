import dataclasses
import posixpath
import sys
from abc import ABC, abstractmethod
from functools import wraps
from typing import Any, Callable, Iterable, Mapping, TypeVar, cast

from mups import normalize_name, parse_ring_filename
from packaging.specifiers import SpecifierSet

from .. import termui
from .._types import CandidateInfo, FileHash, RepositoryConfig, SearchResult
from ..exceptions import CandidateInfoNotFound, CandidateNotFound
from ..utils import cd, path_to_url, url_to_path, url_without_fragments
from .candidates import Candidate, make_candidate
from .link import Link
from .requirements import (
    BaseMuseRequirement,
    MuseRequirement,
    filter_requirements_with_extras,
    parse_requirement,
)

T = TypeVar("T", bound="BaseRepository")
CandidateKey = tuple[str, str | None, str | None, bool]


def cache_result(
    func: Callable[[T, Candidate], CandidateInfo]
) -> Callable[[T, Candidate], CandidateInfo]:
    @wraps(func)
    def wrapper(self: T, candidate: Candidate) -> CandidateInfo:
        result = func(self, candidate)
        prepared = candidate.prepared
        if prepared and prepared.should_cache():
            self._candidate_info_cache.set(candidate, result)
        return result

    return wrapper


class BaseRepository(ABC):
    """A Repository acts as the source of packages and metadata."""

    def __init__(
        self,
        sources: list[RepositoryConfig],
        ignore_compatibility: bool = True,
    ) -> None:
        """Initialize the package manager.

        Args:
            sources (list[RepositoryConfig]): A list of sources to download packages from.
            environment (BaseEnvironment): The bound environment instance.
            ignore_compatibility (bool, optional): If True, don't evaluate candidate against
                the current environment. Defaults to True.
        """
        self.sources = sources
        self.ignore_compatibility = ignore_compatibility
        # self._candidate_info_cache = environment.project.make_candidate_info_cache()
        # self._hash_cache = environment.project.make_hash_cache()

    @abstractmethod
    def dependency_generators(self) -> Iterable[Callable[[Candidate], CandidateInfo]]:
        """Return an iterable of getter functions to get dependencies, which will be
        called one by one.
        """

    @abstractmethod
    def _find_candidates(self, requirement: BaseMuseRequirement) -> Iterable[Candidate]:
        pass

    @abstractmethod
    def search(self, query: str) -> SearchResult:
        """
        Search package by name or summary.

        Args:
            query (str): query string

        Returns:
            SearchResult: search result, a dictionary of name: package metadata
            In _types it's list[Package]
        """

    def get_filtered_sources(self, req: BaseMuseRequirement) -> list[RepositoryConfig]:
        """Get matching sources based on the index attribute."""
        return self.sources

    # def get_dependencies(
    #     self, candidate: Candidate
    # ) -> tuple[list[BaseMuseRequirement], SpecifierSet, str]:
    #     """Get (dependencies, mojo_specifier, summary) of the candidate."""
    #     requires_mojo, summary = "", ""
    #     requirements: list[str] = []
    #     last_ext_info = None
    #     for getter in self.dependency_generators():
    #         try:
    #             requirements, requires_mojo, summary = getter(candidate)
    #         except CandidateInfoNotFound:
    #             last_ext_info = sys.exc_info()
    #             continue
    #         break
    #     else:
    #         if last_ext_info is not None:
    #             raise last_ext_info[1].with_traceback(last_ext_info[2])  # type: ignore[union-attr]
    #     reqs: list[BaseMuseRequirement] = []
    #     for line in requirements:
    #         if line.startswith("-e "):
    #             reqs.append(parse_requirement(line[3:], True))
    #         else:
    #             reqs.append(parse_requirement(line))
    #     if candidate.req.extras:
    #         # XXX: If the requirement has extras, add the original candidate
    #         # (without extras) as its dependency. This ensures the same package with
    #         # different extras resolve to the same version.
    #         self_req = dataclasses.replace(
    #             candidate.req.as_pinned_version(candidate.version),
    #             extras=None,
    #             marker=None,
    #         )
    #         reqs.append(self_req)
    #     # Store the metadata on the candidate for caching
    #     candidate.requires_mojo = requires_mojo
    #     candidate.summary = summary
    #     if not self.ignore_compatibility:
    #         pep508_env = self.environment.marker_environment
    #         reqs = [
    #             req for req in reqs if not req.marker or req.marker.evaluate(pep508_env)
    #         ]
    #     return reqs, SpecifierSet(requires_mojo), summary

    # def is_this_package(self, requirement: BaseMuseRequirement) -> bool:
    # TODO: move to environment or project
    #     """Whether the requirement is the same as this package"""
    #     project = self.environment.project
    #     return (
    #         requirement.is_named
    #         and project.name is not None
    #         and requirement.key == normalize_name(project.name)
    #     )

    # def make_this_candidate(self, requirement: BaseMuseRequirement) -> Candidate:
    # TODO: move to environment or project
    #     """Make a candidate for this package.
    #     In this case the finder will look for a candidate from the package sources
    #     """
    #     from .link import Link

    #     project = self.environment.project
    #     assert project.name
    #     link = Link.from_path(project.root)
    #     candidate = make_candidate(requirement, project.name, link=link)
    #     candidate.prepare(self.environment).metadata
    #     return candidate

    def find_candidates(
        self,
        requirement: BaseMuseRequirement,
        allow_prereleases: bool | None = None,
        ignore_requires_mojo: bool = False,
    ) -> Iterable[Candidate]:
        """Find candidates of the given NamedRequirement. Let it to be implemented in
        subclasses.
        """
        # `allow_prereleases` is None means leave it to specifier to decide whether to
        # include prereleases

        if self.is_this_package(requirement):
            return [self.make_this_candidate(requirement)]
        requires_mojo = requirement.requires_mojo & self.environment.requires_mojo
        cans = self._find_candidates(requirement)
        applicable_cans = [
            c
            for c in cans
            if requirement.specifier.contains(c.version, allow_prereleases)  # type: ignore[arg-type, union-attr]
        ]

        applicable_cans_mojo_compatible = [
            c
            for c in applicable_cans
            if ignore_requires_mojo or requires_mojo.is_subset(c.requires_mojo)
        ]
        # Evaluate data-requires-mojo attr and discard incompatible candidates
        # to reduce the number of candidates to resolve.
        if applicable_cans_mojo_compatible:
            applicable_cans = applicable_cans_mojo_compatible

        if not applicable_cans:
            termui.logger.debug("\tCould not find any matching candidates.")

        if not applicable_cans and allow_prereleases is None:
            # No non-pre-releases is found, force pre-releases now
            applicable_cans = [
                c for c in cans if requirement.specifier.contains(c.version, True)  # type: ignore[arg-type, union-attr]
            ]
            applicable_cans_mojo_compatible = [
                c
                for c in applicable_cans
                if ignore_requires_mojo or requires_mojo.is_subset(c.requires_mojo)
            ]
            if applicable_cans_mojo_compatible:
                applicable_cans = applicable_cans_mojo_compatible

            if not applicable_cans:
                termui.logger.debug(
                    "\tCould not find any matching candidates even when considering pre-releases.",
                )

        def log_candidates(
            title: str, candidates: Iterable[Candidate], max_lines: int = 10
        ) -> None:
            termui.logger.debug("\t" + title)
            logged_lines = set()
            for can in candidates:
                new_line = f"\t  {can!r}"
                if new_line not in logged_lines:
                    logged_lines.add(new_line)
                    if len(logged_lines) > max_lines:
                        termui.logger.debug("\t  ... [more]")
                        break
                    else:
                        termui.logger.debug(new_line)

        if self.environment.project.core.ui.verbosity >= termui.Verbosity.DEBUG:
            if applicable_cans:
                log_candidates("Found matching candidates:", applicable_cans)
            elif cans:
                log_candidates("Found but non-matching candidates:", cans)

        return applicable_cans

    def _get_dependencies_from_cache(self, candidate: Candidate) -> CandidateInfo:
        try:
            result = self._candidate_info_cache.get(candidate)
        except KeyError:
            raise CandidateInfoNotFound(candidate)
        return result

    @cache_result
    def _get_dependencies_from_metadata(self, candidate: Candidate) -> CandidateInfo:
        prepared = candidate.prepare(self.environment)
        deps = prepared.get_dependencies_from_metadata()
        requires_mojo = candidate.requires_mojo
        summary = prepared.metadata.metadata["Summary"]  # TODO: "Summary" or "summary"?
        return deps, requires_mojo, summary

    def _get_dependency_from_local_package(self, candidate: Candidate) -> CandidateInfo:
        """Adds the local package as a candidate only if the candidate
        name is the same as the local package."""
        project = self.environment.project
        if not project.name or candidate.name != project.name:
            raise CandidateInfoNotFound(candidate)
        reqs = project.mojoproject.metadata.get(
            "dependencies", []
        )  # TODO: mojoproject change!
        extra_dependencies = project.mojoproject.settings.get(
            "dev-dependencies", {}
        ).copy()
        extra_dependencies.update(
            project.mojoproject.metadata.get("optional-dependencies", {})
        )
        if candidate.req.extras is not None:
            reqs = sum(
                (extra_dependencies.get(g, []) for g in candidate.req.extras),
                [],
            )

        return (
            reqs,
            str(self.environment.requires_mojo),
            project.mojoproject.metadata.get("description", "UNKNOWN"),
        )

    def _is_mojo_match(self, link: Link) -> bool:
        from packaging.tags import Tag

        # from packaging.utils import parse_wheel_filename

        def is_tag_match(tag: Tag, requires_mojo: SpecifierSet) -> bool:
            if tag.interpreter.startswith(("cp", "py")):
                major, minor = tag.interpreter[2], tag.interpreter[3:]
                if not minor:
                    version = f"{major}.0"
                else:
                    version = f"{major}.{minor}.0"
                if tag.abi == "abi3":
                    spec = SpecifierSet(
                        f">={version}"
                    )  # cp37-abi3 is compatible with >=3.7
                else:
                    spec = SpecifierSet(
                        f"~={version}"
                    )  # cp37-cp37 is only compatible with 3.7.*
                return not (spec & requires_mojo).is_impossible
            else:
                # we don't know about compatility for non-cpython implementations
                # assume it is compatible
                return True

        if not link.is_ring:
            return True
        requires_mojo = self.environment.requires_mojo
        tags = parse_ring_filename(link.filename)[-1]
        result = any(is_tag_match(tag, requires_mojo) for tag in tags)
        if not result:
            termui.logger.debug(
                "Skipping %r because it is not compatible with %r",
                link,
                requires_mojo,
            )
        return result

    def get_hashes(self, candidate: Candidate) -> list[FileHash]:
        """Get hashes of all possible installable candidates
        of a given package version.
        """
        if (
            candidate.req.is_vcs
            or candidate.req.is_file_or_url
            and candidate.req.is_local_dir  # type: ignore[attr-defined]
        ):
            return []
        if candidate.hashes:
            return candidate.hashes
        req = candidate.req.as_pinned_version(candidate.version)
        comes_from = candidate.link.comes_from if candidate.link else None
        result: list[FileHash] = []
        logged = False
        respect_source_order = self.environment.project.pyproject.settings.get(
            "resolution", {}
        ).get("respect-source-order", False)
        if req.is_named and respect_source_order and comes_from:
            sources = [s for s in self.sources if comes_from.startswith(s.url)]
        else:
            sources = self.sources
        with self.environment.get_finder(sources, self.ignore_compatibility) as finder:
            if req.is_file_or_url:
                this_link = cast("Link", candidate.prepare(self.environment).link)
                links: list[Link] = [this_link]
            else:  # the req must be a named requirement
                links = [package.link for package in finder.find_matches(req.as_line())]
                if self.ignore_compatibility:
                    links = [link for link in links if self._is_mojo_match(link)]
            for link in links:
                if not link or link.is_vcs or link.is_file and link.file_path.is_dir():
                    # The links found can still be a local directory or vcs, skippping it.
                    continue
                if not logged:
                    termui.logger.info("Fetching hashes for %s", candidate)
                    logged = True
                result.append(
                    {
                        "url": link.url_without_fragment,
                        "file": link.filename,
                        "hash": self._hash_cache.get_hash(link, finder.session),
                    }
                )
        return result


class MojoPIRepository(BaseRepository):
    """Get package and metadata from MojoPI source."""

    DEFAULT_INDEX_URL = "http://127.0.0.1:5000"  # TODO: make it configurable

    @cache_result
    def _get_dependencies_from_json(self, candidate: Candidate) -> CandidateInfo:
        # TODO: fetch mojo package metadata json
        if not candidate.name or not candidate.version:
            # Only look for json api for named requirements.
            raise CandidateInfoNotFound(candidate)
        sources = self.get_filtered_sources(candidate.req)
        url_prefixes = [
            proc_url[:-7]  # Strip "/simple".
            for proc_url in (
                raw_url.rstrip("/")
                for raw_url in (source.url for source in sources)
                if raw_url
            )
            if proc_url.endswith("/simple")
        ]
        # with self.environment.get_finder(sources) as finder:
        #     session = finder.session
        #     for prefix in url_prefixes:
        #         json_url = f"{prefix}/pypi/{candidate.name}/{candidate.version}/json"
        #         resp = session.get(json_url)
        #         if not resp.ok:
        #             continue

        #         info = resp.json()["info"]

        #         requires_python = info["requires_python"] or ""
        #         summary = info["summary"] or ""
        #         try:
        #             requirement_lines = info["requires_dist"] or []
        #         except KeyError:
        #             requirement_lines = info["requires"] or []
        #         requirements = filter_requirements_with_extras(
        #             cast(str, candidate.req.project_name),
        #             requirement_lines,
        #             candidate.req.extras or (),
        #         )
        #         return requirements, requires_python, summary
        # raise CandidateInfoNotFound(candidate)

    def dependency_generators(self) -> Iterable[Callable[[Candidate], CandidateInfo]]:
        yield self._get_dependencies_from_cache
        yield self._get_dependency_from_local_package
        if self.environment.project.config["mojopi.json_api"]:
            yield self._get_dependencies_from_json
        yield self._get_dependencies_from_metadata

    def _find_candidates(self, requirement: BaseMuseRequirement) -> Iterable[Candidate]:
        sources = self.get_filtered_sources(requirement)
        with self.environment.get_finder(sources, self.ignore_compatibility) as finder:
            cans = [
                Candidate.from_installation_candidate(c, requirement)
                for c in finder.find_all_packages(
                    requirement.project_name, allow_yanked=requirement.is_pinned
                )
            ]
        if not cans:
            raise CandidateNotFound(
                f"Unable to find candidates for {requirement.project_name}. There may "
                "exist some issues with the package name or network condition."
            )
        return cans

    def search(self, query: str) -> SearchResult:
        pass
        # pypi_simple = self.sources[0].url.rstrip("/")  # type: ignore[union-attr]

        # if pypi_simple.endswith("/simple"):
        #     search_url = pypi_simple[:-6] + "search"
        # else:
        #     search_url = pypi_simple + "/search"

        # with self.environment.get_finder() as finder:
        #     session = finder.session
        #     resp = session.get(search_url, params={"q": query})
        #     if resp.status_code == 404:
        #         self.environment.project.core.ui.echo(
        #             f"{pypi_simple!r} doesn't support '/search' endpoint, fallback "
        #             f"to {self.DEFAULT_INDEX_URL!r} now.\n"
        #             "This may take longer depending on your network condition.",
        #             err=True,
        #             style="warning",
        #         )
        #         resp = session.get(f"{self.DEFAULT_INDEX_URL}/search", params={"q": query})
        #     parser = SearchResultParser()
        #     resp.raise_for_status()
        #     parser.feed(resp.text)
        #     return parser.results


class LockedRepository(BaseRepository):
    def __init__(
        self,
        lockfile: Mapping[str, Any],
        sources: list[RepositoryConfig],
    ) -> None:
        super().__init__(sources=sources, ignore_compatibility=False)
        self.packages: dict[CandidateKey, Candidate] = {}
        self.candidate_info: dict[CandidateKey, CandidateInfo] = {}
        self._read_lockfile(lockfile)

    @property
    def all_candidates(self) -> dict[str, Candidate]:
        return {can.req.identify(): can for can in self.packages.values()}

    def _read_lockfile(self, lockfile: Mapping[str, Any]) -> None:
        # root = self.environment.project.root
        # with cd(root):

        for package in lockfile.get("package", []):
            version = package.get("version")
            if version:
                package["version"] = f"=={version}"
            package_name = package.pop("name")
            req_dict = {
                k: v
                for k, v in package.items()
                if k not in ("dependencies", "requires_mojo", "summary", "files")
            }
            req = MuseRequirement.from_req_dict(package_name, req_dict)
            if req.is_file_or_url and req.path and not req.url:  # type: ignore[attr-defined]
                req.url = path_to_url(posixpath.join(root, req.path))  # type: ignore[attr-defined]
            can = make_candidate(req, name=package_name, version=version)
            can.hashes = package.get("files", [])
            can_id = self._identify_candidate(can)
            self.packages[can_id] = can
            candidate_info: CandidateInfo = (
                package.get("dependencies", []),
                package.get("requires_python", ""),
                package.get("summary", ""),
            )
            self.candidate_info[can_id] = candidate_info

    def _identify_candidate(self, candidate: Candidate) -> CandidateKey:
        url: str | None = None
        if candidate.link is not None:
            url = candidate.link.url_without_fragment
            # url = self.environment.project.backend.expand_line(cast(str, url))
            if url.startswith("file://"):
                path = posixpath.normpath(url_to_path(url))
                url = path_to_url(path)
        return (
            candidate.identify(),
            candidate.version if not url else None,
            url,
            candidate.req.editable,
        )

    def _get_dependencies_from_lockfile(self, candidate: Candidate) -> CandidateInfo:
        return self.candidate_info[self._identify_candidate(candidate)]

    def dependency_generators(self) -> Iterable[Callable[[Candidate], CandidateInfo]]:
        return (
            self._get_dependency_from_local_package,
            self._get_dependencies_from_lockfile,
        )

    def _matching_keys(
        self, requirement: BaseMuseRequirement
    ) -> Iterable[CandidateKey]:
        from .requirements import FileMuseRequirement

        for key in self.candidate_info:
            can_req = self.packages[key].req
            if requirement.name:
                if key[0] != requirement.identify():
                    continue
            else:
                assert isinstance(requirement, FileMuseRequirement)
                if not isinstance(can_req, FileMuseRequirement):
                    continue
                if requirement.path and can_req.path:
                    if requirement.path != can_req.path:
                        continue
                elif key[2] is not None and key[2] != url_without_fragments(
                    requirement.url
                ):
                    continue

            yield key

    def find_candidates(
        self,
        requirement: BaseMuseRequirement,
        allow_prereleases: bool | None = None,
        ignore_requires_mojo: bool = False,
    ) -> Iterable[Candidate]:
        if self.is_this_package(requirement):
            candidate = self.make_this_candidate(requirement)
            if candidate is not None:
                yield candidate
                return
        for key in self._matching_keys(requirement):
            info = self.candidate_info[key]
            if not SpecifierSet(info[1]).contains(
                str(self.environment.interpreter.version), True
            ):
                continue
            can = self.packages[key]
            can.requires_mojo = info[1]
            if not requirement.name:
                # make sure can.identify() won't return a randomly-generated name
                requirement.name = can.name
            can.req = requirement
            yield can

    def get_hashes(self, candidate: Candidate) -> list[FileHash]:
        return candidate.hashes
