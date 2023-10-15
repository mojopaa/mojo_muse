"""Original: pdm.cli.filters.py. Used in do_padd()."""

from __future__ import annotations

import argparse
from functools import cached_property
from typing import Iterator, Sequence

from ..exceptions import MuseUsageError
from ..project import Project


class GroupSelection:
    def __init__(
        self,
        project: Project,
        *,
        default: bool = True,
        dev: bool | None = None,
        groups: Sequence[str] = (),
        group: str | None = None,
    ):
        self.project = project
        self.groups = groups
        self.group = group
        self.default = default
        self.dev = dev

    # @classmethod
    # def from_options(
    #     cls, project: Project, options: argparse.Namespace
    # ) -> GroupSelection:
    #     if "group" in options:
    #         return cls(project, group=options.group, dev=options.dev)
    #     return cls(
    #         project,
    #         default=options.default,
    #         dev=options.dev,
    #         groups=options.groups,
    #     )

    def one(self) -> str:
        if self.group:
            return self.group
        if len(self.groups) == 1:
            return self.groups[0]
        return "dev" if self.dev else "default"

    @property
    def is_unset(self) -> bool:
        return self.default and self.dev is None and not self.groups

    def all(self) -> list[str] | None:
        project_groups = list(self.project.iter_groups())
        if self.is_unset:
            if self.project.lockfile.exists():
                groups = self.project.lockfile.groups
                if groups:
                    groups = [g for g in groups if g in project_groups]
                return groups
        return list(self)

    @cached_property
    def _translated_groups(self) -> list[str]:  # TODO: only ["default"]?
        """Translate default, dev and groups containing ":all" into a list of groups"""
        if self.is_unset:
            # Default case, return what is in the lock file
            locked_groups = self.project.lockfile.groups
            project_groups = list(self.project.iter_groups())
            if locked_groups:
                return [g for g in locked_groups if g in project_groups]
        default, dev, groups = self.default, self.dev, self.groups
        if dev is None:  # --prod is not set, include dev-dependencies
            dev = True

        project = self.project
        if project.is_mojoproject:
            project_file = project.mojoproject
        elif project.is_pryproject:
            project_file = project.pyproject
        else:
            project_file = project.mojoproject

        optional_groups = set(project_file.metadata.get("optional-dependencies", {}))
        dev_groups = set(project_file.settings.get("dev-dependencies", {}))
        groups_set = set(groups)
        if groups_set & dev_groups:
            if not dev:
                raise MuseUsageError(
                    "--prod is not allowed with dev groups and should be left"
                )
        elif dev:
            groups_set.update(dev_groups)
        if ":all" in groups:
            groups_set.discard(":all")
            groups_set.update(optional_groups)

        invalid_groups = groups_set - set(project.iter_groups())
        if invalid_groups:
            project.ui.echo(
                "[d]Ignoring non-existing groups: [success]"
                f"{', '.join(invalid_groups)}[/]",
                err=True,
            )
            groups_set -= invalid_groups
        # Sorts the result in ascending order instead of in random order
        # to make this function pure
        result = sorted(groups_set)
        if default:
            result.insert(0, "default")
        return result

    def validate(self) -> None:
        extra_groups = self.project.lockfile.compare_groups(self._translated_groups)
        if extra_groups:
            raise MuseUsageError(
                f"Requested groups not in lockfile: {','.join(extra_groups)}"
            )

    def __iter__(self) -> Iterator[str]:
        return iter(self._translated_groups)

    def __contains__(self, group: str) -> bool:
        return group in self._translated_groups
