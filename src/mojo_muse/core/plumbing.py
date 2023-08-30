from typing import cast

from resolvelib import ResolutionImpossible
from resolvelib.resolvers import RequirementInformation

from ..resolver.python import PythonRequirement
from ..models.specifiers import PySpecSet


def format_resolution_impossible(err: ResolutionImpossible) -> str:
    causes: list[RequirementInformation] = err.causes
    info_lines: set[str] = set()
    if all(isinstance(cause.requirement, PythonRequirement) for cause in causes):
        project_requires: PythonRequirement = next(cause.requirement for cause in causes if cause.parent is None)
        pyspec = cast(PySpecSet, project_requires.specifier)
        conflicting = [
            cause
            for cause in causes
            if cause.parent is not None and not cause.requirement.specifier.is_superset(pyspec)  # TODO: use PySpec.is_superset in specifiers.py
        ]
        result = [
            "Unable to find a resolution because the following dependencies don't work "
            "on all Python versions in the range of the project's `requires-python`: "
            f"[success]{pyspec}[/]."
        ]
        for req, parent in conflicting:
            pyspec &= req.specifier
            info_lines.add(f"  {req.as_line()} (from {parent!r})")
        result.extend(sorted(info_lines))
        if pyspec.is_impossible:
            result.append("Consider changing the version specifiers of the dependencies to be compatible")
        else:
            result.append(
                "A possible solution is to change the value of `requires-python` "
                f"in pyproject.toml to [success]{pyspec}[/]."
            )
        return "\n".join(result)

    if len(causes) == 1:
        return (
            "Unable to find a resolution for "
            f"[success]{causes[0].requirement.identify()}[/]\n"
            "Please make sure the package name is correct."
        )

    result = [
        "Unable to find a resolution for "
        f"[success]{causes[0].requirement.identify()}[/]\n"
        "because of the following conflicts:"
    ]
    for req, parent in causes:
        info_lines.add(f"  {req.as_line()} (from {parent if parent else 'project'})")
    result.extend(sorted(info_lines))
    result.append(
        "To fix this, you could loosen the dependency version constraints in "
        "pyproject.toml. See https://pdm.fming.dev/latest/usage/dependency/"
        "#solve-the-locking-failure for more details."
    )
    return "\n".join(result)