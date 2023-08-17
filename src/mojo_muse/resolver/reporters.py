from typing import Iterable

from .._types import Spinner
from ..models.requirements import BaseMuseRequirement
from ..project import Project
from ..termui import SilentSpinner


def get_reporter(
    project: Project,
    requirements: list[BaseMuseRequirement],
    tracked_names: Iterable[str] | None = None,
    spinner: Spinner | None = None,
) -> BaseReporter:
    """Return the reporter object to construct a resolver.

    :param requirements: requirements to resolve
    :param tracked_names: the names of packages that needs to update
    :param spinner: optional spinner object
    :returns: a reporter
    """

    if spinner is None:
        spinner = SilentSpinner("")

    return SpinnerReporter(spinner, requirements)
