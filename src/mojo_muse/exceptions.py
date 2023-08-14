class MuseException(Exception):
    pass


class MuseUsageError(MuseException):
    pass


class NoConfigError(MuseUsageError, KeyError):
    def __str__(self) -> str:
        return f"Not such config key: {self.args[0]!r}"


class RequirementError(MuseException):
    pass


class CandidateInfoNotFound(MuseException):
    pass


class CandidateNotFound(MuseException):
    pass


class ExtrasWarning(UserWarning):
    def __init__(self, project_name: str, extras: list[str]) -> None:
        super().__init__(f"Extras not found for {project_name}: [{','.join(extras)}]")
        self.extras = tuple(extras)
