from functools import lru_cache

from packaging.specifiers import InvalidSpecifier, Specifier, SpecifierSet
from packaging.version import Version


@lru_cache()
def get_specifier(version_str: SpecifierSet | str) -> SpecifierSet:
    if isinstance(version_str, SpecifierSet):
        return version_str
    if not version_str or version_str == "*":
        return SpecifierSet()
    return SpecifierSet(version_str)
