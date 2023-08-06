from packaging.specifiers import Specifier, SpecifierSet, InvalidSpecifier
from packaging.version import Version

from functools import lru_cache

@lru_cache()
def get_specifier(version_str: SpecifierSet | str) -> SpecifierSet:
    if isinstance(version_str, SpecifierSet):
        return version_str
    if not version_str or version_str == "*":
        return SpecifierSet()
    return SpecifierSet(version_str)