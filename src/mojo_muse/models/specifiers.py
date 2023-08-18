import re
from functools import lru_cache
from re import Match

from packaging.specifiers import InvalidSpecifier, Specifier, SpecifierSet
from packaging.version import Version

from ..exceptions import deprecation_warning


@lru_cache()
def get_specifier(version_str: SpecifierSet | str) -> SpecifierSet:
    if isinstance(version_str, SpecifierSet):
        return version_str
    if not version_str or version_str == "*":
        return SpecifierSet()
    return SpecifierSet(version_str)


@lru_cache()
def fix_legacy_specifier(specifier: str) -> str:
    """Since packaging 22.0, legacy specifiers like '>=4.*' are no longer
    supported. We try to normalize them to the new format.
    """
    _legacy_specifier_re = re.compile(r"(==|!=|<=|>=|<|>)(\s*)([^,;\s)]*)")

    def fix_wildcard(match: Match[str]) -> str:
        operator, _, version = match.groups()
        if operator in ("==", "!="):
            return match.group(0)
        if ".*" in version:
            deprecation_warning(
                ".* suffix can only be used with `==` or `!=` operators", stacklevel=4
            )
            version = version.replace(".*", ".0")
            if operator in ("<", "<="):  # <4.* and <=4.* are equivalent to <4.0
                operator = "<"
            elif operator in (">", ">="):  # >4.* and >=4.* are equivalent to >=4.0
                operator = ">="
        elif "+" in version:  # Drop the local version
            deprecation_warning(
                "Local version label can only be used with `==` or `!=` operators",
                stacklevel=4,
            )
            version = version.split("+")[0]
        return f"{operator}{version}"

    return _legacy_specifier_re.sub(fix_wildcard, specifier)
