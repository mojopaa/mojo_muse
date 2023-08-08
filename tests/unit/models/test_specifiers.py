import pytest

from mojo_muse.models.specifiers import (
    InvalidSpecifier,
    Specifier,
    SpecifierSet,
    get_specifier,
)


def test_get_specifier():
    assert get_specifier(">1.2.3") == SpecifierSet(">1.2.3")
    assert get_specifier(">1.2.3.4") == SpecifierSet(">1.2.3.4")
    assert get_specifier(">1.2.3.4.5") == SpecifierSet(">1.2.3.4.5")

    with pytest.raises(InvalidSpecifier):
        get_specifier("1.2.3")
