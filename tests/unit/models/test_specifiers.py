from mojo_muse.models.specifiers import Specifier, SpecifierSet, InvalidSpecifier, get_specifier
import pytest

def test_get_specifier():
    assert get_specifier(">1.2.3") == SpecifierSet(">1.2.3")
    assert get_specifier(">1.2.3.4") == SpecifierSet(">1.2.3.4")
    assert get_specifier(">1.2.3.4.5") == SpecifierSet(">1.2.3.4.5")

    with pytest.raises(InvalidSpecifier):
        get_specifier("1.2.3")