import itertools

from packaging.markers import Marker

from ..utils import join_list_with


def split_marker_extras(marker: str) -> tuple[set[str], str]:
    """An element can be stripped from the marker only if all parts are connected
    with `and` operator. The rest part are returned as a string or `None` if all are
    stripped.
    """

    def extract_extras(submarker: tuple | list) -> set[str]:
        if isinstance(submarker, tuple):
            if submarker[0].value == "extra":
                if submarker[1].value == "==":
                    return {submarker[2].value}
                elif submarker[1].value == "in":
                    return {v.strip() for v in submarker[2].value.split(",")}
                else:
                    return set()
            else:
                return set()
        else:
            if "and" in submarker:
                return set()
            pure_extras = [extract_extras(m) for m in submarker if m != "or"]
            if all(pure_extras):
                return set(itertools.chain.from_iterable(pure_extras))
            return set()

    if not marker:
        return set(), marker
    new_marker = Marker(marker)
    submarkers = Marker(marker)._markers
    if "or" in submarkers:
        extras = extract_extras(submarkers)
        if extras:
            return extras, ""
        return set(), marker

    extras = set()
    submarkers_no_extras: list[tuple | list] = []
    # Below this point the submarkers are connected with 'and'
    for submarker in submarkers:
        if submarker == "and":
            continue
        new_extras = extract_extras(submarker)
        if new_extras:
            if extras:
                # extras are not allowed to appear in more than one parts
                return set(), marker
            extras.update(new_extras)
        else:
            submarkers_no_extras.append(submarker)

    if not submarkers_no_extras:
        return extras, ""
    new_marker._markers = join_list_with(submarkers_no_extras, "and")
    return extras, str(new_marker)
