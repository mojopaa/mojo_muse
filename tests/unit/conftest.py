import pytest

from mojo_muse.session import MojoPISession


@pytest.fixture()
def session():
    s = MojoPISession()
    try:
        yield s
    finally:
        s.close()
