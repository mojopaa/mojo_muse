from logging import Logger

from mojo_muse.termui import UI, Verbosity, logger, ui


def test_ui():
    assert ui is not None
    ui.echo("test")
    assert ui.verbosity in set(Verbosity)


def test_UI():
    assert UI is not None
    assert UI() is not None


def test_logger():
    assert isinstance(logger, Logger)
