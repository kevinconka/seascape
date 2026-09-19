import seascape


def test_package_is_installed() -> None:
    # __version__ reads distribution metadata, so this fails if seascape is on
    # sys.path without being installed.
    assert seascape.__version__
