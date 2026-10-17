import sd_testing


def test_import_has_version() -> None:
    assert isinstance(sd_testing.__version__, str)
    assert sd_testing.__version__
