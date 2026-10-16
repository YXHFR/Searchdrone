import sd_link_testing


def test_import_has_version() -> None:
    assert isinstance(sd_link_testing.__version__, str)
    assert sd_link_testing.__version__
