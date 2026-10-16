import sd_link


def test_import_has_version() -> None:
    assert isinstance(sd_link.__version__, str)
    assert sd_link.__version__
