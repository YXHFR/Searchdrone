import sd_geo


def test_import_has_version() -> None:
    assert isinstance(sd_geo.__version__, str)
    assert sd_geo.__version__
