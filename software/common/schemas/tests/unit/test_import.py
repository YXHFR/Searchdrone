import sd_schemas


def test_import_has_version() -> None:
    assert isinstance(sd_schemas.__version__, str)
    assert sd_schemas.__version__
