import sd_ground


def test_import_has_version() -> None:
    assert isinstance(sd_ground.__version__, str)
    assert sd_ground.__version__
