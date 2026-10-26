import sd_motion


def test_import_has_version() -> None:
    assert isinstance(sd_motion.__version__, str)
    assert sd_motion.__version__
