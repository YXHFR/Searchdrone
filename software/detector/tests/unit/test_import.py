import sd_detect


def test_import_has_version() -> None:
    assert isinstance(sd_detect.__version__, str)
    assert sd_detect.__version__
