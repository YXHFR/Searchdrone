import sd_safety


def test_import_has_version() -> None:
    assert isinstance(sd_safety.__version__, str)
    assert sd_safety.__version__
