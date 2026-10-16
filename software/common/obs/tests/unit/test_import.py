import sd_obs


def test_import_has_version() -> None:
    assert isinstance(sd_obs.__version__, str)
    assert sd_obs.__version__
