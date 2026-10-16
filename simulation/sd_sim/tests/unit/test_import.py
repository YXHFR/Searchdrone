import sd_sim


def test_import_has_version() -> None:
    assert isinstance(sd_sim.__version__, str)
    assert sd_sim.__version__
