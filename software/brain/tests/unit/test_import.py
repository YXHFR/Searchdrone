import sd_brain


def test_import_has_version() -> None:
    assert isinstance(sd_brain.__version__, str)
    assert sd_brain.__version__
