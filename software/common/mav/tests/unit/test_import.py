import sd_mav


def test_import_has_version() -> None:
    assert isinstance(sd_mav.__version__, str)
    assert sd_mav.__version__
