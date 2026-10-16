import sd_hwcalc


def test_import_has_version() -> None:
    assert isinstance(sd_hwcalc.__version__, str)
    assert sd_hwcalc.__version__
