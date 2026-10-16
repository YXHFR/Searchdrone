import sd_onboard


def test_import_has_version() -> None:
    assert isinstance(sd_onboard.__version__, str)
    assert sd_onboard.__version__
