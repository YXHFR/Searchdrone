"""The image bank's register and its licence rule (tools/bench/datasets.toml)."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
from bench.common import BenchError
from bench.datasets import Dataset, licence_problems, load_register


def _row(**kw: object) -> Dataset:
    base = Dataset(
        name="x",
        title="x",
        licence="CC-BY-4.0",
        attribution="someone",
        layout="heridal_voc",
        use=("train", "measure"),
        publish_weights=False,
        local_only=False,
        url="https://example.org/x.zip",
        size_bytes=1,
        sha256="0" * 64,
        licence_confirmed=True,
    )
    return dataclasses.replace(base, **kw)  # type: ignore[arg-type]


def test_register_keeps_the_licence_rule() -> None:
    register = load_register()
    assert {"heridal", "wisard", "simfarm"} <= set(register)
    for ds in register.values():
        assert licence_problems(ds) == [], ds.name


def test_every_downloaded_archive_is_pinned_by_size() -> None:
    for ds in load_register().values():
        if ds.url:
            assert ds.url.startswith("https://")
            assert ds.size_bytes and ds.size_bytes > 0
            assert ds.sha256 is not None


def test_kios_is_open_credited_and_pinned_with_its_labels() -> None:
    kios = load_register()["kios"]
    assert kios.licence == "CC-BY-4.0" and kios.licence_confirmed
    assert set(kios.use) == {"train", "measure"} and kios.publish_weights
    assert "Makrigiorgis" in kios.attribution and "7740081" in kios.attribution
    for url in (kios.url, kios.labels_url, kios.source_page):
        assert url and url.startswith("https://zenodo.org/")  # the authors' record, no mirror
    assert kios.labels_size_bytes and kios.labels_sha256 and len(kios.labels_sha256) == 64


def test_sim_farm_is_only_for_pre_training() -> None:
    simfarm = load_register()["simfarm"]
    assert simfarm.use == ("pretrain",)
    assert simfarm.local_only


@pytest.mark.parametrize(
    ("row", "why"),
    [
        (_row(licence="proprietary"), "not allowed"),
        (_row(licence="LicenseRef-none"), "not allowed"),
        (_row(licence="GPL-3.0-only", local_only=True, publish_results=False), "GPL"),
        (_row(licence="NONE", use=("measure",), publish_results=False), "GPL"),
        (_row(licence="NONE", use=("measure",), local_only=True), "GPL"),
        (
            _row(
                licence="GPL-3.0-or-later",
                use=("measure",),
                local_only=True,
                publish_weights=True,
                publish_results=False,
            ),
            "GPL",
        ),
        (_row(licence="CC-BY-NC-4.0"), "non-commercial"),
        (_row(licence="CC-BY-NC-SA-4.0", use=("measure",), local_only=False), "non-commercial"),
        (
            _row(licence="CC-BY-NC-4.0", use=("measure",), local_only=True, publish_weights=True),
            "non-commercial",
        ),
        (_row(use=("pretrain", "measure")), "pre-training"),
        (_row(licence_confirmed=False), "data licence is confirmed"),
        (_row(url="http://example.org/x.zip"), "https"),
        (_row(attribution=" "), "attribution"),
        (_row(labels_url="http://example.org/l.zip", labels_size_bytes=1), "labels_url"),
        (_row(labels_url="https://example.org/l.zip"), "labels_url"),
    ],
)
def test_licence_rule_refuses(row: Dataset, why: str) -> None:
    assert any(why in p for p in licence_problems(row))


def test_non_commercial_set_may_be_measured_locally() -> None:
    row = _row(licence="CC-BY-NC-4.0", use=("measure",), local_only=True)
    assert licence_problems(row) == []


@pytest.mark.parametrize("licence", ["GPL-3.0-only", "NONE"])
def test_gpl_or_unlicensed_set_may_only_be_measured_locally_unpublished(licence: str) -> None:
    """The owner's decision of 2026-11-01: measured on this computer, nothing more."""
    row = _row(licence=licence, use=("measure",), local_only=True, publish_results=False)
    assert row.no_grant and licence_problems(row) == []
    for use in (("train",), ("train", "measure"), ("pretrain",)):
        assert licence_problems(dataclasses.replace(row, use=use))  # type: ignore[arg-type]


def test_register_with_a_bad_row_is_refused(tmp_path: Path) -> None:
    bad = tmp_path / "datasets.toml"
    bad.write_text(
        '[[dataset]]\nname = "v"\ntitle = "v"\nlicence = "CC-BY-NC-SA-3.0"\n'
        'attribution = "a"\nlayout = "wisard_vis"\nuse = ["train"]\n'
        'url = "https://example.org/v.zip"\npublish_weights = false\nlocal_only = false\n',
        encoding="utf-8",
    )
    with pytest.raises(BenchError, match="licensregeln"):
        load_register(bad)


@pytest.mark.parametrize(
    ("line", "key"),
    [
        ('publish_weights = "false"', "publish_weights"),
        ("local_only = 0", "local_only"),
        ('publish_results = "false"', "publish_results"),
        ('licence_confirmed = "yes"', "licence_confirmed"),
    ],
)
def test_register_flag_must_be_a_toml_boolean(tmp_path: Path, line: str, key: str) -> None:
    """``bool("false")`` is True: a quoted flag would publish what it means to keep."""
    flags = {"publish_weights": "false", "local_only": "false"}
    flags[key] = line.split(" = ", 1)[1]
    bad = tmp_path / "datasets.toml"
    bad.write_text(
        '[[dataset]]\nname = "v"\ntitle = "v"\nlicence = "CC-BY-4.0"\n'
        'attribution = "a"\nlayout = "wisard_vis"\nuse = ["measure"]\n'
        'url = "https://example.org/v.zip"\n' + "".join(f"{k} = {v}\n" for k, v in flags.items()),
        encoding="utf-8",
    )
    with pytest.raises(BenchError, match=f"{key} i v ska vara true eller false"):
        load_register(bad)
