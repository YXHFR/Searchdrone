"""tools/check_licenses.py: the license policy, without network access."""

from __future__ import annotations

import json

import check_licenses as cl
import pytest

P = cl.Package


@pytest.mark.parametrize(
    "license_text",
    [
        "MIT License",
        "MIT",
        "BSD-3-Clause",
        "BSD License",
        "Apache-2.0",
        "Apache Software License",
        "ISC License (ISCL)",
        "Python Software Foundation License",
        "PSF-2.0",
        "MPL-2.0",
        "Mozilla Public License 2.0 (MPL 2.0)",
        "LGPL-3.0",
        "LGPL-3.0-or-later",
        "GNU Lesser General Public License v3 (LGPLv3)",
        "MIT OR Apache-2.0",
        "GPL-3.0; MIT",
        "(MIT OR GPL-3.0) AND BSD-3-Clause",
        "Apache-2.0 WITH LLVM-exception",
        "GNU Library or Lesser General Public License (LGPL)",
    ],
)
def test_allowed(license_text: str) -> None:
    assert cl.license_allowed(license_text)


@pytest.mark.parametrize(
    "license_text",
    [
        "GPL-3.0",
        "GPL-2.0-or-later",
        "GNU General Public License v3 (GPLv3)",
        "GNU General Public License v2 or later (GPLv2+)",
        "AGPL-3.0",
        "GNU Affero General Public License v3",
        "SSPL-1.0",
        "Proprietary",
        "UNLIMITED",  # contains "MIT" only as part of a word
        "MIT AND GPL-3.0-only",
        "Apache-2.0 AND AGPL-3.0",
        "GPLv3+ (see BSD file)",
        "BSD-3-Clause AND (GPL-2.0 OR AGPL-3.0)",
        "GNU GENERAL PUBLIC LICENSE Version 3 ... GNU Lesser General Public License ... " * 4,
        "",
    ],
)
def test_rejected(license_text: str) -> None:
    assert not cl.license_allowed(license_text)


def test_lgpl_is_not_matched_as_gpl() -> None:
    assert cl.violations([P("pymavlink", "2.4.50", "LGPL-3.0")]) == []
    assert cl.violations([P("x", "1", "GPL-3.0")]) == [P("x", "1", "GPL-3.0")]


def test_unknown_fails_unless_known_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    pkg = P("odd", "1.0", "UNKNOWN")
    assert cl.violations([pkg]) == [pkg]
    monkeypatch.setitem(cl.KNOWN_OK, "odd", "MIT according to its LICENSE file, checked 2026-10-16")
    assert cl.violations([pkg]) == []


def test_tools_are_ignored() -> None:
    assert cl.violations([P("pip-licenses", "5.5.5", "MIT"), P("prettytable", "3", "BSD")]) == []


def test_parse_pip_licenses() -> None:
    text = json.dumps([{"Name": "pymavlink", "Version": "2.4.50", "License": "LGPL-3.0"}])
    assert cl.parse_pip_licenses(text) == [P("pymavlink", "2.4.50", "LGPL-3.0")]


def test_parse_pnpm_licenses() -> None:
    text = json.dumps(
        {
            "MIT": [{"name": "a", "versions": ["1.0.0", "1.1.0"], "license": "MIT"}],
            "AGPL-3.0": [{"name": "b", "versions": ["2.0.0"], "license": "AGPL-3.0"}],
        }
    )
    packages = cl.parse_pnpm_licenses(text)
    assert P("a", "1.1.0", "MIT") in packages
    assert cl.violations(packages) == [P("b", "2.0.0", "AGPL-3.0")]


def test_parse_pnpm_without_packages() -> None:
    assert cl.parse_pnpm_licenses("No licenses in packages found") == []
