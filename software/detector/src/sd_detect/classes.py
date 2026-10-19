"""The class register: which kinds of thing a detector may report, and who reports them.

The register is a table (``classes.toml`` next to this file), one row per class: the id
that becomes the detection's label, a Swedish name, whether it is a search target or
context, the source that reports it, the plausible size on the ground and the source's
settings. A new class needs only a new row and a source (``sd_detect.sources``); nothing
else in the code names a class.

Nothing here identifies anyone: a row may not describe a face, a licence plate or any
other identity, and the loader refuses such ids.
"""

from __future__ import annotations

import re
import tomllib
from collections.abc import Iterable, Iterator
from functools import cache
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import Field, model_validator

from sd_schemas.base import SdModel

__all__ = [
    "CLASS_ID_PATTERN",
    "DEFAULT_REGISTER",
    "ClassRegistry",
    "ClassSpec",
    "RegistryError",
    "default_registry",
    "is_forbidden_class",
]

CLASS_ID_PATTERN = r"^[a-z][a-z0-9_]{1,31}$"
DEFAULT_REGISTER = Path(__file__).with_name("classes.toml")
_FORBIDDEN_ID_WORDS = re.compile(
    r"face|ansikte|identity|identitet|biometric|iris|fingerprint|plate|regnr|person_id"
)

Kind = Literal["target", "context"]


def is_forbidden_class(text: str) -> bool:
    """Whether a class id or name speaks of a face, a licence plate or another identity."""
    return _FORBIDDEN_ID_WORDS.search(text.lower()) is not None


class RegistryError(ValueError):
    """The register file is missing, malformed or breaks a rule."""


class ClassSpec(SdModel):
    """One row of the register."""

    id: str = Field(pattern=CLASS_ID_PATTERN)
    name_sv: str = Field(min_length=1, max_length=40)
    kind: Kind
    source: str = Field(pattern=r"^[a-z][a-z0-9_]{1,31}$")
    size_m: tuple[float, float] | None = None
    params: dict[str, Any] = Field(default_factory=dict[str, Any])

    @model_validator(mode="after")
    def _checks(self) -> Self:
        if is_forbidden_class(self.id) or is_forbidden_class(self.name_sv):
            raise ValueError(f"class {self.id!r}: Searchdrone never recognises identities")
        if self.size_m is not None and not 0 < self.size_m[0] <= self.size_m[1]:
            raise ValueError(f"class {self.id!r}: size_m must be 0 < min <= max")
        return self

    def fits_size(self, longest_side_m: float) -> bool:
        """Whether a thing this long on the ground can be this class (no limit: yes)."""
        if self.size_m is None:
            return True
        low, high = self.size_m
        return low <= longest_side_m <= high


class ClassRegistry:
    """The classes in register order, looked up by id."""

    def __init__(self, classes: Iterable[ClassSpec]) -> None:
        self._by_id: dict[str, ClassSpec] = {}
        for spec in classes:
            if spec.id in self._by_id:
                raise RegistryError(f"class {spec.id!r} appears twice")
            self._by_id[spec.id] = spec
        if not self._by_id:
            raise RegistryError("the register has no classes")

    @classmethod
    def from_toml(cls, text: str) -> Self:
        try:
            data = tomllib.loads(text)
        except tomllib.TOMLDecodeError as exc:
            raise RegistryError(f"the register is not valid TOML: {exc}") from exc
        unknown = set(data) - {"class"}
        if unknown:
            raise RegistryError(f"unknown top-level keys: {sorted(unknown)}")
        rows = data.get("class", [])
        try:
            return cls(ClassSpec.model_validate(row) for row in rows)
        except ValueError as exc:
            raise RegistryError(str(exc)) from exc

    @classmethod
    def load(cls, path: Path = DEFAULT_REGISTER) -> Self:
        try:
            return cls.from_toml(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise RegistryError(f"cannot read the register {path.name}: {exc}") from exc

    def __contains__(self, class_id: object) -> bool:
        return class_id in self._by_id

    def __iter__(self) -> Iterator[ClassSpec]:
        return iter(self._by_id.values())

    def __len__(self) -> int:
        return len(self._by_id)

    def get(self, class_id: str) -> ClassSpec:
        try:
            return self._by_id[class_id]
        except KeyError:
            raise RegistryError(f"no class {class_id!r} in the register") from None

    def ids(self) -> tuple[str, ...]:
        return tuple(self._by_id)

    def by_source(self, source: str) -> list[ClassSpec]:
        return [spec for spec in self if spec.source == source]

    def by_gz_label(self) -> dict[int, ClassSpec]:
        """The simulator's label number of each class that has one."""
        out: dict[int, ClassSpec] = {}
        for spec in self:
            label = spec.params.get("gz_label")
            if label is None:
                continue
            if not isinstance(label, int) or isinstance(label, bool) or label <= 0:
                raise RegistryError(f"class {spec.id!r}: gz_label must be a positive integer")
            if label in out:
                raise RegistryError(
                    f"gz_label {label} is used by {out[label].id!r} and {spec.id!r}"
                )
            out[label] = spec
        return out


@cache
def default_registry() -> ClassRegistry:
    """The register shipped with the package (``classes.toml``)."""
    return ClassRegistry.load()
