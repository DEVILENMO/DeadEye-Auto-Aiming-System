# coding: utf-8
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class ConfigField:
    key: str
    label: str
    field_type: str  # 'bool' | 'int' | 'float' | 'str' | 'path' | 'choice'
    default: Any
    min_value: float | int | None = None
    max_value: float | int | None = None
    step: float | int | None = None
    choices: list[str] | None = None
    path_filetypes: list[tuple[str, str]] | None = None


class ConfigurableMixin:
    """
    子类通过 register_* 声明配置项，GUI 可根据 get_config_spec 自动生成表单。
    """

    _config_declared: bool = False
    _config_spec: dict[str, ConfigField] = {}

    @classmethod
    def declare_config(cls) -> None:
        """子类覆盖：调用 register_* 声明配置项。"""
        return

    @classmethod
    def _ensure_declared(cls) -> None:
        if getattr(cls, "_config_declared", False):
            return
        cls._config_spec = {}
        cls.declare_config()
        cls._config_declared = True

    @classmethod
    def get_config_spec(cls) -> list[ConfigField]:
        cls._ensure_declared()
        return list(cls._config_spec.values())

    @classmethod
    def register_field(
        cls,
        key: str,
        default: Any,
        *,
        label: str | None = None,
        field_type: str,
        min_value: float | int | None = None,
        max_value: float | int | None = None,
        step: float | int | None = None,
        choices: list[str] | None = None,
        path_filetypes: list[tuple[str, str]] | None = None,
    ) -> None:
        cls._config_spec[key] = ConfigField(
            key=key,
            label=label or key,
            field_type=field_type,
            default=default,
            min_value=min_value,
            max_value=max_value,
            step=step,
            choices=choices,
            path_filetypes=path_filetypes,
        )

    @classmethod
    def register_bool(cls, key: str, default: bool, *, label: str | None = None) -> None:
        cls.register_field(key, bool(default), label=label, field_type="bool")

    @classmethod
    def register_int(
        cls,
        key: str,
        default: int,
        *,
        label: str | None = None,
        min_value: int | None = None,
        max_value: int | None = None,
        step: int | None = None,
    ) -> None:
        cls.register_field(
            key,
            int(default),
            label=label,
            field_type="int",
            min_value=min_value,
            max_value=max_value,
            step=step,
        )

    @classmethod
    def register_float(
        cls,
        key: str,
        default: float,
        *,
        label: str | None = None,
        min_value: float | None = None,
        max_value: float | None = None,
        step: float | None = None,
    ) -> None:
        cls.register_field(
            key,
            float(default),
            label=label,
            field_type="float",
            min_value=min_value,
            max_value=max_value,
            step=step,
        )

    @classmethod
    def register_str(cls, key: str, default: str, *, label: str | None = None) -> None:
        cls.register_field(key, str(default), label=label, field_type="str")

    @classmethod
    def register_path(
        cls,
        key: str,
        default: str,
        *,
        label: str | None = None,
        filetypes: list[tuple[str, str]] | None = None,
    ) -> None:
        cls.register_field(key, str(default), label=label, field_type="path", path_filetypes=filetypes)

    @classmethod
    def register_choice(
        cls,
        key: str,
        default: str,
        *,
        label: str | None = None,
        choices: list[str],
    ) -> None:
        if default not in choices:
            default = choices[0]
        cls.register_field(key, str(default), label=label, field_type="choice", choices=choices)

    @classmethod
    def parse_config(cls, raw_config: dict[str, Any]) -> dict[str, Any]:
        cls._ensure_declared()
        parsed: dict[str, Any] = {}
        for field in cls._config_spec.values():
            raw_value = raw_config.get(field.key, field.default)
            parsed[field.key] = _coerce_value(field, raw_value)
        return parsed


def _coerce_value(field: ConfigField, raw_value: Any) -> Any:
    if field.field_type == "bool":
        if isinstance(raw_value, str):
            return raw_value.strip().lower() in ("1", "true", "yes", "y", "on")
        return bool(raw_value)
    if field.field_type == "int":
        return int(float(raw_value))
    if field.field_type == "float":
        return float(raw_value)
    if field.field_type in ("str", "path"):
        return str(raw_value)
    if field.field_type == "choice":
        value = str(raw_value)
        if field.choices and value not in field.choices:
            return field.default
        return value
    return raw_value

