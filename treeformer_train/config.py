from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from omegaconf import DictConfig, OmegaConf


class AttrDict:
    """Small recursive namespace that preserves the legacy ``config.TRAIN.LR`` access style."""

    def __init__(self, values: Mapping[str, Any]) -> None:
        for key, value in values.items():
            if isinstance(value, Mapping):
                value = AttrDict(value)
            elif isinstance(value, list):
                value = [AttrDict(item) if isinstance(item, Mapping) else item for item in value]
            setattr(self, key, value)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in self.__dict__.items():
            if isinstance(value, AttrDict):
                result[key] = value.to_dict()
            elif isinstance(value, list):
                result[key] = [item.to_dict() if isinstance(item, AttrDict) else item for item in value]
            else:
                result[key] = value
        return result

    def __repr__(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)


def as_plain_container(config: DictConfig | Mapping[str, Any]) -> dict[str, Any]:
    """Convert an OmegaConf or mapping object to a plain Python dictionary."""

    if isinstance(config, DictConfig):
        return OmegaConf.to_container(config, resolve=True)  # type: ignore[return-value]
    return dict(config)


def load_legacy_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"legacy YAML must contain a mapping at top level: {path}")
    return data


def deep_update(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively update ``base`` in place and return it."""

    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            deep_update(base[key], value)  # type: ignore[index]
        else:
            base[key] = value
    return base


def make_legacy_config(config: DictConfig | Mapping[str, Any]) -> AttrDict:
    """Build the legacy TreeFormer config namespace from a Hydra config.

    The existing model/loss/dataset code expects top-level ``DATA``, ``MODEL``,
    ``TRAIN`` and ``log`` attributes.  This function either loads a legacy YAML
    file and overlays Hydra values, or directly converts the Hydra fields when
    no legacy file is specified.
    """

    plain = as_plain_container(config)
    legacy_path = plain.get("legacy_config_path")
    if legacy_path:
        legacy = load_legacy_yaml(legacy_path)
    else:
        legacy = {}

    for key in ("DATA", "MODEL", "TRAIN", "log"):
        section = plain.get(key)
        if isinstance(section, Mapping):
            existing = legacy.get(key, {})
            if not isinstance(existing, dict):
                existing = {}
            legacy[key] = deep_update(existing, section)

    train_section = legacy.get("TRAIN", {})
    data_section = legacy.get("DATA", {})
    if isinstance(train_section, Mapping) and isinstance(data_section, dict):
        for key in ("AUX_DETAIL_THRESHOLD", "AUX_DETAIL_SCALES", "AUX_DETAIL_SUPPORT_KERNEL_SIZE"):
            if key in train_section and key not in data_section:
                data_section[key] = train_section[key]

    missing = [key for key in ("DATA", "MODEL", "TRAIN", "log") if key not in legacy]
    if missing:
        raise ValueError(f"Hydra config cannot be converted to legacy config; missing sections: {missing}")

    return AttrDict(legacy)
