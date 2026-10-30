"""Load and validate configuration for the 3D box checker."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


CONFIG_DIRECTORY = Path(__file__).resolve().parent.parent / "config"
DEFAULT_THRESHOLDS_PATH = CONFIG_DIRECTORY / "thresholds.yaml"
DEFAULT_PHONES_PATH = CONFIG_DIRECTORY / "phones.yaml"

REQUIRED_THRESHOLD_SECTIONS = {
    "video",
    "preprocessing",
    "quality",
    "roi",
    "features",
    "pairing",
    "geometry",
    "scoring",
    "debug",
}


class SettingsError(ValueError):
    """Raised when a configuration file is missing or invalid."""


def _load_yaml(path: Path) -> dict[str, Any]:
    """Read one YAML file and require a mapping at its top level."""
    if not path.is_file():
        raise SettingsError(f"Configuration file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
    except yaml.YAMLError as error:
        raise SettingsError(f"Invalid YAML in configuration file {path}: {error}") from error
    except OSError as error:
        raise SettingsError(f"Could not read configuration file {path}: {error}") from error

    if not isinstance(data, dict):
        raise SettingsError(f"Configuration file must contain a YAML mapping: {path}")

    return data


def load_thresholds(path: str | Path | None = None) -> dict[str, Any]:
    """Load checker thresholds and verify all required sections exist."""
    config_path = Path(path) if path is not None else DEFAULT_THRESHOLDS_PATH
    thresholds = _load_yaml(config_path)

    missing_sections = REQUIRED_THRESHOLD_SECTIONS - thresholds.keys()
    if missing_sections:
        missing = ", ".join(sorted(missing_sections))
        raise SettingsError(
            f"Threshold configuration is missing required sections: {missing}"
        )

    return thresholds


def load_phones(path: str | Path | None = None) -> dict[str, Any]:
    """Load default and phone-specific camera information."""
    config_path = Path(path) if path is not None else DEFAULT_PHONES_PATH
    phones = _load_yaml(config_path)

    default = phones.get("default")
    if not isinstance(default, dict) or "horizontal_field_of_view_degrees" not in default:
        raise SettingsError(
            "Phone configuration must define "
            "default.horizontal_field_of_view_degrees"
        )

    if not isinstance(phones.get("phones"), dict):
        raise SettingsError("Phone configuration must define 'phones' as a mapping")

    return phones


def load_settings(
    thresholds_path: str | Path | None = None,
    phones_path: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Load all checker configuration in one call."""
    return {
        "thresholds": load_thresholds(thresholds_path),
        "phones": load_phones(phones_path),
    }
