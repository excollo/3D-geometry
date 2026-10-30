"""Smoke tests for the checker package and its configuration."""

import boxcheck

from boxcheck.settings import load_settings


def test_boxcheck_package_can_be_imported() -> None:
    """The checker package should be available to Python."""
    assert boxcheck is not None


def test_default_configuration_can_be_loaded() -> None:
    """Both configuration files should load through one public function."""
    settings = load_settings()

    assert "thresholds" in settings
    assert "phones" in settings


def test_default_configuration_contains_essential_values() -> None:
    """The initial configuration should contain values needed by the checker."""
    settings = load_settings()
    thresholds = settings["thresholds"]
    phones = settings["phones"]

    assert thresholds["video"]["maximum_duration_seconds"] > 0
    assert thresholds["features"]["detector"] in {"sift", "akaze", "orb"}
    assert thresholds["features"]["minimum_matches_per_pair"] > 0
    assert phones["default"]["horizontal_field_of_view_degrees"] > 0
    assert isinstance(phones["phones"], dict)
