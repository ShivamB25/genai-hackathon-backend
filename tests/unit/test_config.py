import importlib
import os

import pytest
from pydantic import ValidationError

# Import the module that contains the settings
from src.core import config


@pytest.fixture(autouse=True)
def reload_settings():
    """Fixture to reload settings before each test."""
    importlib.reload(config)
    yield
    importlib.reload(config)


def test_settings_load_from_env():
    """Test that settings are correctly loaded from environment variables."""
    os.environ["ENVIRONMENT"] = "testing"
    os.environ["APP_NAME"] = "Test App"
    os.environ["FIREBASE_PROJECT_ID"] = "test-project"
    os.environ["GOOGLE_MAPS_API_KEY"] = "test-maps-key"

    importlib.reload(config)
    settings = config.get_settings()

    assert settings.environment == "testing"
    assert settings.app_name == "Test App"
    assert settings.firebase_project_id == "test-project"
    assert settings.google_maps_api_key == "test-maps-key"

    # Clean up environment variables
    os.environ.pop("ENVIRONMENT")
    os.environ.pop("APP_NAME")
    os.environ.pop("FIREBASE_PROJECT_ID")
    os.environ.pop("GOOGLE_MAPS_API_KEY")


def test_settings_default_values():
    """Test that settings have correct default values for development."""
    # Unset env to ensure defaults are used
    if "ENVIRONMENT" in os.environ:
        os.environ.pop("ENVIRONMENT")

    importlib.reload(config)
    settings = config.get_settings()

    assert settings.environment == "development"
    assert settings.app_version == "0.1.0"
    assert settings.database_timeout == 30


def test_settings_validation_error():
    """Test that settings raise a validation error for invalid data."""
    os.environ["GEMINI_TEMPERATURE"] = "invalid"

    importlib.reload(config)

    with pytest.raises(ValidationError):
        config.get_settings()

    # Unset to avoid affecting other tests
    os.environ.pop("GEMINI_TEMPERATURE")


def test_settings_for_production_env():
    """Test settings in a production environment."""
    os.environ["ENVIRONMENT"] = "production"

    importlib.reload(config)
    settings = config.get_settings()

    assert settings.environment == "production"
    assert settings.debug is False

    os.environ.pop("ENVIRONMENT")


def test_settings_for_development_env():
    """Test settings in a development environment."""
    os.environ["ENVIRONMENT"] = "development"

    importlib.reload(config)
    settings = config.get_settings()

    assert settings.environment == "development"
    assert settings.debug is True

    os.environ.pop("ENVIRONMENT")
