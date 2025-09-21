import pytest
from vertexai.generative_models import GenerationConfig, SafetySetting

from src.ai_services.model_config import ModelConfigurationError, VertexAIModelConfig


def test_model_config_initialization():
    """Test that the VertexAIModelConfig initializes with default values."""
    config = VertexAIModelConfig(
        project_id="test-project", location="us-central1", model_name="gemini-pro"
    )

    assert config.project_id == "test-project"
    assert config.temperature > 0


def test_model_config_validation():
    """Test that the model configuration validation works."""
    with pytest.raises(ModelConfigurationError):
        # Invalid temperature
        VertexAIModelConfig(
            project_id="p", location="l", model_name="m", temperature=3.0
        )

    with pytest.raises(ModelConfigurationError):
        # Missing project_id
        VertexAIModelConfig(location="l", model_name="m")


def test_get_generation_config():
    """Test the generation of GenerationConfig."""
    config = VertexAIModelConfig(
        project_id="p",
        location="l",
        model_name="m",
        temperature=0.5,
        max_output_tokens=100,
    )
    gen_config = config.get_generation_config()

    assert isinstance(gen_config, GenerationConfig)
    assert getattr(gen_config, "temperature", None) == pytest.approx(config.temperature)
    assert getattr(gen_config, "max_output_tokens", None) == config.max_output_tokens


def test_get_safety_settings():
    """Test the generation of SafetySettings."""
    config = VertexAIModelConfig(project_id="p", location="l", model_name="m")
    safety_settings = config.get_safety_settings()

    assert isinstance(safety_settings, list)
    assert len(safety_settings) > 0
    assert all(isinstance(setting, SafetySetting) for setting in safety_settings)
    assert all(
        getattr(setting, "category", None) is not None for setting in safety_settings
    )
