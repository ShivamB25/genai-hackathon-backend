"""Vertex AI Model Configuration for AI-Powered Trip Planner Backend.

This module provides Vertex AI client initialization, Gemini model configuration,
and async model interaction with comprehensive error handling and retry logic.
"""

import asyncio
import os
from functools import lru_cache
from typing import Any

import vertexai
from google.auth.exceptions import DefaultCredentialsError
from google.cloud import aiplatform
from vertexai.generative_models import (
    GenerationConfig,
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
    SafetySetting,
)

from src.ai_services.exceptions import (
    AuthenticationError,
    ModelConfigurationError,
    ModelConnectionError,
    ModelInitializationError,
    VertexAIError,
)
from src.core.config import settings
from src.core.logging import get_logger

logger = get_logger(__name__)

# Global Vertex AI client instance
_vertex_ai_initialized: bool = False
_generative_model: GenerativeModel | None = None


class VertexAIModelConfig:
    """Vertex AI model configuration and management."""

    def __init__(
        self,
        project_id: str | None = None,
        location: str | None = None,
        model_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize Vertex AI model configuration.

        Args:
            project_id: Google Cloud project ID
            location: Vertex AI region
            model_name: Gemini model name
            **kwargs: Additional model parameters
        """
        self.project_id = (
            project_id or settings.vertex_ai_project_id or settings.google_cloud_project
        )
        self.location = location or settings.vertex_ai_region
        self.model_name = model_name or settings.gemini_model

        # Model generation parameters
        self.temperature = kwargs.get("temperature", settings.gemini_temperature)
        self.max_output_tokens = kwargs.get(
            "max_output_tokens", settings.gemini_max_tokens
        )
        self.top_p = kwargs.get("top_p", settings.gemini_top_p)
        self.top_k = kwargs.get("top_k", settings.gemini_top_k)

        # Validate configuration
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate model configuration parameters."""
        if not self.project_id:
            msg = "Project ID is required for Vertex AI initialization"
            raise ModelConfigurationError(msg)

        if not self.location:
            msg = "Location is required for Vertex AI initialization"
            raise ModelConfigurationError(msg)

        if not self.model_name:
            msg = "Model name is required for Vertex AI initialization"
            raise ModelConfigurationError(msg)

        # Validate temperature range
        MAX_TEMPERATURE = 2.0
        if not 0.0 <= self.temperature <= MAX_TEMPERATURE:
            msg = f"Temperature must be between 0.0 and {MAX_TEMPERATURE}, got {self.temperature}"
            raise ModelConfigurationError(msg)

        # Validate top_p range
        MAX_TOP_P = 1.0
        if not 0.0 <= self.top_p <= MAX_TOP_P:
            msg = f"top_p must be between 0.0 and {MAX_TOP_P}, got {self.top_p}"
            raise ModelConfigurationError(msg)

        # Validate top_k range
        MIN_TOP_K = 1
        MAX_TOP_K = 100
        if not MIN_TOP_K <= self.top_k <= MAX_TOP_K:
            msg = f"top_k must be between {MIN_TOP_K} and {MAX_TOP_K}, got {self.top_k}"
            raise ModelConfigurationError(msg)

        # Validate max_output_tokens
        MIN_MAX_OUTPUT_TOKENS = 1
        MAX_MAX_OUTPUT_TOKENS = 32768
        if not MIN_MAX_OUTPUT_TOKENS <= self.max_output_tokens <= MAX_MAX_OUTPUT_TOKENS:
            msg = f"max_output_tokens must be between {MIN_MAX_OUTPUT_TOKENS} and {MAX_MAX_OUTPUT_TOKENS}, got {self.max_output_tokens}"
            raise ModelConfigurationError(msg)

    def get_generation_config(self) -> GenerationConfig:
        """Get model generation configuration.

        Returns:
            GenerationConfig: Vertex AI generation configuration
        """
        return GenerationConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            top_p=self.top_p,
            top_k=self.top_k,
        )

    def get_safety_settings(self) -> list[SafetySetting]:
        """Get model safety settings.

        Returns:
            List[SafetySetting]: Vertex AI safety settings
        """
        return [
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            ),
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            ),
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            ),
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            ),
        ]

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        return {
            "project_id": self.project_id,
            "location": self.location,
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "top_p": self.top_p,
            "top_k": self.top_k,
        }


def initialize_vertex_ai(config: VertexAIModelConfig | None = None) -> None:
    """Initialize Vertex AI with proper authentication.

    Args:
        config: Model configuration instance

    Raises:
        AuthenticationError: If authentication fails
        ModelInitializationError: If initialization fails
    """
    global _vertex_ai_initialized

    if _vertex_ai_initialized:
        return

    try:
        if not config:
            config = get_model_config()

        # Set up authentication
        credentials_path = None

        # Check for service account JSON file
        if os.path.exists("the-sandbox-460908-i8-5cdb12311a99.json"):
            credentials_path = "the-sandbox-460908-i8-5cdb12311a99.json"
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
        elif settings.google_application_credentials and os.path.exists(
            settings.google_application_credentials
        ):
            credentials_path = settings.google_application_credentials
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

        # Initialize Vertex AI
        if credentials_path:
            logger.info(
                "Initializing Vertex AI with service account credentials",
                project_id=config.project_id,
                location=config.location,
                credentials_path=credentials_path,
            )
        else:
            logger.info(
                "Initializing Vertex AI with default credentials",
                project_id=config.project_id,
                location=config.location,
            )

        # Initialize Vertex AI
        vertexai.init(
            project=config.project_id,
            location=config.location,
        )

        # Initialize AI Platform
        aiplatform.init(
            project=config.project_id,
            location=config.location,
        )

        _vertex_ai_initialized = True

        logger.info(
            "Vertex AI initialized successfully",
            project_id=config.project_id,
            location=config.location,
        )

    except DefaultCredentialsError as e:
        error_msg = f"Failed to authenticate with Google Cloud: {e!s}"
        logger.error(error_msg, exc_info=True)
        raise AuthenticationError(error_msg) from e

    except Exception as e:
        error_msg = f"Failed to initialize Vertex AI: {e!s}"
        logger.error(error_msg, exc_info=True)
        raise ModelInitializationError(error_msg) from e


def get_generative_model(config: VertexAIModelConfig | None = None) -> GenerativeModel:
    """Get or create Vertex AI generative model instance.

    Args:
        config: Model configuration instance

    Returns:
        GenerativeModel: Vertex AI generative model instance

    Raises:
        ModelInitializationError: If model initialization fails
    """
    global _generative_model

    if _generative_model is not None:
        return _generative_model

    try:
        if not config:
            config = get_model_config()

        # Ensure Vertex AI is initialized
        initialize_vertex_ai(config)

        # Create generative model
        _generative_model = GenerativeModel(
            model_name=config.model_name,
            generation_config=config.get_generation_config(),
            safety_settings=config.get_safety_settings(),
        )

        logger.info(
            "Generative model initialized successfully",
            model_name=config.model_name,
        )

        return _generative_model

    except Exception as e:
        error_msg = f"Failed to initialize generative model: {e!s}"
        logger.error(error_msg, exc_info=True)
        raise ModelInitializationError(error_msg) from e


class AsyncVertexAIClient:
    """Async wrapper for Vertex AI generative model with error handling."""

    def __init__(self, config: VertexAIModelConfig | None = None) -> None:
        """Initialize async Vertex AI client.

        Args:
            config: Model configuration instance
        """
        self.config = config or get_model_config()
        self._model: GenerativeModel | None = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the async client."""
        if self._initialized:
            return

        try:
            # Initialize in thread pool to avoid blocking
            await asyncio.to_thread(initialize_vertex_ai, self.config)
            self._model = await asyncio.to_thread(get_generative_model, self.config)
            self._initialized = True

            logger.debug("AsyncVertexAIClient initialized successfully")

        except Exception as e:
            error_msg = f"Failed to initialize AsyncVertexAIClient: {e!s}"
            logger.error(error_msg, exc_info=True)
            raise ModelInitializationError(error_msg) from e

    async def generate_content(self, prompt: str, **kwargs) -> str:
        """Generate content using the model.

        Args:
            prompt: Input prompt for generation
            **kwargs: Additional generation parameters

        Returns:
            str: Generated content

        Raises:
            ModelConnectionError: If model interaction fails
            VertexAIError: If Vertex AI specific error occurs
        """
        await self.initialize()

        if not self._model:
            msg = "Model not initialized"
            raise ModelConnectionError(msg)

        try:
            # Generate content in thread pool
            response = await asyncio.to_thread(
                self._model.generate_content, prompt, **kwargs
            )

            if not response.text:
                msg = "Empty response from model"
                raise VertexAIError(msg)

            logger.debug(
                "Content generated successfully",
                prompt_length=len(prompt),
                response_length=len(response.text),
            )

            return response.text

        except Exception as e:
            error_msg = f"Failed to generate content: {e!s}"
            logger.error(
                error_msg,
                exc_info=True,
                prompt_length=len(prompt) if prompt else 0,
            )
            raise VertexAIError(error_msg) from e

    async def generate_content_stream(self, prompt: str, **kwargs) -> Any:
        """Generate content using streaming.

        Args:
            prompt: Input prompt for generation
            **kwargs: Additional generation parameters

        Returns:
            Generator: Streaming response generator

        Raises:
            ModelConnectionError: If model interaction fails
            VertexAIError: If Vertex AI specific error occurs
        """
        await self.initialize()

        if not self._model:
            msg = "Model not initialized"
            raise ModelConnectionError(msg)

        try:
            # Generate streaming content
            stream = await asyncio.to_thread(
                self._model.generate_content, prompt, stream=True, **kwargs
            )

            logger.debug(
                "Streaming content generation started",
                prompt_length=len(prompt),
            )

            return stream

        except Exception as e:
            error_msg = f"Failed to generate streaming content: {e!s}"
            logger.error(
                error_msg,
                exc_info=True,
                prompt_length=len(prompt) if prompt else 0,
            )
            raise VertexAIError(error_msg) from e

    async def health_check(self) -> bool:
        """Perform health check on the model.

        Returns:
            bool: True if model is healthy, False otherwise
        """
        try:
            await self.initialize()

            # Simple test generation
            test_response = await self.generate_content("Hello")

            is_healthy = bool(test_response and len(test_response.strip()) > 0)

            logger.debug("Model health check completed", is_healthy=is_healthy)

            return is_healthy

        except Exception as e:
            logger.warning(
                "Model health check failed",
                error=str(e),
                exc_info=True,
            )
            return False


@lru_cache
def get_model_config() -> VertexAIModelConfig:
    """Get cached model configuration.

    Returns:
        VertexAIModelConfig: Model configuration instance
    """
    return VertexAIModelConfig()


@lru_cache
def get_async_client() -> AsyncVertexAIClient:
    """Get cached async Vertex AI client.

    Returns:
        AsyncVertexAIClient: Async client instance
    """
    return AsyncVertexAIClient()


# Initialize on module import if not in test mode
if not settings.enable_test_mode and not settings.mock_external_apis:
    try:
        # Initialize Vertex AI in background
        logger.info("Initializing Vertex AI on module import")
        initialize_vertex_ai()
    except Exception as e:
        logger.warning("Failed to initialize Vertex AI on import", error=str(e))
