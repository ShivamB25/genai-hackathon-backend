"""AI Service Exceptions for AI-Powered Trip Planner Backend.

This module provides custom exceptions for AI service operations including
Vertex AI model interactions, token management, and agent communication.
"""

from typing import Any


class AIServiceError(Exception):
    """Base exception for AI service operations."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        self.message = message
        self.details = details or {}
        super().__init__(message)


class ModelInitializationError(AIServiceError):
    """Raised when AI model initialization fails."""


class ModelConfigurationError(AIServiceError):
    """Raised when AI model configuration is invalid."""


class ModelConnectionError(AIServiceError):
    """Raised when connection to AI model service fails."""


class TokenLimitExceededError(AIServiceError):
    """Raised when token limit is exceeded for AI model."""

    def __init__(
        self,
        message: str,
        used_tokens: int | None = None,
        max_tokens: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.used_tokens = used_tokens
        self.max_tokens = max_tokens
        super().__init__(message, details)


class ContextOverflowError(AIServiceError):
    """Raised when context window overflows."""

    def __init__(
        self,
        message: str,
        context_length: int | None = None,
        max_context_length: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.context_length = context_length
        self.max_context_length = max_context_length
        super().__init__(message, details)


class RateLimitExceededError(AIServiceError):
    """Raised when API rate limit is exceeded."""

    def __init__(
        self,
        message: str,
        retry_after: int | None = None,
        quota_type: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.retry_after = retry_after
        self.quota_type = quota_type
        super().__init__(message, details)


class QuotaExceededError(AIServiceError):
    """Raised when service quota is exceeded."""

    def __init__(
        self,
        message: str,
        quota_type: str | None = None,
        reset_time: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.quota_type = quota_type
        self.reset_time = reset_time
        super().__init__(message, details)


class AuthenticationError(AIServiceError):
    """Raised when AI service authentication fails."""


class AuthorizationError(AIServiceError):
    """Raised when AI service authorization fails."""


class ServiceUnavailableError(AIServiceError):
    """Raised when AI service is temporarily unavailable."""

    def __init__(
        self,
        message: str,
        retry_after: int | None = None,
        service_status: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.retry_after = retry_after
        self.service_status = service_status
        super().__init__(message, details)


class InvalidPromptError(AIServiceError):
    """Raised when prompt is invalid or malformed."""


class PromptTemplateError(AIServiceError):
    """Raised when prompt template processing fails."""


class FunctionCallError(AIServiceError):
    """Raised when function call execution fails."""

    def __init__(
        self,
        message: str,
        function_name: str | None = None,
        function_args: dict[str, Any] | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.function_name = function_name
        self.function_args = function_args or {}
        super().__init__(message, details)


class FunctionToolError(AIServiceError):
    """Raised when function tool registration or execution fails."""


class AgentError(AIServiceError):
    """Raised when AI agent operation fails."""

    def __init__(
        self,
        message: str,
        agent_id: str | None = None,
        agent_state: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.agent_id = agent_id
        self.agent_state = agent_state
        super().__init__(message, details)


class AgentCommunicationError(AgentError):
    """Raised when agent communication fails."""


class AgentStateError(AgentError):
    """Raised when agent state is invalid."""


class SessionError(AIServiceError):
    """Raised when session management fails."""

    def __init__(
        self,
        message: str,
        session_id: str | None = None,
        session_state: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.session_id = session_id
        self.session_state = session_state
        super().__init__(message, details)


class SessionExpiredError(SessionError):
    """Raised when session has expired."""


class SessionNotFoundError(SessionError):
    """Raised when session is not found."""


class SessionStorageError(SessionError):
    """Raised when session storage operation fails."""


class VertexAIError(AIServiceError):
    """Raised when Vertex AI specific operation fails."""


class GeminiError(AIServiceError):
    """Raised when Gemini model specific operation fails."""


class ModelResponseError(AIServiceError):
    """Raised when AI model response is invalid or malformed."""

    def __init__(
        self,
        message: str,
        response_data: Any = None,
        expected_format: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.response_data = response_data
        self.expected_format = expected_format
        super().__init__(message, details)


class SafetyFilterError(AIServiceError):
    """Raised when AI model response is filtered by safety systems."""

    def __init__(
        self,
        message: str,
        filter_reason: str | None = None,
        safety_ratings: dict[str, Any] | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.filter_reason = filter_reason
        self.safety_ratings = safety_ratings or {}
        super().__init__(message, details)


class ModelTimeoutError(AIServiceError):
    """Raised when AI model request times out."""

    def __init__(
        self,
        message: str,
        timeout_duration: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.timeout_duration = timeout_duration
        super().__init__(message, details)


class RetryExhaustedError(AIServiceError):
    """Raised when retry attempts are exhausted."""

    def __init__(
        self,
        message: str,
        max_retries: int | None = None,
        last_error: Exception | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.max_retries = max_retries
        self.last_error = last_error
        super().__init__(message, details)
