"""Configuration management for the AI-Powered Trip Planner Backend.

This module provides environment-based configuration management using Pydantic
BaseSettings with support for development and production environments.
"""

import os
from functools import lru_cache

from pydantic import Field, ValidationError, field_validator
from pydantic_settings import BaseSettings


class BaseConfig(BaseSettings):
    """Base configuration class with common settings."""

    # Application Configuration
    app_name: str = Field(default="genai-trip-planner")
    app_version: str = Field(default="0.1.0")
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")
    environment: str = Field(default="development")

    # FastAPI Configuration
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    workers: int = Field(default=1)

    # Google Cloud Configuration
    google_cloud_project: str | None = Field(default=None)
    google_cloud_region: str = Field(default="asia-south1")
    google_cloud_zone: str = Field(default="asia-south1-a")
    google_application_credentials: str | None = Field(default=None)
    google_credentials_json: str | None = Field(default=None)

    # Vertex AI Configuration
    vertex_ai_project_id: str | None = Field(default=None)
    vertex_ai_region: str = Field(default="asia-south1")
    vertex_ai_endpoint: str | None = Field(default=None)

    # Gemini Model Configuration
    gemini_model: str = Field(default="gemini-2.0-flash-exp")
    gemini_temperature: float = Field(default=0.7)
    gemini_max_tokens: int = Field(default=8192)
    gemini_top_p: float = Field(default=0.95)
    gemini_top_k: int = Field(default=40)

    # Firebase Configuration
    firebase_project_id: str | None = Field(default=None)
    firebase_private_key_id: str | None = Field(default=None)
    firebase_private_key: str | None = Field(default=None)
    firebase_client_email: str | None = Field(default=None)
    firebase_client_id: str | None = Field(default=None)
    firebase_auth_uri: str = Field(default="https://accounts.google.com/o/oauth2/auth")
    firebase_token_uri: str = Field(default="https://oauth2.googleapis.com/token")
    firebase_client_cert_url: str | None = Field(default=None)

    # Firebase Web API Configuration
    firebase_web_api_key: str | None = Field(default=None)
    firebase_auth_domain: str | None = Field(default=None)
    firebase_storage_bucket: str | None = Field(default=None)

    # Firestore Configuration
    firestore_database_id: str = Field(default="(default)")
    firestore_emulator_host: str | None = Field(default=None)

    # Google Maps API Configuration
    google_maps_api_key: str | None = Field(default=None)
    maps_places_api_enabled: bool = Field(default=True)
    maps_directions_api_enabled: bool = Field(default=True)
    maps_distance_matrix_api_enabled: bool = Field(default=True)
    maps_geocoding_api_enabled: bool = Field(default=True)
    maps_timezone_api_enabled: bool = Field(default=True)
    maps_api_rate_limit: int = Field(default=100)
    maps_api_quota_user: str | None = Field(default=None)
    ticketmaster_api_key: str | None = Field(default=None)

    # Maps API Rate Limiting and Performance
    maps_requests_per_second: float = Field(default=10.0)
    maps_burst_limit: int = Field(default=20)
    maps_request_timeout: int = Field(default=30)
    maps_max_retries: int = Field(default=3)
    maps_retry_backoff_factor: float = Field(default=2.0)
    maps_enable_connection_pooling: bool = Field(default=True)
    maps_max_connections: int = Field(default=100)
    maps_keepalive_connections: int = Field(default=20)

    # Maps API Caching Configuration
    maps_enable_caching: bool = Field(default=True)
    maps_cache_ttl: int = Field(default=3600)  # 1 hour
    maps_cache_max_size: int = Field(default=1000)
    maps_geocoding_cache_ttl: int = Field(default=86400)  # 24 hours for geocoding
    maps_places_cache_ttl: int = Field(default=1800)  # 30 minutes for places
    maps_directions_cache_ttl: int = Field(default=900)  # 15 minutes for directions

    # Maps Search Configuration
    maps_default_radius: int = Field(default=5000)  # meters
    maps_max_radius: int = Field(default=50000)  # Google Maps limit
    maps_default_language: str = Field(default="en")
    maps_default_region: str = Field(default="IN")  # India
    maps_supported_languages: list[str] = Field(
        default=["en", "hi", "te", "ta", "bn", "gu", "kn", "ml", "mr", "or", "pa", "ur"]
    )

    # Place Search Defaults
    maps_default_place_fields: list[str] = Field(
        default=[
            "place_id",
            "name",
            "formatted_address",
            "geometry",
            "rating",
            "user_ratings_total",
            "price_level",
            "opening_hours",
            "types",
            "business_status",
            "photos",
        ]
    )
    maps_max_search_results: int = Field(default=20)
    maps_min_rating_filter: float = Field(default=0.0)
    maps_enable_photo_references: bool = Field(default=True)
    maps_photo_max_width: int = Field(default=400)
    maps_photo_max_height: int = Field(default=400)

    # Directions Configuration
    maps_default_travel_mode: str = Field(default="driving")
    maps_supported_travel_modes: list[str] = Field(
        default=["driving", "walking", "bicycling", "transit"]
    )
    maps_enable_alternative_routes: bool = Field(default=True)
    maps_max_waypoints: int = Field(default=25)  # Google Maps limit
    maps_enable_traffic_model: bool = Field(default=True)
    maps_default_traffic_model: str = Field(default="best_guess")
    maps_default_units: str = Field(default="metric")
    maps_enable_route_optimization: bool = Field(default=True)

    # Geocoding Configuration
    maps_geocoding_result_types: list[str] = Field(
        default=[
            "street_address",
            "route",
            "intersection",
            "political",
            "country",
            "administrative_area_level_1",
            "administrative_area_level_2",
            "locality",
            "neighborhood",
            "premise",
            "subpremise",
            "postal_code",
        ]
    )
    maps_geocoding_location_types: list[str] = Field(
        default=["ROOFTOP", "RANGE_INTERPOLATED", "GEOMETRIC_CENTER", "APPROXIMATE"]
    )
    maps_enable_address_validation: bool = Field(default=True)
    maps_enable_batch_geocoding: bool = Field(default=True)
    maps_batch_geocoding_max_size: int = Field(default=100)

    # Maps Error Handling
    maps_enable_error_logging: bool = Field(default=True)
    maps_log_api_responses: bool = Field(default=False)  # Set to True for debugging
    maps_enable_fallback_geocoding: bool = Field(default=True)
    maps_enable_request_validation: bool = Field(default=True)

    # Maps Business Logic
    maps_popular_place_min_rating: float = Field(default=4.0)
    maps_popular_place_min_reviews: int = Field(default=10)
    maps_nearby_search_priority_types: list[str] = Field(
        default=[
            "tourist_attraction",
            "restaurant",
            "lodging",
            "museum",
            "park",
            "shopping_mall",
            "gas_station",
            "hospital",
        ]
    )

    # Maps Development and Testing
    maps_enable_mock_responses: bool = Field(default=False)
    maps_mock_response_delay: float = Field(default=0.1)
    maps_enable_request_logging: bool = Field(default=True)
    maps_enable_performance_monitoring: bool = Field(default=True)
    maps_enable_usage_analytics: bool = Field(default=True)

    # Maps Security and Privacy
    maps_sanitize_api_keys_in_logs: bool = Field(default=True)
    maps_enable_request_signing: bool = Field(default=False)
    maps_allowed_domains: list[str] = Field(default_factory=list)
    maps_enable_ip_restrictions: bool = Field(default=False)
    maps_allowed_ip_ranges: list[str] = Field(default_factory=list)

    # AI Service Configuration
    ai_model_provider: str = Field(default="vertex-ai")
    ai_fallback_provider: str = Field(default="google-ai")
    enable_function_calling: bool = Field(default=True)
    max_function_calls: int = Field(default=5)
    function_call_timeout: int = Field(default=30)
    system_prompt_version: str = Field(default="v1.0")
    enable_prompt_caching: bool = Field(default=True)
    prompt_cache_ttl: int = Field(default=3600)

    # Vertex AI Advanced Configuration
    vertex_ai_max_retries: int = Field(default=3)
    vertex_ai_retry_delay: float = Field(default=1.0)
    vertex_ai_request_timeout: int = Field(default=60)
    vertex_ai_health_check_interval: int = Field(default=300)
    vertex_ai_concurrent_requests: int = Field(default=10)

    # Gemini Safety Configuration
    gemini_safety_threshold: str = Field(default="BLOCK_MEDIUM_AND_ABOVE")
    gemini_enable_safety_filters: bool = Field(default=True)
    gemini_block_dangerous_content: bool = Field(default=True)
    gemini_block_harassment: bool = Field(default=True)
    gemini_block_hate_speech: bool = Field(default=True)
    gemini_block_sexually_explicit: bool = Field(default=True)

    # Token Management Configuration
    enable_token_tracking: bool = Field(default=True)
    max_tokens_per_session: int = Field(default=100000)
    max_tokens_per_hour: int = Field(default=500000)
    token_usage_warning_threshold: float = Field(default=0.8)
    enable_token_rate_limiting: bool = Field(default=True)

    # Agent Configuration
    max_active_agents: int = Field(default=50)
    agent_idle_timeout: int = Field(default=1800)  # 30 minutes
    agent_max_iterations: int = Field(default=20)
    agent_response_timeout: int = Field(default=120)
    enable_agent_streaming: bool = Field(default=True)
    agent_cleanup_interval: int = Field(default=600)  # 10 minutes

    # Session Management Configuration
    session_max_duration: int = Field(default=86400)  # 24 hours
    session_idle_timeout: int = Field(default=3600)  # 1 hour
    session_cleanup_interval: int = Field(default=300)  # 5 minutes
    max_sessions_per_user: int = Field(default=5)
    session_persistence_enabled: bool = Field(default=True)

    # Function Tools Configuration
    function_tools_enabled: bool = Field(default=True)
    max_concurrent_tool_calls: int = Field(default=3)
    tool_execution_timeout: int = Field(default=30)
    enable_tool_caching: bool = Field(default=True)
    tool_cache_ttl: int = Field(default=1800)  # 30 minutes
    tool_rate_limit_per_hour: int = Field(default=1000)

    # Model Performance Tuning
    gemini_streaming_enabled: bool = Field(default=True)
    gemini_response_format: str = Field(default="text")
    gemini_candidate_count: int = Field(default=1)
    gemini_stop_sequences: list[str] = Field(default_factory=list)

    # Context Management
    max_context_length: int = Field(default=32768)
    context_window_overlap: int = Field(default=512)
    enable_context_compression: bool = Field(default=True)
    context_summary_threshold: int = Field(default=16384)

    # Error Handling and Monitoring
    ai_error_retry_attempts: int = Field(default=3)
    ai_error_backoff_factor: float = Field(default=2.0)
    ai_error_max_backoff: float = Field(default=60.0)
    enable_ai_error_alerts: bool = Field(default=True)
    ai_response_validation: bool = Field(default=True)

    # Development and Testing AI Configuration
    ai_mock_responses: bool = Field(default=False)
    ai_debug_logging: bool = Field(default=False)
    ai_performance_logging: bool = Field(default=True)
    ai_token_usage_logging: bool = Field(default=True)

    # Security Configuration
    jwt_secret_key: str | None = Field(default=None)
    jwt_algorithm: str = Field(default="HS256")
    jwt_expiration_time: int = Field(default=3600)
    jwt_refresh_expiration_time: int = Field(default=604800)
    api_rate_limit: int = Field(default=1000)
    enable_cors: bool = Field(default=True)
    allowed_origins: list[str] = Field(default=["http://localhost:3000"])
    allowed_methods: list[str] = Field(default=["GET", "POST", "PUT", "DELETE"])
    allowed_headers: list[str] = Field(default=["*"])

    # Database Configuration
    database_type: str = Field(default="firestore")
    database_timeout: int = Field(default=30)
    database_retry_attempts: int = Field(default=3)
    cache_type: str = Field(default="memory")
    redis_url: str | None = Field(default=None)
    cache_ttl: int = Field(default=3600)
    cache_max_size: int = Field(default=1000)

    # Monitoring and Logging Configuration
    enable_cloud_logging: bool = Field(default=True)
    log_format: str = Field(default="structured")
    log_correlation_id: bool = Field(default=True)
    enable_metrics: bool = Field(default=True)
    metrics_port: int = Field(default=9090)
    health_check_path: str = Field(default="/health")
    enable_error_tracking: bool = Field(default=True)
    error_tracking_service: str = Field(default="cloud-logging")
    enable_performance_monitoring: bool = Field(default=True)
    trace_sample_rate: float = Field(default=0.1)

    # Business Logic Configuration
    default_trip_duration_days: int = Field(default=7)
    max_trip_duration_days: int = Field(default=30)
    min_trip_duration_days: int = Field(default=1)
    default_budget_currency: str = Field(default="INR")
    default_search_radius: int = Field(default=50)
    max_places_per_day: int = Field(default=8)
    min_places_per_day: int = Field(default=3)
    default_country: str = Field(default="India")
    supported_languages: list[str] = Field(default=["en", "hi"])
    default_timezone: str = Field(default="Asia/Kolkata")

    # Development Configuration
    enable_debug_toolbar: bool = Field(default=False)
    enable_profiler: bool = Field(default=False)
    enable_hot_reload: bool = Field(default=True)
    test_database_url: str = Field(default="memory://")
    enable_test_mode: bool = Field(default=False)
    test_data_seed: bool = Field(default=True)
    local_development: bool = Field(default=True)
    mock_external_apis: bool = Field(default=False)
    use_local_credentials: bool = Field(default=True)

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: str | list[str]) -> list[str]:
        """Parse CORS origins from environment variable."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    @field_validator("allowed_methods", mode="before")
    @classmethod
    def parse_cors_methods(cls, v: str | list[str]) -> list[str]:
        """Parse CORS methods from environment variable."""
        if isinstance(v, str):
            return [method.strip() for method in v.split(",")]
        return v

    @field_validator("allowed_headers", mode="before")
    @classmethod
    def parse_cors_headers(cls, v: str | list[str]) -> list[str]:
        """Parse CORS headers from environment variable."""
        if isinstance(v, str):
            return [header.strip() for header in v.split(",")]
        return v

    @field_validator("supported_languages", mode="before")
    @classmethod
    def parse_supported_languages(cls, v: str | list[str]) -> list[str]:
        """Parse supported languages from environment variable."""
        if isinstance(v, str):
            return [lang.strip() for lang in v.split(",")]
        return v

    @field_validator("maps_supported_languages", mode="before")
    @classmethod
    def parse_maps_supported_languages(cls, v: str | list[str]) -> list[str]:
        """Parse maps supported languages from environment variable."""
        if isinstance(v, str):
            return [lang.strip() for lang in v.split(",")]
        return v

    @field_validator("maps_default_place_fields", mode="before")
    @classmethod
    def parse_maps_default_place_fields(cls, v: str | list[str]) -> list[str]:
        """Parse maps default place fields from environment variable."""
        if isinstance(v, str):
            return [field.strip() for field in v.split(",")]
        return v

    @field_validator("maps_supported_travel_modes", mode="before")
    @classmethod
    def parse_maps_supported_travel_modes(cls, v: str | list[str]) -> list[str]:
        """Parse maps supported travel modes from environment variable."""
        if isinstance(v, str):
            return [mode.strip() for mode in v.split(",")]
        return v

    @field_validator("maps_geocoding_result_types", mode="before")
    @classmethod
    def parse_maps_geocoding_result_types(cls, v: str | list[str]) -> list[str]:
        """Parse maps geocoding result types from environment variable."""
        if isinstance(v, str):
            return [type_name.strip() for type_name in v.split(",")]
        return v

    @field_validator("maps_geocoding_location_types", mode="before")
    @classmethod
    def parse_maps_geocoding_location_types(cls, v: str | list[str]) -> list[str]:
        """Parse maps geocoding location types from environment variable."""
        if isinstance(v, str):
            return [location_type.strip() for location_type in v.split(",")]
        return v

    @field_validator("maps_nearby_search_priority_types", mode="before")
    @classmethod
    def parse_maps_nearby_search_priority_types(cls, v: str | list[str]) -> list[str]:
        """Parse maps nearby search priority types from environment variable."""
        if isinstance(v, str):
            return [place_type.strip() for place_type in v.split(",")]
        return v

    @field_validator("maps_allowed_domains", mode="before")
    @classmethod
    def parse_maps_allowed_domains(cls, v: str | list[str]) -> list[str]:
        """Parse maps allowed domains from environment variable."""
        if isinstance(v, str):
            return [domain.strip() for domain in v.split(",")]
        return v

    @field_validator("maps_allowed_ip_ranges", mode="before")
    @classmethod
    def parse_maps_allowed_ip_ranges(cls, v: str | list[str]) -> list[str]:
        """Parse maps allowed IP ranges from environment variable."""
        if isinstance(v, str):
            return [ip_range.strip() for ip_range in v.split(",")]
        return v

    @field_validator("maps_requests_per_second")
    @classmethod
    def validate_maps_requests_per_second(cls, v: float) -> float:
        """Validate maps requests per second is reasonable."""
        if v <= 0:
            raise ValueError("Maps requests per second must be greater than 0")
        if v > 1000:  # Sanity check
            raise ValueError("Maps requests per second seems too high (max 1000)")
        return v

    @field_validator("maps_default_radius")
    @classmethod
    def validate_maps_default_radius(cls, v: int) -> int:
        """Validate maps default radius is within Google Maps limits."""
        if v <= 0:
            raise ValueError("Maps default radius must be greater than 0")
        if v > 50000:  # Google Maps limit
            raise ValueError("Maps default radius cannot exceed 50,000 meters")
        return v

    @field_validator("maps_max_radius")
    @classmethod
    def validate_maps_max_radius(cls, v: int) -> int:
        """Validate maps max radius is within Google Maps limits."""
        if v <= 0:
            raise ValueError("Maps max radius must be greater than 0")
        if v > 50000:  # Google Maps limit
            raise ValueError(
                "Maps max radius cannot exceed 50,000 meters (Google Maps limit)"
            )
        return v

    @field_validator("maps_popular_place_min_rating")
    @classmethod
    def validate_maps_popular_place_min_rating(cls, v: float) -> float:
        """Validate popular place minimum rating is within valid range."""
        if not 0.0 <= v <= 5.0:
            raise ValueError("Popular place minimum rating must be between 0.0 and 5.0")
        return v

    @field_validator("maps_min_rating_filter")
    @classmethod
    def validate_maps_min_rating_filter(cls, v: float) -> float:
        """Validate minimum rating filter is within valid range."""
        if not 0.0 <= v <= 5.0:
            raise ValueError("Minimum rating filter must be between 0.0 and 5.0")
        return v

    def validate_required_fields(self) -> None:
        """Validate that required fields are present for production use."""
        required_fields = {
            "google_cloud_project": self.google_cloud_project,
            "vertex_ai_project_id": self.vertex_ai_project_id,
            "firebase_project_id": self.firebase_project_id,
            "google_maps_api_key": self.google_maps_api_key,
            "jwt_secret_key": self.jwt_secret_key,
        }

        missing_fields = [
            field for field, value in required_fields.items() if not value
        ]

        if missing_fields and not self.mock_external_apis:
            msg = (
                f"Missing required configuration fields: {', '.join(missing_fields)}. "
                f"Please set these in your environment variables or .env file."
            )
            raise ValueError(msg)

        # Additional validation for Maps configuration
        if self.google_maps_api_key and not self.mock_external_apis:
            # Validate Maps-specific configuration
            if self.maps_default_radius > self.maps_max_radius:
                raise ValueError(
                    "Maps default radius cannot be greater than max radius"
                )

            if self.maps_cache_ttl < 0:
                raise ValueError("Maps cache TTL must be non-negative")

            if self.maps_request_timeout <= 0:
                raise ValueError("Maps request timeout must be positive")

    def get_maps_config(self) -> dict:
        """Get Maps API specific configuration as a dictionary."""
        return {
            "api_key": self.google_maps_api_key,
            "requests_per_second": self.maps_requests_per_second,
            "cache_ttl": self.maps_cache_ttl,
            "cache_size": self.maps_cache_max_size,
            "timeout": self.maps_request_timeout,
            "max_retries": self.maps_max_retries,
            "default_language": self.maps_default_language,
            "default_region": self.maps_default_region,
            "default_radius": self.maps_default_radius,
            "enable_caching": self.maps_enable_caching,
            "enable_error_logging": self.maps_enable_error_logging,
        }

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
    }


class DevelopmentConfig(BaseConfig):
    """Development-specific configuration."""

    debug: bool = True
    log_level: str = "DEBUG"
    enable_hot_reload: bool = True
    enable_debug_toolbar: bool = True
    mock_external_apis: bool = True

    model_config = {
        "env_file": [".env.development", ".env"],
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
    }


class ProductionConfig(BaseConfig):
    """Production-specific configuration."""

    debug: bool = False
    log_level: str = "INFO"
    enable_hot_reload: bool = False
    enable_debug_toolbar: bool = False
    mock_external_apis: bool = False
    local_development: bool = False

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.validate_required_fields()

    model_config = {
        "env_file": [".env.production", ".env"],
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
    }


class TestingConfig(BaseConfig):
    """Testing-specific configuration."""

    debug: bool = True
    log_level: str = "DEBUG"
    enable_test_mode: bool = True
    mock_external_apis: bool = True
    database_type: str = "memory"

    model_config = {
        "env_file": [".env.testing", ".env"],
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
    }


@lru_cache
def get_settings() -> BaseConfig:
    """Get cached application settings based on environment.

    Returns:
        BaseConfig: The appropriate configuration instance.
    """
    environment = os.getenv("ENVIRONMENT", "development").lower()

    try:
        if environment == "production":
            return ProductionConfig()
        if environment == "testing":
            return TestingConfig()
        return DevelopmentConfig()
    except (ValidationError, ValueError) as e:
        # In development, provide helpful error message
        if environment == "development":
            print(f"Warning: Configuration validation failed: {e}")
            print("Continuing with mock external APIs enabled for development.")
            config = DevelopmentConfig()
            config.mock_external_apis = True
            return config
        raise


# Global settings instance
settings = get_settings()
