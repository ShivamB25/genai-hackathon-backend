"""AI-Powered Trip Planner Backend - Main FastAPI Application

This module provides the main FastAPI application with proper configuration,
middleware, exception handling, and authentication for the AI-powered trip planner.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.health import router as health_router
from src.api.middleware import register_middleware
from src.auth.firebase_auth import initialize_firebase
from src.auth.routes import router as auth_router
from src.core.config import settings
from src.core.exceptions import register_exception_handlers
from src.core.logging import get_logger
from src.database.firestore_client import close_firestore_client, get_firestore_client
from src.maps_services.routes import router as maps_router
from src.trip_planner.routes import router as trip_planner_router

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager for FastAPI application startup and shutdown.

    Args:
        app: FastAPI application instance.

    Yields:
        None: Control during application lifecycle.
    """
    # Startup
    logger.info(
        "Starting AI-Powered Trip Planner Backend",
        app_name=settings.app_name,
        version=settings.app_version,
        environment=settings.environment,
        debug=settings.debug,
    )

    # Initialize Firebase and Firestore
    try:
        if not settings.enable_test_mode:
            # Initialize Firebase Admin SDK
            logger.info("Initializing Firebase services...")
            initialize_firebase()

            # Initialize Firestore client
            firestore_client = get_firestore_client()
            health_check = await firestore_client.health_check()
            if health_check:
                logger.info("Firestore connection established successfully")
            else:
                logger.warning("Firestore health check failed, but continuing startup")
        else:
            logger.info("Test mode enabled - skipping Firebase initialization")
    except Exception:
        logger.exception("Failed to initialize Firebase services")
        if not settings.debug and not settings.mock_external_apis:
            raise  # Fail startup in production if Firebase can't be initialized

    logger.info("Application startup complete")

    yield

    # Shutdown
    logger.info("Shutting down AI-Powered Trip Planner Backend")

    try:
        # Clean up Firestore client
        await close_firestore_client()
        logger.info("Firestore client closed")
    except Exception:
        logger.exception("Error during Firestore cleanup")

    logger.info("Application shutdown complete")


# Initialize FastAPI application
app = FastAPI(
    title="AI-Powered Trip Planner Backend",
    description=(
        "A comprehensive trip planning backend using Google Vertex AI Gemini, "
        "Firebase, and Google Maps APIs to create personalized travel itineraries. "
        "Features include AI-powered trip generation, multi-agent workflow orchestration, "
        "Firebase authentication, Firestore persistence, and comprehensive Maps API integration."
    ),
    version=settings.app_version,
    debug=settings.debug,
    lifespan=lifespan,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    openapi_url="/openapi.json" if settings.debug else None,
    contact={
        "name": "AI Trip Planner API",
        "email": "support@aitripplanner.com",
    },
    license_info={
        "name": "MIT License",
    },
)


# Register exception handlers
register_exception_handlers(app)

# Add middleware
register_middleware(app)


# Include API routers with proper prefixes and Firebase auth
app.include_router(
    trip_planner_router,
    prefix="",  # Router already has /api/v1/trips prefix
    tags=["Trip Planning"],
)

app.include_router(
    maps_router,
    prefix="",  # Router already has /api/v1/places prefix
    tags=["Maps & Places"],
)

app.include_router(
    auth_router,
    prefix="",  # Router already has /api/v1/users prefix
    tags=["User Management"],
)

app.include_router(
    health_router,
    prefix="/api/v1",
    tags=["Health & Monitoring"],
)


# Health check endpoints
@app.get(
    settings.health_check_path,
    tags=["Health"],
    summary="Health Check",
    description="Check if the API and its dependencies are healthy",
)
async def health_check() -> dict:
    """Health check endpoint for monitoring and load balancers.

    Returns:
        dict: Health status information.
    """
    checks = {}
    overall_status = "healthy"

    # Check database
    try:
        if not settings.enable_test_mode:
            firestore_client = get_firestore_client()
            firestore_healthy = await firestore_client.health_check()
            checks["database"] = "healthy" if firestore_healthy else "unhealthy"
            checks["firebase"] = "healthy" if firestore_healthy else "unhealthy"

            if not firestore_healthy:
                overall_status = "degraded"
        else:
            checks["database"] = "test_mode"
            checks["firebase"] = "test_mode"
    except Exception:
        logger.exception("Health check - Firestore error")
        checks["database"] = "error"
        checks["firebase"] = "error"
        overall_status = "degraded"

    # Check required configuration
    if settings.google_cloud_project:
        checks["vertex_ai"] = "configured"

    if settings.google_maps_api_key:
        checks["maps_api"] = "configured"

    return {
        "status": overall_status,
        "service": settings.app_name,
        "version": settings.app_version,
        "checks": checks,
        "timestamp": logger._context.get("timestamp"),
    }


@app.get(
    "/health/live",
    tags=["Health"],
    summary="Liveness Check",
    description="Check if the API process is alive",
)
async def liveness_check() -> dict[str, str]:
    """Liveness check endpoint for Kubernetes/container orchestration.

    Returns:
        dict[str, str]: Liveness status information.
    """
    return {
        "status": "alive",
        "service": settings.app_name,
        "version": settings.app_version,
    }


# Root endpoint
@app.get(
    "/",
    tags=["Root"],
    summary="Root Endpoint",
    description="Welcome message and API information",
)
async def root() -> dict:
    """Root endpoint providing basic API information.

    Returns:
        dict: API welcome message and information.
    """
    return {
        "message": "Welcome to AI-Powered Trip Planner Backend",
        "service": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
        "docs_url": (
            "/docs" if settings.debug else "Documentation not available in production"
        ),
        "health_check": settings.health_check_path,
        "features": {
            "authentication": "Firebase Auth",
            "database": "Firestore",
            "ai_model": "Vertex AI Gemini Multi-Agent System",
            "maps": "Google Maps API",
            "trip_planning": "AI-Powered Itinerary Generation",
            "background_tasks": "Async Trip Generation & Optimization",
        },
        "api_endpoints": {
            "trip_planning": "/api/v1/trips",
            "places_search": "/api/v1/places",
            "user_management": "/api/v1/users",
            "health_monitoring": "/api/v1/health",
        },
    }


# Development-specific endpoints
if settings.debug:

    @app.get(
        "/debug/config",
        tags=["Debug"],
        summary="Configuration Debug",
        description="Display current configuration (development only)",
    )
    async def debug_config() -> dict:
        """Debug endpoint to display current configuration.

        Returns:
            dict: Current configuration (sensitive values masked).
        """
        config_dict = settings.model_dump()

        # Mask sensitive configuration values
        sensitive_keys = [
            "jwt_secret_key",
            "firebase_private_key",
            "firebase_private_key_id",
            "google_application_credentials",
            "google_credentials_json",
            "google_maps_api_key",
        ]

        for key in sensitive_keys:
            if config_dict.get(key):
                config_dict[key] = "***masked***"

        return {
            "configuration": config_dict,
            "environment": settings.environment,
            "debug": settings.debug,
        }

    @app.get(
        "/debug/middleware",
        tags=["Debug"],
        summary="Middleware Debug",
        description="Display registered middleware information (development only)",
    )
    async def debug_middleware() -> dict:
        """Debug endpoint to display middleware information.

        Returns:
            dict: Middleware information.
        """
        middleware_info = []
        for middleware in app.user_middleware:
            try:
                middleware_info.append(
                    {
                        "class": getattr(
                            middleware.cls, "__name__", str(middleware.cls)
                        ),
                        "options": getattr(middleware, "options", {}),
                    }
                )
            except Exception:
                middleware_info.append(
                    {
                        "class": str(middleware),
                        "options": {},
                    }
                )

        return {
            "middleware": middleware_info,
            "count": len(middleware_info),
        }

    @app.get(
        "/debug/health",
        tags=["Debug"],
        summary="Extended Health Check",
        description="Extended health check with detailed service information (development only)",
    )
    async def debug_health() -> dict:
        """Extended health check for development debugging.

        Returns:
            dict: Detailed health information.
        """
        health_info = {
            "application": {
                "name": settings.app_name,
                "version": settings.app_version,
                "environment": settings.environment,
                "debug": settings.debug,
            },
            "configuration": {
                "test_mode": settings.enable_test_mode,
                "mock_apis": settings.mock_external_apis,
                "vertex_ai_configured": bool(settings.vertex_ai_project_id),
                "maps_configured": bool(settings.google_maps_api_key),
                "firebase_configured": bool(settings.firebase_project_id),
            },
            "api_routes": {
                "trip_planning": "POST /api/v1/trips/plan, GET /api/v1/trips, PUT /api/v1/trips/{id}",
                "places_search": "GET /api/v1/places/search, GET /api/v1/places/nearby",
                "user_management": "GET /api/v1/users/profile, PUT /api/v1/users/profile",
                "directions": "POST /api/v1/places/directions",
                "geocoding": "POST /api/v1/places/geocode",
            },
        }

        # Add database health if not in test mode
        if not settings.enable_test_mode:
            try:
                firestore_client = get_firestore_client()
                db_health = await firestore_client.health_check()
                health_info["database"] = {
                    "status": "healthy" if db_health else "unhealthy",
                    "type": "Firestore",
                    "project": settings.firebase_project_id,
                }
            except Exception as e:
                health_info["database"] = {
                    "status": "error",
                    "error": str(e),
                }

        return health_info


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )
