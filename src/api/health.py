"""Google Cloud Health Integration for AI-Powered Trip Planner Backend.

This module provides comprehensive health checks for Vertex AI connectivity,
model availability verification, service account permissions validation,
and overall system health monitoring.
"""

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.ai_services.function_tools import get_tool_registry
from src.ai_services.gemini_agents import get_agent_orchestrator
from src.ai_services.model_config import get_async_client
from src.ai_services.session_manager import get_session_manager
from src.auth.firebase_auth import initialize_firebase
from src.core.config import settings
from src.core.logging import get_logger
from src.database.firestore_client import get_firestore_client

logger = get_logger(__name__)

# Create router
router = APIRouter(prefix="/health", tags=["Health"])


class HealthStatus(BaseModel):
    """Health check status model."""

    status: str = Field(description="Health status: healthy, unhealthy, degraded")
    message: str = Field(description="Status message")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    details: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional details"
    )
    response_time_ms: Optional[float] = Field(
        default=None, description="Response time in milliseconds"
    )


class ComponentHealth(BaseModel):
    """Individual component health status."""

    name: str = Field(description="Component name")
    status: HealthStatus = Field(description="Component health status")
    dependencies: List[str] = Field(
        default_factory=list, description="Component dependencies"
    )
    version: Optional[str] = Field(default=None, description="Component version")


class SystemHealthResponse(BaseModel):
    """Complete system health response."""

    overall_status: str = Field(description="Overall system health")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    uptime_seconds: float = Field(description="System uptime in seconds")
    components: List[ComponentHealth] = Field(description="Individual component health")
    summary: Dict[str, Any] = Field(description="Health summary")


class HealthChecker:
    """Comprehensive health checker for all system components."""

    def __init__(self) -> None:
        """Initialize health checker."""
        self.start_time = datetime.now(timezone.utc)
        self.check_timeout = 10.0  # seconds

    async def check_vertex_ai_health(self) -> ComponentHealth:
        """Check Vertex AI model health.

        Returns:
            ComponentHealth: Vertex AI health status
        """
        start_time = datetime.now(timezone.utc)

        try:
            # Get async client
            client = get_async_client()

            # Perform health check
            is_healthy = await asyncio.wait_for(
                client.health_check(), timeout=self.check_timeout
            )

            response_time = (
                datetime.now(timezone.utc) - start_time
            ).total_seconds() * 1000

            if is_healthy:
                status = HealthStatus(
                    status="healthy",
                    message="Vertex AI model is operational",
                    response_time_ms=response_time,
                    details={
                        "project_id": settings.vertex_ai_project_id,
                        "region": settings.vertex_ai_region,
                        "model": settings.gemini_model,
                    },
                )
            else:
                status = HealthStatus(
                    status="unhealthy",
                    message="Vertex AI model health check failed",
                    response_time_ms=response_time,
                )

            return ComponentHealth(
                name="vertex_ai",
                status=status,
                dependencies=["google_cloud_auth"],
                version=settings.gemini_model,
            )

        except asyncio.TimeoutError:
            return ComponentHealth(
                name="vertex_ai",
                status=HealthStatus(
                    status="unhealthy",
                    message="Vertex AI health check timed out",
                    response_time_ms=self.check_timeout * 1000,
                ),
                dependencies=["google_cloud_auth"],
            )

        except Exception as e:
            logger.error("Vertex AI health check failed", error=str(e), exc_info=True)
            return ComponentHealth(
                name="vertex_ai",
                status=HealthStatus(
                    status="unhealthy",
                    message=f"Vertex AI error: {e!s}",
                    details={"error_type": type(e).__name__},
                ),
                dependencies=["google_cloud_auth"],
            )

    async def check_firestore_health(self) -> ComponentHealth:
        """Check Firestore database health.

        Returns:
            ComponentHealth: Firestore health status
        """
        start_time = datetime.now(timezone.utc)

        try:
            # Get Firestore client
            firestore_client = get_firestore_client()

            # Perform health check
            is_healthy = await asyncio.wait_for(
                firestore_client.health_check(), timeout=self.check_timeout
            )

            response_time = (
                datetime.now(timezone.utc) - start_time
            ).total_seconds() * 1000

            if is_healthy:
                status = HealthStatus(
                    status="healthy",
                    message="Firestore database is operational",
                    response_time_ms=response_time,
                    details={
                        "project_id": settings.firebase_project_id,
                        "database_id": settings.firestore_database_id,
                    },
                )
            else:
                status = HealthStatus(
                    status="unhealthy",
                    message="Firestore health check failed",
                    response_time_ms=response_time,
                )

            return ComponentHealth(
                name="firestore",
                status=status,
                dependencies=["firebase_auth"],
                version="cloud_firestore",
            )

        except asyncio.TimeoutError:
            return ComponentHealth(
                name="firestore",
                status=HealthStatus(
                    status="unhealthy",
                    message="Firestore health check timed out",
                    response_time_ms=self.check_timeout * 1000,
                ),
            )

        except Exception as e:
            logger.error("Firestore health check failed", error=str(e), exc_info=True)
            return ComponentHealth(
                name="firestore",
                status=HealthStatus(
                    status="unhealthy",
                    message=f"Firestore error: {e!s}",
                    details={"error_type": type(e).__name__},
                ),
            )

    async def check_firebase_auth_health(self) -> ComponentHealth:
        """Check Firebase Authentication health.

        Returns:
            ComponentHealth: Firebase Auth health status
        """
        start_time = datetime.now(timezone.utc)

        try:
            # Try to initialize Firebase
            await asyncio.wait_for(
                asyncio.to_thread(initialize_firebase), timeout=self.check_timeout
            )

            response_time = (
                datetime.now(timezone.utc) - start_time
            ).total_seconds() * 1000

            status = HealthStatus(
                status="healthy",
                message="Firebase Authentication is operational",
                response_time_ms=response_time,
                details={
                    "project_id": settings.firebase_project_id,
                    "auth_domain": settings.firebase_auth_domain,
                },
            )

            return ComponentHealth(
                name="firebase_auth",
                status=status,
                dependencies=["google_cloud_auth"],
                version="firebase_admin_sdk",
            )

        except asyncio.TimeoutError:
            return ComponentHealth(
                name="firebase_auth",
                status=HealthStatus(
                    status="unhealthy",
                    message="Firebase Auth initialization timed out",
                    response_time_ms=self.check_timeout * 1000,
                ),
            )

        except Exception as e:
            logger.error(
                "Firebase Auth health check failed", error=str(e), exc_info=True
            )
            return ComponentHealth(
                name="firebase_auth",
                status=HealthStatus(
                    status="unhealthy",
                    message=f"Firebase Auth error: {e!s}",
                    details={"error_type": type(e).__name__},
                ),
            )

    async def check_session_manager_health(self) -> ComponentHealth:
        """Check session manager health.

        Returns:
            ComponentHealth: Session manager health status
        """
        start_time = datetime.now(timezone.utc)

        try:
            # Get session manager
            session_manager = get_session_manager()

            # Get stats as health indicator
            stats = await asyncio.wait_for(
                session_manager.get_session_stats(), timeout=self.check_timeout
            )

            response_time = (
                datetime.now(timezone.utc) - start_time
            ).total_seconds() * 1000

            # Check for errors in stats
            if "error" in stats:
                status = HealthStatus(
                    status="degraded",
                    message=f"Session manager has issues: {stats['error']}",
                    response_time_ms=response_time,
                    details=stats,
                )
            else:
                status = HealthStatus(
                    status="healthy",
                    message="Session manager is operational",
                    response_time_ms=response_time,
                    details={
                        "active_sessions": stats.get("active_sessions_in_memory", 0),
                        "cleanup_running": stats.get("cleanup_task_running", False),
                    },
                )

            return ComponentHealth(
                name="session_manager",
                status=status,
                dependencies=["firestore"],
                version="v1.0",
            )

        except asyncio.TimeoutError:
            return ComponentHealth(
                name="session_manager",
                status=HealthStatus(
                    status="unhealthy",
                    message="Session manager health check timed out",
                    response_time_ms=self.check_timeout * 1000,
                ),
            )

        except Exception as e:
            logger.error(
                "Session manager health check failed", error=str(e), exc_info=True
            )
            return ComponentHealth(
                name="session_manager",
                status=HealthStatus(
                    status="unhealthy",
                    message=f"Session manager error: {e!s}",
                    details={"error_type": type(e).__name__},
                ),
            )

    async def check_function_tools_health(self) -> ComponentHealth:
        """Check function tools registry health.

        Returns:
            ComponentHealth: Function tools health status
        """
        start_time = datetime.now(timezone.utc)

        try:
            # Get tool registry
            tool_registry = get_tool_registry()

            # Get tool stats
            stats = tool_registry.get_tool_stats()

            response_time = (
                datetime.now(timezone.utc) - start_time
            ).total_seconds() * 1000

            total_tools = stats.get("total_tools", 0)

            if total_tools > 0:
                status = HealthStatus(
                    status="healthy",
                    message=f"Function tools registry operational with {total_tools} tools",
                    response_time_ms=response_time,
                    details={
                        "total_tools": total_tools,
                        "categories": stats.get("categories", {}),
                    },
                )
            else:
                status = HealthStatus(
                    status="degraded",
                    message="No function tools registered",
                    response_time_ms=response_time,
                )

            return ComponentHealth(name="function_tools", status=status, version="v1.0")

        except Exception as e:
            logger.error(
                "Function tools health check failed", error=str(e), exc_info=True
            )
            return ComponentHealth(
                name="function_tools",
                status=HealthStatus(
                    status="unhealthy",
                    message=f"Function tools error: {e!s}",
                    details={"error_type": type(e).__name__},
                ),
            )

    async def check_agent_orchestrator_health(self) -> ComponentHealth:
        """Check agent orchestrator health.

        Returns:
            ComponentHealth: Agent orchestrator health status
        """
        start_time = datetime.now(timezone.utc)

        try:
            # Get orchestrator
            orchestrator = get_agent_orchestrator()

            # Get agent stats
            stats = orchestrator.get_agent_stats()

            response_time = (
                datetime.now(timezone.utc) - start_time
            ).total_seconds() * 1000

            total_agents = stats.get("total_agents", 0)
            ready_agents = stats.get("agents_by_state", {}).get("ready", 0)
            error_agents = stats.get("agents_by_state", {}).get("error", 0)

            if error_agents > 0:
                status_level = "degraded"
                message = f"Agent orchestrator has {error_agents} agents in error state"
            elif total_agents > 0:
                status_level = "healthy"
                message = f"Agent orchestrator operational with {total_agents} agents"
            else:
                status_level = "healthy"
                message = "Agent orchestrator ready (no active agents)"

            status = HealthStatus(
                status=status_level,
                message=message,
                response_time_ms=response_time,
                details={
                    "total_agents": total_agents,
                    "ready_agents": ready_agents,
                    "error_agents": error_agents,
                    "agents_by_role": stats.get("agents_by_role", {}),
                },
            )

            return ComponentHealth(
                name="agent_orchestrator",
                status=status,
                dependencies=["vertex_ai", "session_manager", "function_tools"],
                version="v1.0",
            )

        except Exception as e:
            logger.error(
                "Agent orchestrator health check failed", error=str(e), exc_info=True
            )
            return ComponentHealth(
                name="agent_orchestrator",
                status=HealthStatus(
                    status="unhealthy",
                    message=f"Agent orchestrator error: {e!s}",
                    details={"error_type": type(e).__name__},
                ),
            )

    async def check_google_cloud_auth_health(self) -> ComponentHealth:
        """Check Google Cloud authentication health.

        Returns:
            ComponentHealth: Google Cloud auth health status
        """
        start_time = datetime.now(timezone.utc)

        try:
            # Check credentials availability
            credentials_available = bool(
                settings.google_application_credentials
                or settings.google_credentials_json
                or settings.firebase_private_key
            )

            response_time = (
                datetime.now(timezone.utc) - start_time
            ).total_seconds() * 1000

            if credentials_available:
                status = HealthStatus(
                    status="healthy",
                    message="Google Cloud credentials are configured",
                    response_time_ms=response_time,
                    details={
                        "project_id": settings.google_cloud_project
                        or settings.firebase_project_id,
                        "region": settings.google_cloud_region,
                        "has_service_account": bool(
                            settings.google_application_credentials
                        ),
                        "has_env_credentials": bool(settings.google_credentials_json),
                    },
                )
            else:
                status = HealthStatus(
                    status="unhealthy",
                    message="Google Cloud credentials not configured",
                    response_time_ms=response_time,
                )

            return ComponentHealth(
                name="google_cloud_auth", status=status, version="google_auth"
            )

        except Exception as e:
            logger.error(
                "Google Cloud auth health check failed", error=str(e), exc_info=True
            )
            return ComponentHealth(
                name="google_cloud_auth",
                status=HealthStatus(
                    status="unhealthy",
                    message=f"Google Cloud auth error: {e!s}",
                    details={"error_type": type(e).__name__},
                ),
            )

    async def perform_comprehensive_health_check(self) -> SystemHealthResponse:
        """Perform comprehensive system health check.

        Returns:
            SystemHealthResponse: Complete system health status
        """
        logger.info("Starting comprehensive health check")

        # Run all health checks concurrently
        health_checks = await asyncio.gather(
            self.check_google_cloud_auth_health(),
            self.check_firebase_auth_health(),
            self.check_firestore_health(),
            self.check_vertex_ai_health(),
            self.check_session_manager_health(),
            self.check_function_tools_health(),
            self.check_agent_orchestrator_health(),
            return_exceptions=True,
        )

        components = []
        healthy_count = 0
        unhealthy_count = 0
        degraded_count = 0

        # Process results
        for result in health_checks:
            if isinstance(result, Exception):
                logger.error("Health check failed with exception", error=str(result))
                components.append(
                    ComponentHealth(
                        name="unknown_component",
                        status=HealthStatus(
                            status="unhealthy",
                            message=f"Health check exception: {result!s}",
                        ),
                    )
                )
                unhealthy_count += 1
            else:
                components.append(result)
                if hasattr(result, "status") and result.status.status == "healthy":
                    healthy_count += 1
                elif hasattr(result, "status") and result.status.status == "unhealthy":
                    unhealthy_count += 1
                elif hasattr(result, "status") and result.status.status == "degraded":
                    degraded_count += 1

        # Determine overall status
        total_components = len(components)
        if unhealthy_count == 0 and degraded_count == 0:
            overall_status = "healthy"
        elif (
            unhealthy_count == 0 and degraded_count > 0
        ) or unhealthy_count < total_components // 2:
            overall_status = "degraded"
        else:
            overall_status = "unhealthy"

        # Calculate uptime
        uptime = (datetime.now(timezone.utc) - self.start_time).total_seconds()

        # Create summary
        summary = {
            "total_components": total_components,
            "healthy_components": healthy_count,
            "degraded_components": degraded_count,
            "unhealthy_components": unhealthy_count,
            "health_score": (
                (healthy_count / total_components * 100) if total_components > 0 else 0
            ),
            "environment": settings.environment,
            "debug_mode": settings.debug,
            "mock_apis": settings.mock_external_apis,
        }

        logger.info(
            "Health check completed",
            overall_status=overall_status,
            healthy=healthy_count,
            degraded=degraded_count,
            unhealthy=unhealthy_count,
            health_score=summary["health_score"],
        )

        return SystemHealthResponse(
            overall_status=overall_status,
            uptime_seconds=uptime,
            components=components,
            summary=summary,
        )


# Global health checker instance
health_checker = HealthChecker()


@router.get("/", response_model=SystemHealthResponse)
async def get_system_health() -> SystemHealthResponse:
    """Get comprehensive system health status.

    Returns:
        SystemHealthResponse: Complete system health information
    """
    try:
        return await health_checker.perform_comprehensive_health_check()
    except Exception as e:
        logger.error("System health check failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Health check failed: {e!s}",
        ) from e


@router.get("/live")
async def liveness_probe() -> JSONResponse:
    """Simple liveness probe for container orchestration.

    Returns:
        JSONResponse: Simple status response
    """
    return JSONResponse(
        content={
            "status": "alive",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "service": settings.app_name,
            "version": settings.app_version,
        },
        status_code=200,
    )


@router.get("/ready")
async def readiness_probe() -> JSONResponse:
    """Readiness probe checking critical services.

    Returns:
        JSONResponse: Readiness status
    """
    try:
        # Check critical services only
        critical_checks = await asyncio.gather(
            health_checker.check_google_cloud_auth_health(),
            health_checker.check_vertex_ai_health(),
            health_checker.check_firestore_health(),
            return_exceptions=True,
        )

        # Check if all critical services are healthy
        all_healthy = True
        for result in critical_checks:
            if isinstance(result, Exception) or (
                hasattr(result, "status") and result.status.status != "healthy"
            ):
                all_healthy = False
                break

        if all_healthy:
            return JSONResponse(
                content={
                    "status": "ready",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "message": "All critical services are healthy",
                },
                status_code=200,
            )
        else:
            return JSONResponse(
                content={
                    "status": "not_ready",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "message": "Critical services are not healthy",
                },
                status_code=503,
            )

    except Exception as e:
        logger.error("Readiness probe failed", error=str(e), exc_info=True)
        return JSONResponse(
            content={
                "status": "error",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "message": f"Readiness check failed: {e!s}",
            },
            status_code=503,
        )


@router.get("/vertex-ai")
async def get_vertex_ai_health() -> ComponentHealth:
    """Get Vertex AI specific health status.

    Returns:
        ComponentHealth: Vertex AI health information
    """
    return await health_checker.check_vertex_ai_health()


@router.get("/firestore")
async def get_firestore_health() -> ComponentHealth:
    """Get Firestore specific health status.

    Returns:
        ComponentHealth: Firestore health information
    """
    return await health_checker.check_firestore_health()


@router.get("/sessions")
async def get_session_manager_health() -> ComponentHealth:
    """Get session manager specific health status.

    Returns:
        ComponentHealth: Session manager health information
    """
    return await health_checker.check_session_manager_health()


@router.get("/agents")
async def get_agents_health() -> ComponentHealth:
    """Get agent orchestrator specific health status.

    Returns:
        ComponentHealth: Agent orchestrator health information
    """
    return await health_checker.check_agent_orchestrator_health()


@router.get("/tools")
async def get_function_tools_health() -> ComponentHealth:
    """Get function tools specific health status.

    Returns:
        ComponentHealth: Function tools health information
    """
    return await health_checker.check_function_tools_health()
