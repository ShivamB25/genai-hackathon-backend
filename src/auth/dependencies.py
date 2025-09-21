"""FastAPI Authentication Dependencies for AI-Powered Trip Planner Backend.

This module provides FastAPI dependencies for user authentication, authorization,
and token validation using Firebase ID tokens.
"""

from typing import Any

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.auth.firebase_auth import (
    FirebaseAuthError,
    TokenVerificationError,
    UserNotFoundError,
    authenticate_user,
)
from src.core.logging import get_logger

logger = get_logger(__name__)

# HTTPBearer security scheme for token extraction
security = HTTPBearer(auto_error=False)


class AuthenticationError(HTTPException):
    """Custom authentication error with proper HTTP status."""

    def __init__(self, detail: str = "Authentication failed") -> None:
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            headers={"WWW-Authenticate": "Bearer"},
        )


class AuthorizationError(HTTPException):
    """Custom authorization error with proper HTTP status."""

    def __init__(self, detail: str = "Insufficient permissions") -> None:
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=detail,
        )


async def extract_token(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> str | None:
    """Extract Bearer token from Authorization header.

    Args:
        credentials: HTTP Authorization credentials from request.

    Returns:
        Optional[str]: The extracted token if present, None otherwise.
    """
    if credentials is None:
        return None

    return credentials.credentials


async def extract_token_from_request(request: Request) -> str | None:
    """Extract token from various sources in the request.

    Args:
        request: The FastAPI request object.

    Returns:
        Optional[str]: The extracted token if found, None otherwise.
    """
    # Try Authorization header first
    auth_header = request.headers.get("authorization")
    if auth_header and auth_header.startswith("Bearer "):
        return auth_header.split(" ", 1)[1]

    # Try X-Auth-Token header
    auth_token = request.headers.get("x-auth-token")
    if auth_token:
        return auth_token

    # Try query parameter (less secure, mainly for WebSocket upgrades)
    token = request.query_params.get("token")
    if token:
        logger.warning("Token extracted from query parameter", path=request.url.path)
        return token

    return None


async def get_current_user(
    token: str | None = Depends(extract_token),
) -> dict[str, Any]:
    """Get current authenticated user from Firebase ID token.

    Args:
        token: The Firebase ID token.

    Returns:
        Dict[str, Any]: The authenticated user data.

    Raises:
        AuthenticationError: If authentication fails.
    """
    if not token:
        logger.warning("No authentication token provided")
        msg = "Authentication token required"
        raise AuthenticationError(msg)

    try:
        user_data = await authenticate_user(token)

        logger.debug(
            "User authenticated successfully",
            uid=user_data.get("uid"),
            email=user_data.get("auth_user", {}).get("email"),
        )

        return user_data

    except TokenVerificationError as e:
        logger.warning("Token verification failed", error=str(e))
        msg = "Invalid or expired token"
        raise AuthenticationError(msg) from e

    except UserNotFoundError as e:
        logger.warning("User not found", error=str(e))
        msg = "User account not found"
        raise AuthenticationError(msg) from e

    except FirebaseAuthError as e:
        logger.error("Firebase authentication error", error=str(e), exc_info=True)
        msg = "Authentication service error"
        raise AuthenticationError(msg) from e

    except Exception as e:
        logger.error("Unexpected authentication error", error=str(e), exc_info=True)
        msg = "Authentication failed"
        raise AuthenticationError(msg) from e


async def get_current_user_optional(
    token: str | None = Depends(extract_token),
) -> dict[str, Any] | None:
    """Get current user if authenticated, otherwise return None.

    Args:
        token: The Firebase ID token.

    Returns:
        Optional[Dict[str, Any]]: The authenticated user data if valid token, None otherwise.
    """
    if not token:
        return None

    try:
        return await authenticate_user(token)

    except (TokenVerificationError, UserNotFoundError, FirebaseAuthError):
        logger.debug("Optional authentication failed, returning None")
        return None

    except Exception as e:
        logger.warning("Unexpected error in optional authentication", error=str(e))
        return None


async def require_authenticated_user(
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Require an authenticated user (alias for get_current_user for clarity).

    Args:
        user: The authenticated user data.

    Returns:
        Dict[str, Any]: The authenticated user data.

    Raises:
        AuthenticationError: If user is not authenticated.
    """
    return user


async def require_verified_email(
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Require user with verified email address.

    Args:
        user: The authenticated user data.

    Returns:
        Dict[str, Any]: The authenticated user data.

    Raises:
        AuthorizationError: If email is not verified.
    """
    auth_user = user.get("auth_user", {})
    if not auth_user.get("email_verified", False):
        logger.warning(
            "Unverified email access attempt",
            uid=user.get("uid"),
            email=auth_user.get("email"),
        )
        msg = "Email verification required"
        raise AuthorizationError(msg)

    return user


async def require_admin_user(
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Require user with admin role.

    Args:
        user: The authenticated user data.

    Returns:
        Dict[str, Any]: The authenticated user data.

    Raises:
        AuthorizationError: If user is not an admin.
    """
    # Check for admin role in token claims
    token_claims = user.get("token_claims", {})
    custom_claims = token_claims.get("custom_claims", {})

    # Check admin role from custom claims
    is_admin = custom_claims.get("admin", False)
    user_roles = custom_claims.get("roles", [])

    if not is_admin and "admin" not in user_roles:
        logger.warning(
            "Admin access denied",
            uid=user.get("uid"),
            email=user.get("auth_user", {}).get("email"),
            roles=user_roles,
        )
        msg = "Admin access required"
        raise AuthorizationError(msg)

    logger.info(
        "Admin access granted",
        uid=user.get("uid"),
        email=user.get("auth_user", {}).get("email"),
    )

    return user


async def require_complete_profile(
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Require user with complete profile.

    Args:
        user: The authenticated user data.

    Returns:
        Dict[str, Any]: The authenticated user data.

    Raises:
        AuthorizationError: If profile is not complete.
    """
    profile = user.get("profile", {})
    if not profile.get("profile_complete", False):
        logger.warning(
            "Incomplete profile access attempt",
            uid=user.get("uid"),
            email=user.get("auth_user", {}).get("email"),
        )
        msg = "Complete profile setup required"
        raise AuthorizationError(msg)

    return user


async def require_terms_accepted(
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Require user who has accepted terms of service.

    Args:
        user: The authenticated user data.

    Returns:
        Dict[str, Any]: The authenticated user data.

    Raises:
        AuthorizationError: If terms are not accepted.
    """
    profile = user.get("profile", {})
    if not profile.get("terms_accepted", False):
        logger.warning(
            "Terms not accepted access attempt",
            uid=user.get("uid"),
            email=user.get("auth_user", {}).get("email"),
        )
        msg = "Terms of service acceptance required"
        raise AuthorizationError(msg)

    return user


def require_role(required_role: str):
    """Create a dependency that requires a specific role.

    Args:
        required_role: The role that is required.

    Returns:
        Callable: A FastAPI dependency function.
    """

    async def role_dependency(
        user: dict[str, Any] = Depends(get_current_user),
    ) -> dict[str, Any]:
        """Check if user has the required role.

        Args:
            user: The authenticated user data.

        Returns:
            Dict[str, Any]: The authenticated user data.

        Raises:
            AuthorizationError: If user doesn't have the required role.
        """
        token_claims = user.get("token_claims", {})
        custom_claims = token_claims.get("custom_claims", {})
        user_roles = custom_claims.get("roles", [])

        if required_role not in user_roles:
            logger.warning(
                "Role access denied",
                uid=user.get("uid"),
                email=user.get("auth_user", {}).get("email"),
                required_role=required_role,
                user_roles=user_roles,
            )
            msg = f"Role '{required_role}' required"
            raise AuthorizationError(msg)

        return user

    return role_dependency


def require_any_role(*roles: str):
    """Create a dependency that requires any of the specified roles.

    Args:
        *roles: The roles, any of which satisfies the requirement.

    Returns:
        Callable: A FastAPI dependency function.
    """

    async def any_role_dependency(
        user: dict[str, Any] = Depends(get_current_user),
    ) -> dict[str, Any]:
        """Check if user has any of the required roles.

        Args:
            user: The authenticated user data.

        Returns:
            Dict[str, Any]: The authenticated user data.

        Raises:
            AuthorizationError: If user doesn't have any of the required roles.
        """
        token_claims = user.get("token_claims", {})
        custom_claims = token_claims.get("custom_claims", {})
        user_roles = custom_claims.get("roles", [])

        if not any(role in user_roles for role in roles):
            logger.warning(
                "Any role access denied",
                uid=user.get("uid"),
                email=user.get("auth_user", {}).get("email"),
                required_roles=list(roles),
                user_roles=user_roles,
            )
            msg = f"One of roles {list(roles)} required"
            raise AuthorizationError(msg)

        return user

    return any_role_dependency


async def get_user_id(user: dict[str, Any] = Depends(get_current_user)) -> str:
    """Extract user ID from authenticated user.

    Args:
        user: The authenticated user data.

    Returns:
        str: The user's Firebase UID.

    Raises:
        AuthenticationError: If UID is not available.
    """
    uid = user.get("uid")
    if not uid:
        logger.error(
            "User ID not found in authenticated user data", user_keys=list(user.keys())
        )
        msg = "User ID not available"
        raise AuthenticationError(msg)
    return uid


async def get_user_email(
    user: dict[str, Any] = Depends(get_current_user),
) -> str | None:
    """Extract user email from authenticated user.

    Args:
        user: The authenticated user data.

    Returns:
        Optional[str]: The user's email if available.
    """
    return user.get("auth_user", {}).get("email")


# Commonly used dependency combinations
CurrentUser = Depends(get_current_user)
OptionalUser = Depends(get_current_user_optional)
VerifiedUser = Depends(require_verified_email)
AdminUser = Depends(require_admin_user)
CompleteUser = Depends(require_complete_profile)
AcceptedTermsUser = Depends(require_terms_accepted)
UserId = Depends(get_user_id)
UserEmail = Depends(get_user_email)
