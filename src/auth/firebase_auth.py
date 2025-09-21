"""Firebase Authentication Integration for AI-Powered Trip Planner Backend.

This module provides Firebase Admin SDK integration with async authentication functions,
user profile management, and secure token verification.
"""

import asyncio
import os
from typing import Any

import firebase_admin
from firebase_admin import auth, credentials
from firebase_admin.auth import (
    ExpiredIdTokenError,
    InvalidIdTokenError,
    RevokedIdTokenError,
)

from src.core.config import settings
from src.core.logging import get_logger
from src.database.firestore_client import (
    DocumentNotFoundError,
    FirestoreClient,
)
from src.database.firestore_client import (
    get_firestore_client as get_firestore_service,
)

logger = get_logger(__name__)

# Global Firebase app instance
_firebase_app: firebase_admin.App | None = None


class FirebaseAuthError(Exception):
    """Base exception for Firebase authentication errors."""


class TokenVerificationError(FirebaseAuthError):
    """Raised when Firebase ID token verification fails."""


class UserNotFoundError(FirebaseAuthError):
    """Raised when user is not found in Firebase Auth or Firestore."""


class UserCreationError(FirebaseAuthError):
    """Raised when user creation fails."""


def initialize_firebase() -> firebase_admin.App:
    """Initialize Firebase Admin SDK with service account credentials.

    Returns:
        firebase_admin.App: The initialized Firebase app instance.

    Raises:
        FirebaseAuthError: If Firebase initialization fails.
    """
    global _firebase_app

    if _firebase_app is not None:
        return _firebase_app

    try:
        # Try to get credentials from service account file
        service_account_path = None

        # Check if service account JSON file exists
        if os.path.exists("the-sandbox-460908-i8-5cdb12311a99.json"):
            service_account_path = "the-sandbox-460908-i8-5cdb12311a99.json"
        elif settings.google_application_credentials and os.path.exists(
            settings.google_application_credentials
        ):
            service_account_path = settings.google_application_credentials

        if service_account_path:
            logger.info(
                "Initializing Firebase Admin SDK with service account file",
                path=service_account_path,
            )
            cred = credentials.Certificate(service_account_path)
        # Try to use environment credentials or service account from config
        elif settings.firebase_private_key and settings.firebase_client_email:
            logger.info("Initializing Firebase Admin SDK with environment credentials")
            service_account_info = {
                "type": "service_account",
                "project_id": settings.firebase_project_id,
                "private_key_id": settings.firebase_private_key_id,
                "private_key": settings.firebase_private_key.replace("\\n", "\n"),
                "client_email": settings.firebase_client_email,
                "client_id": settings.firebase_client_id,
                "auth_uri": settings.firebase_auth_uri,
                "token_uri": settings.firebase_token_uri,
                "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                "client_x509_cert_url": settings.firebase_client_cert_url,
            }
            cred = credentials.Certificate(service_account_info)
        else:
            logger.info("Initializing Firebase Admin SDK with default credentials")
            cred = credentials.ApplicationDefault()

        # Initialize Firebase app
        _firebase_app = firebase_admin.initialize_app(
            cred,
            {
                "projectId": settings.firebase_project_id,
                "databaseURL": f"https://{settings.firebase_project_id}.firebaseio.com",
                "storageBucket": f"{settings.firebase_project_id}.appspot.com",
            },
            name="trip-planner-backend",
        )

        logger.info(
            "Firebase Admin SDK initialized successfully",
            project_id=settings.firebase_project_id,
        )
        return _firebase_app

    except Exception as e:
        error_msg = f"Failed to initialize Firebase Admin SDK: {e!s}"
        logger.error(error_msg, exc_info=True)
        raise FirebaseAuthError(error_msg) from e


async def verify_firebase_token(id_token: str) -> dict[str, Any]:
    """Verify Firebase ID token and return decoded claims.

    Args:
        id_token: The Firebase ID token to verify.

    Returns:
        Dict[str, Any]: The decoded token claims.

    Raises:
        TokenVerificationError: If token verification fails.
    """
    if not id_token:
        msg = "ID token is required"
        raise TokenVerificationError(msg)

    try:
        # Ensure Firebase is initialized
        initialize_firebase()

        # Verify the ID token
        decoded_token = await asyncio.to_thread(auth.verify_id_token, id_token)

        logger.info(
            "Firebase ID token verified successfully",
            uid=decoded_token.get("uid"),
            email=decoded_token.get("email"),
        )

        return decoded_token

    except ExpiredIdTokenError as e:
        error_msg = f"Expired ID token: {e!s}"
        logger.warning(error_msg, id_token_prefix=id_token[:20] if id_token else None)
        raise TokenVerificationError(error_msg) from e

    except RevokedIdTokenError as e:
        error_msg = f"Revoked ID token: {e!s}"
        logger.warning(error_msg, id_token_prefix=id_token[:20] if id_token else None)
        raise TokenVerificationError(error_msg) from e

    except InvalidIdTokenError as e:
        error_msg = f"Invalid ID token: {e!s}"
        logger.warning(error_msg, id_token_prefix=id_token[:20] if id_token else None)
        raise TokenVerificationError(error_msg) from e

    except Exception as e:
        error_msg = f"Token verification failed: {e!s}"
        logger.error(
            error_msg,
            exc_info=True,
            id_token_prefix=id_token[:20] if id_token else None,
        )
        raise TokenVerificationError(error_msg) from e


async def get_user_by_uid(uid: str) -> dict[str, Any] | None:
    """Get user record from Firebase Auth by UID.

    Args:
        uid: The user's Firebase UID.

    Returns:
        Optional[Dict[str, Any]]: User record if found, None otherwise.

    Raises:
        UserNotFoundError: If user retrieval fails.
    """
    if not uid:
        return None

    try:
        # Ensure Firebase is initialized
        initialize_firebase()

        # Get user record from Firebase Auth
        user_record = await asyncio.to_thread(auth.get_user, uid)

        # Convert to dictionary
        user_data = {
            "uid": user_record.uid,
            "email": user_record.email,
            "email_verified": user_record.email_verified,
            "display_name": user_record.display_name,
            "photo_url": user_record.photo_url,
            "phone_number": user_record.phone_number,
            "disabled": user_record.disabled,
            "creation_time": (
                user_record.user_metadata.creation_timestamp.isoformat()
                if user_record.user_metadata.creation_timestamp
                else None
            ),
            "last_sign_in_time": (
                user_record.user_metadata.last_sign_in_timestamp.isoformat()
                if user_record.user_metadata.last_sign_in_timestamp
                else None
            ),
            "provider_data": [
                {
                    "uid": provider.uid,
                    "email": provider.email,
                    "provider_id": provider.provider_id,
                    "display_name": provider.display_name,
                    "photo_url": provider.photo_url,
                }
                for provider in user_record.provider_data
            ],
        }

        logger.debug(
            "User record retrieved successfully", uid=uid, email=user_record.email
        )
        return user_data

    except auth.UserNotFoundError:
        logger.warning("User not found in Firebase Auth", uid=uid)
        return None

    except Exception as e:
        error_msg = f"Failed to retrieve user by UID: {e!s}"
        logger.error(error_msg, exc_info=True, uid=uid)
        raise UserNotFoundError(error_msg) from e


async def get_user_profile(uid: str) -> dict[str, Any] | None:
    """Get user profile from Firestore.

    Args:
        uid: The user's Firebase UID.

    Returns:
        Optional[Dict[str, Any]]: User profile if found, None otherwise.
    """
    if not uid:
        return None

    try:
        initialize_firebase()
        firestore_client: FirestoreClient = get_firestore_service()

        profile = await firestore_client.get_document("users", uid)
        if profile:
            profile.pop("id", None)
            logger.debug("User profile retrieved from Firestore", uid=uid)
            return profile
        logger.debug("User profile not found in Firestore", uid=uid)
        return None

    except Exception as e:
        error_msg = f"Failed to retrieve user profile: {e!s}"
        logger.error(error_msg, exc_info=True, uid=uid)
        return None


async def create_user_profile(uid: str, user_data: dict[str, Any]) -> dict[str, Any]:
    """Create or update user profile in Firestore.

    Args:
        uid: The user's Firebase UID.
        user_data: User profile data to save.

    Returns:
        Dict[str, Any]: The created/updated user profile.

    Raises:
        UserCreationError: If user profile creation fails.
    """
    if not uid:
        msg = "UID is required"
        raise UserCreationError(msg)

    try:
        initialize_firebase()
        firestore_client: FirestoreClient = get_firestore_service()

        profile_data = {
            "uid": uid,
            "email": user_data.get("email"),
            "display_name": user_data.get("display_name"),
            "photo_url": user_data.get("photo_url"),
            "phone_number": user_data.get("phone_number"),
            "email_verified": user_data.get("email_verified", False),
            "preferences": {
                "currency": user_data.get("currency", settings.default_budget_currency),
                "timezone": user_data.get("timezone", settings.default_timezone),
                "language": user_data.get("language", "en"),
                "country": user_data.get("country", settings.default_country),
            },
            "profile_complete": False,
            "terms_accepted": False,
        }

        if "preferences" in user_data:
            profile_data["preferences"].update(user_data["preferences"])

        try:
            await firestore_client.update_document(
                "users", uid, profile_data, merge=True
            )
        except DocumentNotFoundError:
            await firestore_client.create_document(
                "users", profile_data, document_id=uid
            )

        stored_profile = await firestore_client.get_document("users", uid)
        if stored_profile:
            stored_profile.pop("id", None)
            profile_payload = stored_profile
        else:
            profile_payload = profile_data

        logger.info(
            "User profile created/updated successfully",
            uid=uid,
            email=user_data.get("email"),
        )
        return profile_payload

    except Exception as e:
        error_msg = f"Failed to create user profile: {e!s}"
        logger.error(error_msg, exc_info=True, uid=uid)
        raise UserCreationError(error_msg) from e


async def authenticate_user(id_token: str) -> dict[str, Any]:
    """Authenticate user with Firebase ID token and return user data.

    Args:
        id_token: The Firebase ID token.

    Returns:
        Dict[str, Any]: Complete user data including profile.

    Raises:
        TokenVerificationError: If token verification fails.
        UserNotFoundError: If user retrieval fails.
    """
    # Verify the ID token
    decoded_token = await verify_firebase_token(id_token)
    uid = decoded_token.get("uid")

    if not uid:
        msg = "Invalid token: UID not found"
        raise TokenVerificationError(msg)

    # Get user record from Firebase Auth
    auth_user = await get_user_by_uid(uid)
    if not auth_user:
        msg = f"User not found: {uid}"
        raise UserNotFoundError(msg)

    # Get or create user profile in Firestore
    profile = await get_user_profile(uid)
    if not profile:
        # Create new profile from auth data
        profile = await create_user_profile(uid, auth_user)

    # Combine auth data and profile
    user_data = {
        "uid": uid,
        "token_claims": decoded_token,
        "auth_user": auth_user,
        "profile": profile,
    }

    logger.info(
        "User authenticated successfully", uid=uid, email=auth_user.get("email")
    )
    return user_data


async def revoke_user_tokens(uid: str) -> bool:
    """Revoke all refresh tokens for a user.

    Args:
        uid: The user's Firebase UID.

    Returns:
        bool: True if successful, False otherwise.
    """
    if not uid:
        return False

    try:
        # Ensure Firebase is initialized
        initialize_firebase()

        # Revoke all refresh tokens for the user
        await asyncio.to_thread(auth.revoke_refresh_tokens, uid)

        logger.info("User tokens revoked successfully", uid=uid)
        return True

    except Exception as e:
        logger.error(
            "Failed to revoke user tokens", exc_info=True, uid=uid, error=str(e)
        )
        return False


async def delete_user_account(uid: str) -> bool:
    """Delete user account from Firebase Auth and Firestore.

    Args:
        uid: The user's Firebase UID.

    Returns:
        bool: True if successful, False otherwise.
    """
    if not uid:
        return False

    try:
        # Ensure Firebase is initialized
        initialize_firebase()
        firestore_client: FirestoreClient = get_firestore_service()

        # Delete user from Firebase Auth
        await asyncio.to_thread(auth.delete_user, uid)

        # Delete user profile from Firestore
        profile_deleted = await firestore_client.delete_document("users", uid)

        logger.info(
            "User account deleted successfully",
            uid=uid,
            profile_deleted=profile_deleted,
        )
        return True

    except Exception as e:
        logger.error(
            "Failed to delete user account", exc_info=True, uid=uid, error=str(e)
        )
        return False


# Initialize Firebase on module import if not in test mode
if not settings.enable_test_mode:
    try:
        initialize_firebase()
    except Exception as e:
        logger.warning("Failed to initialize Firebase on import", error=str(e))
