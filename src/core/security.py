"""Core security utilities for AI-Powered Trip Planner Backend.

This module provides security functions including encryption, decryption,
API key validation, input sanitization, and security headers configuration.
"""

import base64
import hashlib
import hmac
import re
import secrets
import string
from typing import Any
from urllib.parse import urlparse

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from src.core.config import settings
from src.core.logging import get_logger

logger = get_logger(__name__)


class SecurityError(Exception):
    """Base exception for security-related errors."""


class EncryptionError(SecurityError):
    """Raised when encryption/decryption operations fail."""


class ValidationError(SecurityError):
    """Raised when input validation fails."""


# Encryption utilities
def generate_key_from_password(password: str, salt: bytes) -> bytes:
    """Generate encryption key from password using PBKDF2.

    Args:
        password: The password to derive key from.
        salt: Salt bytes for key derivation.

    Returns:
        bytes: The derived encryption key.
    """
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )
    return base64.urlsafe_b64encode(kdf.derive(password.encode()))


def encrypt_secret(data: str, key: str | None = None) -> str:
    """Encrypt sensitive data using Fernet symmetric encryption.

    Args:
        data: The data to encrypt.
        key: Optional encryption key. Uses app secret key if not provided.

    Returns:
        str: The encrypted data as base64 string.

    Raises:
        EncryptionError: If encryption fails.
    """
    if not data:
        return ""

    try:
        # Use provided key or derive from app secret
        if key is None:
            if not settings.jwt_secret_key:
                msg = "No encryption key available"
                raise EncryptionError(msg)
            key = settings.jwt_secret_key

        # Generate salt and derive key
        salt = secrets.token_bytes(16)
        encryption_key = generate_key_from_password(key, salt)

        # Create Fernet instance and encrypt
        fernet = Fernet(encryption_key)
        encrypted_data = fernet.encrypt(data.encode())

        # Combine salt and encrypted data
        combined = salt + encrypted_data
        return base64.urlsafe_b64encode(combined).decode()

    except Exception as e:
        logger.exception("Encryption failed", error=str(e))
        error_msg = "Failed to encrypt data"
        raise EncryptionError(error_msg) from e


def decrypt_secret(encrypted_data: str, key: str | None = None) -> str:
    """Decrypt data encrypted with encrypt_secret.

    Args:
        encrypted_data: The encrypted data as base64 string.
        key: Optional decryption key. Uses app secret key if not provided.

    Returns:
        str: The decrypted data.

    Raises:
        EncryptionError: If decryption fails.
    """
    if not encrypted_data:
        return ""

    try:
        # Use provided key or derive from app secret
        if key is None:
            if not settings.jwt_secret_key:
                msg = "No decryption key available"
                raise EncryptionError(msg)
            key = settings.jwt_secret_key

        # Decode base64 data
        combined = base64.urlsafe_b64decode(encrypted_data.encode())

        # Extract salt and encrypted data
        salt = combined[:16]
        encrypted_bytes = combined[16:]

        # Derive key and decrypt
        decryption_key = generate_key_from_password(key, salt)
        fernet = Fernet(decryption_key)
        decrypted_data = fernet.decrypt(encrypted_bytes)

        return decrypted_data.decode()

    except Exception as e:
        logger.exception("Decryption failed")
        error_msg = "Failed to decrypt data"
        raise EncryptionError(error_msg) from e


# Hash utilities
def hash_api_key(api_key: str) -> str:
    """Create a secure hash of an API key.

    Args:
        api_key: The API key to hash.

    Returns:
        str: The hashed API key.
    """
    if not api_key:
        return ""

    # Use SHA-256 with salt from secret key
    salt = (
        settings.jwt_secret_key[:16].encode()
        if settings.jwt_secret_key
        else b"default_salt"
    )
    return hashlib.pbkdf2_hmac("sha256", api_key.encode(), salt, 100000).hex()


def verify_api_key(api_key: str, hashed_key: str) -> bool:
    """Verify an API key against its hash.

    Args:
        api_key: The API key to verify.
        hashed_key: The stored hash to verify against.

    Returns:
        bool: True if the key is valid, False otherwise.
    """
    if not api_key or not hashed_key:
        return False

    try:
        computed_hash = hash_api_key(api_key)
        return hmac.compare_digest(computed_hash, hashed_key)
    except Exception as e:
        logger.warning("API key verification failed", error=str(e))
        return False


def generate_api_key(length: int = 32) -> str:
    """Generate a cryptographically secure API key.

    Args:
        length: Length of the API key to generate.

    Returns:
        str: The generated API key.
    """
    length = max(length, 16)
    length = min(length, 128)

    alphabet = string.ascii_letters + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


def generate_csrf_token() -> str:
    """Generate a CSRF token.

    Returns:
        str: A secure CSRF token.
    """
    return secrets.token_urlsafe(32)


# Input sanitization
def sanitize_string(input_str: str, max_length: int = 1000) -> str:
    """Sanitize string input by removing potentially dangerous characters.

    Args:
        input_str: The string to sanitize.
        max_length: Maximum allowed length.

    Returns:
        str: The sanitized string.
    """
    if not isinstance(input_str, str):
        return ""

    # Truncate to max length
    sanitized = input_str[:max_length]

    # Remove null bytes and control characters
    sanitized = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", sanitized)

    # Remove potentially dangerous HTML/script content
    sanitized = re.sub(
        r"<script[^>]*>.*?</script>", "", sanitized, flags=re.IGNORECASE | re.DOTALL
    )
    sanitized = re.sub(
        r"<style[^>]*>.*?</style>", "", sanitized, flags=re.IGNORECASE | re.DOTALL
    )
    sanitized = re.sub(r"javascript:", "", sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r"vbscript:", "", sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r"on\w+\s*=", "", sanitized, flags=re.IGNORECASE)

    return sanitized.strip()


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent directory traversal and other attacks.

    Args:
        filename: The filename to sanitize.

    Returns:
        str: The sanitized filename.
    """
    if not isinstance(filename, str):
        return "file"

    # Remove directory traversal attempts
    sanitized = filename.replace("..", "").replace("/", "").replace("\\", "")

    # Remove potentially dangerous characters
    sanitized = re.sub(r'[<>:"|?*\x00-\x1f]', "", sanitized)

    # Limit length
    sanitized = sanitized[:255]

    # Ensure we have a valid filename
    if not sanitized or sanitized.startswith("."):
        sanitized = f"file_{secrets.token_hex(4)}"

    return sanitized


def validate_url(url: str, allowed_schemes: list[str] | None = None) -> bool:
    """Validate URL to prevent SSRF attacks.

    Args:
        url: The URL to validate.
        allowed_schemes: List of allowed URL schemes.

    Returns:
        bool: True if URL is safe, False otherwise.
    """
    if not isinstance(url, str) or not url:
        return False

    if allowed_schemes is None:
        allowed_schemes = ["http", "https"]

    try:
        parsed = urlparse(url)

        # Check scheme
        if parsed.scheme.lower() not in allowed_schemes:
            return False

        # Check for private/local IP addresses
        hostname = parsed.hostname
        if hostname:
            hostname = hostname.lower()

            # Block localhost and loopback
            if hostname in ["localhost", "127.0.0.1", "::1"]:
                return False

            # Block private IP ranges (basic check)
            if hostname.startswith(("10.", "192.168.", "172.")):
                return False

            # Block other local addresses
            if hostname.startswith(("0.", "169.254.")):
                return False

        return True

    except Exception:
        return False


def validate_email_domain(email: str, allowed_domains: list[str] | None = None) -> bool:
    """Validate email domain against allowlist.

    Args:
        email: The email address to validate.
        allowed_domains: List of allowed domains. If None, all domains are allowed.

    Returns:
        bool: True if email domain is allowed, False otherwise.
    """
    if not email or "@" not in email:
        return False

    if allowed_domains is None:
        return True

    try:
        domain = email.split("@")[1].lower()
        return domain in [d.lower() for d in allowed_domains]
    except (IndexError, AttributeError):
        return False


# Security headers
def get_security_headers() -> dict[str, str]:
    """Get security headers for HTTP responses.

    Returns:
        Dict[str, str]: Dictionary of security headers.
    """
    return {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Content-Security-Policy": (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://apis.google.com; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "img-src 'self' data: https: blob:; "
            "font-src 'self' https://fonts.gstatic.com; "
            "connect-src 'self' https://api.openai.com https://*.googleapis.com "
            "https://*.google.com https://maps.googleapis.com; "
            "frame-ancestors 'none'; "
            "base-uri 'self'; "
            "form-action 'self';"
        ),
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "Permissions-Policy": (
            "camera=(), microphone=(), geolocation=(), "
            "gyroscope=(), magnetometer=(), payment=()"
        ),
    }


def get_cors_headers(origin: str = "*") -> dict[str, str]:
    """Get CORS headers for cross-origin requests.

    Args:
        origin: The allowed origin. Use "*" for all origins
               (not recommended for production).

    Returns:
        Dict[str, str]: Dictionary of CORS headers.
    """
    return {
        "Access-Control-Allow-Origin": origin,
        "Access-Control-Allow-Methods": ", ".join(settings.allowed_methods),
        "Access-Control-Allow-Headers": ", ".join(settings.allowed_headers),
        "Access-Control-Allow-Credentials": "true",
        "Access-Control-Max-Age": "86400",  # 24 hours
    }


# Rate limiting utilities
def generate_rate_limit_key(
    identifier: str, endpoint: str, window: str = "hour"
) -> str:
    """Generate a rate limiting key for Redis or memory cache.

    Args:
        identifier: User identifier (IP, user ID, etc.).
        endpoint: API endpoint being accessed.
        window: Time window for rate limiting.

    Returns:
        str: The rate limiting key.
    """
    sanitized_identifier = sanitize_string(identifier, 100)
    sanitized_endpoint = sanitize_string(endpoint, 200)
    return f"rate_limit:{window}:{sanitized_identifier}:{sanitized_endpoint}"


# Input validation utilities
def validate_json_structure(data: Any, required_fields: list[str]) -> bool:
    """Validate JSON structure has required fields.

    Args:
        data: The data to validate.
        required_fields: List of required field names.

    Returns:
        bool: True if all required fields are present, False otherwise.
    """
    if not isinstance(data, dict):
        return False

    return all(field in data for field in required_fields)


def sanitize_json_input(data: dict[str, Any], max_depth: int = 10) -> dict[str, Any]:
    """Recursively sanitize JSON input to prevent attacks.

    Args:
        data: The JSON data to sanitize.
        max_depth: Maximum nesting depth allowed.

    Returns:
        Dict[str, Any]: The sanitized JSON data.

    Raises:
        ValidationError: If input exceeds maximum depth.
    """

    def _sanitize_recursive(obj: Any, depth: int = 0) -> Any:
        if depth > max_depth:
            error_msg = f"JSON nesting depth exceeds maximum of {max_depth}"
            raise ValidationError(error_msg)

        if isinstance(obj, dict):
            return {
                sanitize_string(str(k), 100): _sanitize_recursive(v, depth + 1)
                for k, v in obj.items()
            }
        if isinstance(obj, list):
            return [
                _sanitize_recursive(item, depth + 1) for item in obj[:1000]
            ]  # Limit array size
        if isinstance(obj, str):
            return sanitize_string(obj, 10000)  # Limit string length
        if isinstance(obj, int | float | bool) or obj is None:
            return obj
        return str(obj)  # Convert unknown types to string

    if not isinstance(data, dict):
        error_msg = "Input must be a JSON object"
        raise ValidationError(error_msg)

    return _sanitize_recursive(data)


# Password constants
MIN_PASSWORD_LENGTH = 8
MAX_PASSWORD_LENGTH = 128


def validate_password_strength(password: str) -> tuple[bool, list[str]]:
    """Validate password strength.

    Args:
        password: The password to validate.

    Returns:
        tuple[bool, List[str]]: (is_valid, list_of_issues)
    """
    issues = []

    if len(password) < MIN_PASSWORD_LENGTH:
        issues.append("Password must be at least 8 characters long")

    if len(password) > MAX_PASSWORD_LENGTH:
        issues.append("Password must be no more than 128 characters long")

    if not re.search(r"[A-Z]", password):
        issues.append("Password must contain at least one uppercase letter")

    if not re.search(r"[a-z]", password):
        issues.append("Password must contain at least one lowercase letter")

    if not re.search(r"\d", password):
        issues.append("Password must contain at least one digit")

    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        issues.append("Password must contain at least one special character")

    # Check for common weak patterns
    common_patterns = [
        r"(.)\1{2,}",  # Three or more repeated characters
        r"123456|abcdef|qwerty|password",  # Common sequences
    ]

    for pattern in common_patterns:
        if re.search(pattern, password.lower()):
            issues.append("Password contains common weak patterns")
            break

    return len(issues) == 0, issues


def mask_sensitive_data(data: str, mask_char: str = "*", visible_chars: int = 4) -> str:
    """Mask sensitive data for logging.

    Args:
        data: The sensitive data to mask.
        mask_char: Character to use for masking.
        visible_chars: Number of characters to keep visible at the end.

    Returns:
        str: The masked data.
    """
    if not data or len(data) <= visible_chars:
        return mask_char * 8

    visible_part = data[-visible_chars:]
    masked_part = mask_char * max(8, len(data) - visible_chars)
    return masked_part + visible_part
