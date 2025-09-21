import pytest

from src.core.exceptions import (
    AuthenticationException,
    AuthorizationException,
    DatabaseException,
    ExternalServiceException,
    TripPlannerBaseException,
    UserNotFoundException,
    ValidationException,
)


def test_app_exception():
    """Test the base application exception."""
    with pytest.raises(TripPlannerBaseException) as excinfo:
        raise TripPlannerBaseException(message="Generic error", status_code=500)

    assert excinfo.value.status_code == 500
    assert excinfo.value.message == "Generic error"
    assert excinfo.value.details == {}


def test_database_exception():
    """Test the database exception."""
    with pytest.raises(DatabaseException) as excinfo:
        raise DatabaseException(message="DB error", details={"query": "SELECT *"})

    assert excinfo.value.status_code == 503
    assert excinfo.value.message == "DB error"
    assert excinfo.value.details == {"query": "SELECT *"}


def test_not_found_exception():
    """Test the not found exception."""
    with pytest.raises(UserNotFoundException) as excinfo:
        raise UserNotFoundException(details={"user_id": "123"})

    assert excinfo.value.status_code == 404
    assert "User not found" in excinfo.value.message
    assert excinfo.value.details == {"user_id": "123"}


def test_validation_exception():
    """Test the validation exception."""
    errors = [{"field": "is required"}]
    with pytest.raises(ValidationException) as excinfo:
        raise ValidationException(errors=errors)

    assert excinfo.value.status_code == 422
    assert excinfo.value.message == "Validation failed"
    assert excinfo.value.details["validation_errors"] == errors


def test_authentication_exception():
    """Test the authentication exception."""
    with pytest.raises(AuthenticationException) as excinfo:
        raise AuthenticationException(message="Invalid token")

    assert excinfo.value.status_code == 401
    assert excinfo.value.message == "Invalid token"


def test_authorization_exception():
    """Test the authorization exception."""
    with pytest.raises(AuthorizationException) as excinfo:
        raise AuthorizationException(message="Permission denied")

    assert excinfo.value.status_code == 403
    assert excinfo.value.message == "Permission denied"


def test_external_service_exception():
    """Test the external service exception."""
    with pytest.raises(ExternalServiceException) as excinfo:
        raise ExternalServiceException(
            service="Google Maps",
            message="API limit",
            details={"error": "limit exceeded"},
        )

    assert excinfo.value.status_code == 503
    assert "Google Maps: API limit" in excinfo.value.message
    assert excinfo.value.details == {
        "service": "Google Maps",
        "error": "limit exceeded",
    }
