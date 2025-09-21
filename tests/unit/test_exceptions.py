import pytest
from src.core.exceptions import (
    DatabaseException,
    NotFoundException,
    ValidationException,
    AuthenticationException,
    AuthorizationException,
    ExternalServiceException,
    AppException,
)

def test_app_exception():
    """Test the base application exception."""
    with pytest.raises(AppException) as excinfo:
        raise AppException(message="Generic error", status_code=500)
    
    assert excinfo.value.status_code == 500
    assert excinfo.value.message == "Generic error"
    assert excinfo.value.details is None

def test_database_exception():
    """Test the database exception."""
    with pytest.raises(DatabaseException) as excinfo:
        raise DatabaseException(message="DB error", details={"query": "SELECT *"})
    
    assert excinfo.value.status_code == 500
    assert excinfo.value.message == "DB error"
    assert excinfo.value.details == {"query": "SELECT *"}

def test_not_found_exception():
    """Test the not found exception."""
    with pytest.raises(NotFoundException) as excinfo:
        raise NotFoundException(resource="user", resource_id="123")
    
    assert excinfo.value.status_code == 404
    assert "not found" in excinfo.value.message
    assert excinfo.value.details == {"resource": "user", "resource_id": "123"}

def test_validation_exception():
    """Test the validation exception."""
    errors = {"field": "is required"}
    with pytest.raises(ValidationException) as excinfo:
        raise ValidationException(errors=errors)
    
    assert excinfo.value.status_code == 422
    assert excinfo.value.message == "Validation failed"
    assert excinfo.value.details == {"errors": errors}

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
        raise ExternalServiceException(service="Google Maps", details={"error": "API limit"})
    
    assert excinfo.value.status_code == 503
    assert "External service error" in excinfo.value.message
    assert excinfo.value.details == {"service": "Google Maps", "error": "API limit"}