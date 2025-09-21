import pytest
from unittest.mock import patch, MagicMock
from src.auth.dependencies import get_current_user
from src.auth.firebase_auth import TokenVerificationError, UserNotFoundError
from fastapi import HTTPException

@pytest.mark.asyncio
async def test_get_current_user_success():
    """Test successful user authentication with a valid token."""
    mock_user_data = {"uid": "test_uid", "email": "test@example.com"}
    
    with patch("src.auth.dependencies.authenticate_user") as mock_authenticate:
        mock_authenticate.return_value = mock_user_data
        
        # The dependency is a function that takes a token
        result = await get_current_user(token="valid_token")
        
        assert result == mock_user_data
        mock_authenticate.assert_called_once_with("valid_token")

@pytest.mark.asyncio
async def test_get_current_user_no_token():
    """Test that an authentication error is raised when no token is provided."""
    with pytest.raises(HTTPException) as excinfo:
        await get_current_user(token=None)
    
    assert excinfo.value.status_code == 401
    assert "Authentication token required" in excinfo.value.detail

@pytest.mark.asyncio
async def test_get_current_user_invalid_token():
    """Test that an authentication error is raised for an invalid token."""
    with patch("src.auth.dependencies.authenticate_user") as mock_authenticate:
        mock_authenticate.side_effect = TokenVerificationError("Invalid token")
        
        with pytest.raises(HTTPException) as excinfo:
            await get_current_user(token="invalid_token")
            
        assert excinfo.value.status_code == 401
        assert "Invalid or expired token" in excinfo.value.detail

@pytest.mark.asyncio
async def test_get_current_user_not_found():
    """Test that an authentication error is raised when the user is not found."""
    with patch("src.auth.dependencies.authenticate_user") as mock_authenticate:
        mock_authenticate.side_effect = UserNotFoundError("User not found")
        
        with pytest.raises(HTTPException) as excinfo:
            await get_current_user(token="valid_token_for_nonexistent_user")
            
        assert excinfo.value.status_code == 401
        assert "User account not found" in excinfo.value.detail