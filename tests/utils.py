from datetime import datetime, timedelta, timezone

import jwt

SECRET_KEY = "test-secret-key-for-testing-only"


def generate_mock_jwt(user_id: str, email: str) -> str:
    """
    Generates a mock JWT token for testing purposes.
    """
    payload = {
        "uid": user_id,
        "email": email,
        "exp": datetime.now(timezone.utc) + timedelta(hours=1),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")
