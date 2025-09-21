import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.asyncio


async def test_health_check(async_client: AsyncClient):
    """
    Tests that the health check endpoint is working correctly.
    """
    response = await async_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
