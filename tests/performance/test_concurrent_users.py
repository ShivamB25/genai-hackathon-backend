import asyncio

import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.asyncio


async def test_concurrent_health_checks(async_client: AsyncClient):
    """
    Tests that the server can handle multiple concurrent requests.
    """
    num_requests = 50
    tasks = [async_client.get("/health") for _ in range(num_requests)]
    responses = await asyncio.gather(*tasks)

    for response in responses:
        assert response.status_code == 200
