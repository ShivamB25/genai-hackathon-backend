import os
import time

import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.asyncio


async def test_health_check_performance(async_client: AsyncClient):
    """
    Tests the performance of the health check endpoint.
    """
    start_time = time.time()
    response = await async_client.get("/health")
    end_time = time.time()

    assert response.status_code == 200
    # Assert that the response time is less than the configurable threshold (default: 300ms)
    PERFORMANCE_THRESHOLD = float(
        os.getenv("HEALTH_CHECK_PERFORMANCE_THRESHOLD", "0.3")
    )
    assert (end_time - start_time) < PERFORMANCE_THRESHOLD
