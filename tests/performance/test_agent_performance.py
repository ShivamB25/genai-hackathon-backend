import os
import time
from unittest.mock import AsyncMock

import pytest

from src.ai_services.gemini_agents import AgentCapabilities, AgentRole, GeminiAgent
from src.ai_services.prompt_templates import PromptType

pytestmark = pytest.mark.asyncio


async def test_single_agent_performance(mock_vertex_ai_client: AsyncMock):
    """
    Tests the performance of a single AI agent.
    """
    capabilities = AgentCapabilities(
        role=AgentRole.TRIP_PLANNER,
        prompt_type=PromptType.TRIP_PLANNER,
    )
    agent = GeminiAgent(
        agent_id="test_agent",
        capabilities=capabilities,
        model_client=mock_vertex_ai_client,
    )

    start_time = time.time()
    await agent.process_message("Test content")
    end_time = time.time()

    # Assert that the agent's response time is less than the configurable threshold (default 500ms)
    threshold = float(os.getenv("AGENT_PERFORMANCE_THRESHOLD", "0.5"))
    assert (end_time - start_time) < threshold
