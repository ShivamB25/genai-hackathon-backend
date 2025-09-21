from unittest.mock import AsyncMock, MagicMock

import pytest

from src.ai_services.gemini_agents import AgentCapabilities, AgentRole, GeminiAgent
from src.ai_services.model_config import AsyncVertexAIClient
from src.ai_services.prompt_templates import PromptType


@pytest.fixture
def mock_model_client():
    """Fixture for a mocked AsyncVertexAIClient."""
    return AsyncMock(spec=AsyncVertexAIClient)


@pytest.fixture
def agent_capabilities():
    """Fixture for AgentCapabilities."""
    return AgentCapabilities(
        role=AgentRole.TRIP_PLANNER, prompt_type=PromptType.TRIP_PLANNER
    )


@pytest.fixture
def gemini_agent(mock_model_client, agent_capabilities):
    """Fixture for a GeminiAgent with a mocked model."""
    agent = GeminiAgent(
        agent_id="test_agent",
        capabilities=agent_capabilities,
        model_client=mock_model_client,
    )
    return agent


@pytest.mark.asyncio
async def test_agent_initialization(agent_capabilities):
    """Test that the GeminiAgent initializes correctly."""
    agent = GeminiAgent(agent_id="test_agent", capabilities=agent_capabilities)
    assert agent.model_client is not None


@pytest.mark.asyncio
async def test_generate_response(gemini_agent, mock_model_client):
    """Test that the agent can generate a response."""
    mock_model_client.generate_content.return_value = "This is a test response."

    response = await gemini_agent.process_message("Test prompt")

    assert response == "This is a test response."
    mock_model_client.generate_content.assert_called_once()


@pytest.mark.asyncio
async def test_generate_response_with_history(gemini_agent, mock_model_client):
    """Test that the agent can generate a response with chat history."""
    mock_model_client.generate_content.return_value = "Follow-up response."

    gemini_agent.conversation_history = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]

    await gemini_agent.process_message("Another question")

    # The prompt sent to the model should include the history
    mock_model_client.generate_content.assert_called_once()
