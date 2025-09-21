from unittest.mock import AsyncMock

import pytest

from src.ai_services.agent_orchestrator import (
    AgentExecutionMode,
    WorkflowDefinition,
    WorkflowStep,
    WorkflowType,
    get_trip_planner_orchestrator,
)
from src.ai_services.gemini_agents import AgentRole, GeminiAgent


@pytest.fixture
def mock_agent_one():
    """Fixture for a mocked agent."""
    agent = AsyncMock(spec=GeminiAgent)
    agent.process_message.return_value = "Response from agent one."
    agent.capabilities.role = AgentRole.DESTINATION_EXPERT
    return agent


@pytest.fixture
def mock_agent_two():
    """Fixture for another mocked agent."""
    agent = AsyncMock(spec=GeminiAgent)
    agent.process_message.return_value = "Response from agent two."
    agent.capabilities.role = AgentRole.TRIP_PLANNER
    return agent


@pytest.fixture
def orchestrator(mock_agent_one, mock_agent_two):
    """Fixture for the AgentOrchestrator with mocked agents."""
    orchestrator = get_trip_planner_orchestrator()
    orchestrator.sequential_agent.add_agent(mock_agent_one)
    orchestrator.sequential_agent.add_agent(mock_agent_two)
    return orchestrator


@pytest.mark.asyncio
async def test_orchestrator_run_workflow(orchestrator, mock_agent_one, mock_agent_two):
    """Test a simple workflow execution."""

    workflow_def = WorkflowDefinition(
        name="Test Workflow",
        description="A test",
        workflow_type=WorkflowType.SEQUENTIAL,
        steps=[
            WorkflowStep(
                step_id="step1",
                agent_roles=[AgentRole.DESTINATION_EXPERT],
                execution_mode=AgentExecutionMode.SEQUENTIAL,
            ),
            WorkflowStep(
                step_id="step2",
                agent_roles=[AgentRole.TRIP_PLANNER],
                execution_mode=AgentExecutionMode.SEQUENTIAL,
                dependencies=["step1"],
            ),
        ],
    )

    initial_context = {"prompt": "Test workflow prompt"}

    execution = await orchestrator.execute_workflow(workflow_def, initial_context)

    assert execution.state.value == "completed"
    assert "Response from agent two." in str(execution.context)

    mock_agent_one.process_message.assert_called_once()
    mock_agent_two.process_message.assert_called_once()
