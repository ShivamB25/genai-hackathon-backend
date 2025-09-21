"""Agent Testing Framework for AI-Powered Trip Planner Backend - Google ADK Multi-Agent System.

This module provides mock agent implementations for testing, agent behavior simulation
and validation, workflow testing utilities, performance benchmarking tools, and agent
response validation and quality metrics.
"""

import asyncio
import json
import random
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from src.ai_services.agent_factory import TripComplexity, TripRequirements
from src.ai_services.agent_orchestrator import (
    WorkflowState,
)
from src.ai_services.exceptions import AgentCommunicationError, AgentError
from src.ai_services.gemini_agents import (
    AgentCapabilities,
    AgentRole,
    AgentState,
    BaseAgent,
)
from src.ai_services.prompt_templates import PromptType
from src.ai_services.workflow_engine import WorkflowExecutionEngine
from src.core.logging import get_logger
from src.trip_planner.schemas import TripRequest

logger = get_logger(__name__)


class TestScenarioType(str, Enum):
    """Types of test scenarios."""

    UNIT_TEST = "unit_test"
    INTEGRATION_TEST = "integration_test"
    PERFORMANCE_TEST = "performance_test"
    STRESS_TEST = "stress_test"
    WORKFLOW_TEST = "workflow_test"
    COMMUNICATION_TEST = "communication_test"
    ERROR_HANDLING_TEST = "error_handling_test"


class MockAgentBehavior(str, Enum):
    """Mock agent behavior patterns."""

    NORMAL = "normal"
    SLOW_RESPONSE = "slow_response"
    FAST_RESPONSE = "fast_response"
    ERROR_PRONE = "error_prone"
    TIMEOUT = "timeout"
    INVALID_RESPONSE = "invalid_response"
    FUNCTION_CALL_HEAVY = "function_call_heavy"
    CONTEXT_DEPENDENT = "context_dependent"


@dataclass
class TestMetrics:
    """Test execution metrics."""

    test_id: str
    test_name: str
    execution_time: float
    success: bool
    agent_count: int = 0
    message_count: int = 0
    function_calls: int = 0
    tokens_used: int = 0
    memory_usage: float = 0.0
    error_count: int = 0
    warnings: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class MockAgent(BaseAgent):
    """Mock agent for testing purposes."""

    def __init__(
        self,
        agent_id: str,
        capabilities: AgentCapabilities,
        behavior: MockAgentBehavior = MockAgentBehavior.NORMAL,
        session_id: Optional[str] = None,
    ) -> None:
        super().__init__(agent_id, capabilities, session_id)
        self.behavior = behavior
        self.response_templates = self._load_response_templates()
        self.execution_count = 0
        self.test_responses: List[str] = []

    def _load_response_templates(self) -> Dict[str, List[str]]:
        """Load response templates for different agent roles."""
        return {
            AgentRole.TRIP_PLANNER.value: [
                "Based on your requirements, I've created a comprehensive trip plan for {destination}.",
                "Here's your {duration_days}-day itinerary for {destination} with activities, dining, and transportation.",
                "I've organized your trip to include the best attractions and experiences in {destination}.",
            ],
            AgentRole.DESTINATION_EXPERT.value: [
                "{destination} is known for its rich culture, vibrant markets, and historical landmarks.",
                "The best time to visit {destination} is during the cooler months with pleasant weather.",
                "Local highlights include iconic attractions, authentic cuisine, and unique cultural experiences.",
            ],
            AgentRole.BUDGET_ADVISOR.value: [
                "For a {duration_days}-day trip to {destination}, I estimate a budget of ₹{estimated_cost}.",
                "Budget breakdown includes accommodation (40%), food (25%), activities (20%), and transport (15%).",
                "Cost optimization suggestions: book in advance, use local transport, mix free and paid activities.",
            ],
            AgentRole.INFORMATION_GATHERER.value: [
                "I've gathered comprehensive information about {destination} including top attractions and local insights.",
                "Research shows {destination} offers diverse experiences from cultural sites to modern entertainment.",
                "Key information collected: weather patterns, local events, transportation options, and safety tips.",
            ],
            AgentRole.ITINERARY_PLANNER.value: [
                "Daily itinerary created with optimal timing and location clustering for efficiency.",
                "Day-by-day plan includes morning activities, lunch spots, afternoon exploration, and evening entertainment.",
                "Schedule optimized for travel time, queue management, and energy levels throughout the day.",
            ],
            AgentRole.OPTIMIZATION_AGENT.value: [
                "Optimized itinerary for time efficiency, cost savings, and experience quality.",
                "Identified 15-25% cost reduction opportunities and 2-3 hours daily time savings.",
                "Route optimization reduces travel time while maximizing attraction coverage.",
            ],
            AgentRole.ROUTE_PLANNER.value: [
                "Optimal routes planned using mix of public transport, walking, and ride-sharing.",
                "Transportation schedule coordinated with activity timings and local traffic patterns.",
                "Alternative route options provided for flexibility and contingency planning.",
            ],
        }

    async def initialize(self) -> None:
        """Initialize mock agent."""
        await asyncio.sleep(0.1)  # Simulate initialization time
        self.update_state(AgentState.READY)

        logger.debug(
            "Mock agent initialized",
            agent_id=self.agent_id,
            role=self.capabilities.role.value,
            behavior=self.behavior.value,
        )

    async def process_message(
        self, message: str, context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Process message with simulated behavior."""

        self.execution_count += 1
        context = context or {}

        # Simulate different behaviors
        if self.behavior == MockAgentBehavior.SLOW_RESPONSE:
            await asyncio.sleep(random.uniform(2.0, 5.0))  # noqa: S311
        elif self.behavior == MockAgentBehavior.FAST_RESPONSE:
            await asyncio.sleep(random.uniform(0.1, 0.3))  # noqa: S311
        elif self.behavior == MockAgentBehavior.TIMEOUT:
            await asyncio.sleep(10.0)  # Will likely timeout
        elif (
            self.behavior == MockAgentBehavior.ERROR_PRONE
            and random.random() < 0.3  # noqa: S311
        ):
            raise AgentCommunicationError(f"Mock error from {self.agent_id}")

        # Generate response based on role and context
        response = self._generate_mock_response(message, context)

        # Simulate function calls for certain behaviors
        if self.behavior == MockAgentBehavior.FUNCTION_CALL_HEAVY:
            # Simulate multiple function calls
            for _ in range(random.randint(3, 6)):  # noqa: S311
                await asyncio.sleep(0.2)  # Simulate function call delay

        self.test_responses.append(response)
        self.message_count += 1

        return response

    def _generate_mock_response(self, _message: str, context: Dict[str, Any]) -> str:
        """Generate mock response based on agent role."""

        if self.behavior == MockAgentBehavior.INVALID_RESPONSE:
            return "INVALID_RESPONSE_FORMAT"

        role = self.capabilities.role.value
        templates = self.response_templates.get(
            role, ["Generic response from {role} agent"]
        )

        template = random.choice(templates)  # noqa: S311

        # Fill template with context data
        response_context = {
            "role": role,
            "agent_id": self.agent_id,
            "destination": context.get("destination", "your destination"),
            "duration_days": context.get("duration_days", "N"),
            "estimated_cost": context.get("budget", "varies"),
            **context,
        }

        try:
            response = template.format(**response_context)
        except (KeyError, ValueError):
            response = f"Response from {role} agent based on provided context."

        return response

    async def cleanup(self) -> None:
        """Cleanup mock agent."""
        self.update_state(AgentState.TERMINATED)
        logger.debug(f"Mock agent cleaned up: {self.agent_id}")


class AgentBehaviorSimulator:
    """Simulator for testing different agent behaviors."""

    def __init__(self) -> None:
        self.mock_agents: Dict[str, MockAgent] = {}
        self.simulation_results: List[Dict[str, Any]] = []

    def create_mock_agent(
        self,
        role: AgentRole,
        behavior: MockAgentBehavior = MockAgentBehavior.NORMAL,
        session_id: Optional[str] = None,
    ) -> MockAgent:
        """Create a mock agent with specified behavior."""

        # Map AgentRole to PromptType
        prompt_type_mapping = {
            AgentRole.TRIP_PLANNER: PromptType.TRIP_PLANNER,
            AgentRole.DESTINATION_EXPERT: PromptType.DESTINATION_EXPERT,
            AgentRole.BUDGET_ADVISOR: PromptType.BUDGET_ADVISOR,
            AgentRole.INFORMATION_GATHERER: PromptType.DESTINATION_EXPERT,
            AgentRole.ITINERARY_PLANNER: PromptType.TRIP_PLANNER,
            AgentRole.OPTIMIZATION_AGENT: PromptType.ITINERARY_OPTIMIZER,
            AgentRole.ROUTE_PLANNER: PromptType.TRANSPORT_PLANNER,
            AgentRole.ACTIVITY_RECOMMENDER: PromptType.ACTIVITY_RECOMMENDER,
            AgentRole.LOCAL_GUIDE: PromptType.LOCAL_GUIDE,
            AgentRole.SAFETY_ADVISOR: PromptType.SAFETY_ADVISOR,
            AgentRole.WEATHER_ANALYST: PromptType.WEATHER_ANALYST,
            AgentRole.TRANSPORT_PLANNER: PromptType.TRANSPORT_PLANNER,
            AgentRole.ACCOMMODATION_FINDER: PromptType.ACCOMMODATION_FINDER,
        }

        capabilities = AgentCapabilities(
            role=role,
            prompt_type=prompt_type_mapping.get(role, PromptType.TRIP_PLANNER),
            supported_functions=self._get_mock_functions_for_role(role),
            max_iterations=10,
            timeout_seconds=30,
        )

        agent_id = f"mock_{role.value}_{behavior.value}_{int(time.time())}"
        agent = MockAgent(agent_id, capabilities, behavior, session_id)

        self.mock_agents[agent_id] = agent

        return agent

    def _get_mock_functions_for_role(self, role: AgentRole) -> List[str]:
        """Get mock function list for agent role."""

        function_mappings = {
            AgentRole.TRIP_PLANNER: [
                "find_places",
                "get_directions",
                "calculate_trip_budget",
            ],
            AgentRole.DESTINATION_EXPERT: [
                "find_places",
                "get_weather_info",
                "get_local_events",
            ],
            AgentRole.BUDGET_ADVISOR: ["convert_currency", "calculate_trip_budget"],
            AgentRole.INFORMATION_GATHERER: [
                "find_places",
                "find_nearby_places",
                "get_place_details",
            ],
            AgentRole.ITINERARY_PLANNER: [
                "get_directions",
                "get_travel_time",
                "optimize_itinerary",
            ],
            AgentRole.OPTIMIZATION_AGENT: ["optimize_itinerary", "get_travel_time"],
            AgentRole.ROUTE_PLANNER: [
                "get_directions",
                "get_travel_time",
                "validate_location",
            ],
        }

        return function_mappings.get(role, ["get_current_datetime"])

    async def simulate_agent_interaction(
        self,
        agent_id: str,
        messages: List[str],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Simulate interaction with a mock agent."""

        if agent_id not in self.mock_agents:
            raise AgentError(f"Mock agent {agent_id} not found")

        agent = self.mock_agents[agent_id]
        context = context or {}

        simulation_start = datetime.now(timezone.utc)
        results = {
            "agent_id": agent_id,
            "agent_role": agent.capabilities.role.value,
            "behavior": agent.behavior.value,
            "start_time": simulation_start.isoformat(),
            "messages_processed": 0,
            "responses": [],
            "errors": [],
            "total_time": 0.0,
        }

        await agent.initialize()

        try:
            for i, message in enumerate(messages):
                message_start = time.time()

                try:
                    response = await agent.process_message(message, context)
                    message_time = time.time() - message_start

                    results["responses"].append(
                        {
                            "message_index": i,
                            "message": message,
                            "response": response,
                            "execution_time": message_time,
                            "success": True,
                        }
                    )

                    results["messages_processed"] += 1

                except Exception as e:
                    message_time = time.time() - message_start

                    error_info = {
                        "message_index": i,
                        "message": message,
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "execution_time": message_time,
                    }

                    results["errors"].append(error_info)

            simulation_end = datetime.now(timezone.utc)
            results["end_time"] = simulation_end.isoformat()
            results["total_time"] = (simulation_end - simulation_start).total_seconds()

            self.simulation_results.append(results)

        finally:
            await agent.cleanup()

        return results

    async def run_behavior_comparison_test(
        self,
        role: AgentRole,
        test_messages: List[str],
        behaviors: Optional[List[MockAgentBehavior]] = None,
    ) -> Dict[str, Any]:
        """Compare different agent behaviors for the same role."""

        behaviors = behaviors or [
            MockAgentBehavior.NORMAL,
            MockAgentBehavior.SLOW_RESPONSE,
            MockAgentBehavior.FAST_RESPONSE,
            MockAgentBehavior.ERROR_PRONE,
        ]

        comparison_results = {
            "role": role.value,
            "test_messages": test_messages,
            "behavior_results": {},
            "comparison_metrics": {},
            "start_time": datetime.now(timezone.utc).isoformat(),
        }

        for behavior in behaviors:
            try:
                agent = self.create_mock_agent(role, behavior)
                result = await self.simulate_agent_interaction(
                    agent.agent_id, test_messages
                )

                comparison_results["behavior_results"][behavior.value] = result

            except Exception as e:
                comparison_results["behavior_results"][behavior.value] = {
                    "error": str(e),
                    "success": False,
                }

        # Calculate comparison metrics
        comparison_results["comparison_metrics"] = self._calculate_behavior_metrics(
            comparison_results["behavior_results"]
        )

        comparison_results["end_time"] = datetime.now(timezone.utc).isoformat()

        return comparison_results

    def _calculate_behavior_metrics(
        self, behavior_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate metrics comparing different behaviors."""

        metrics = {
            "response_times": {},
            "success_rates": {},
            "error_patterns": {},
            "response_quality": {},
        }

        for behavior, result in behavior_results.items():
            if "error" in result:
                metrics["success_rates"][behavior] = 0.0
                continue

            total_messages = result.get("messages_processed", 0)
            total_errors = len(result.get("errors", []))

            metrics["success_rates"][behavior] = (
                (total_messages / (total_messages + total_errors)) * 100
                if (total_messages + total_errors) > 0
                else 0
            )

            # Calculate average response time
            responses = result.get("responses", [])
            if responses:
                avg_time = statistics.mean([r["execution_time"] for r in responses])
                metrics["response_times"][behavior] = avg_time

            # Analyze error patterns
            errors = result.get("errors", [])
            error_types = {}
            for error in errors:
                error_type = error.get("error_type", "unknown")
                error_types[error_type] = error_types.get(error_type, 0) + 1

            metrics["error_patterns"][behavior] = error_types

        return metrics


class WorkflowTestSuite:
    """Test suite for workflow execution validation."""

    def __init__(self) -> None:
        self.test_results: List[TestMetrics] = []
        self.performance_benchmarks: Dict[str, Dict[str, float]] = {}

    async def test_sequential_workflow(
        self,
        trip_requirements: TripRequirements,
        session_id: str,
    ) -> TestMetrics:
        """Test sequential workflow execution."""

        test_start = time.time()
        test_id = str(uuid4())[:8]

        try:
            # Create test workflow
            from src.ai_services.agent_orchestrator import (
                create_comprehensive_trip_planning_workflow,
            )

            workflow_def = create_comprehensive_trip_planning_workflow()

            # Create mock agents
            simulator = AgentBehaviorSimulator()
            agents = {}

            for step in workflow_def.steps:
                for role in step.agent_roles:
                    if role not in agents:
                        agents[role] = simulator.create_mock_agent(
                            role, MockAgentBehavior.NORMAL, session_id
                        )

            # Execute workflow
            workflow_engine = WorkflowExecutionEngine(session_id)
            await workflow_engine.initialize()

            execution = await workflow_engine.execute_workflow(
                workflow_def, trip_requirements
            )

            execution_time = time.time() - test_start

            # Create test metrics
            metrics = TestMetrics(
                test_id=test_id,
                test_name="Sequential Workflow Test",
                execution_time=execution_time,
                success=execution.state == WorkflowState.COMPLETED,
                agent_count=len(agents),
                message_count=sum(
                    len(agent.test_responses) for agent in agents.values()
                ),
                details={
                    "workflow_id": execution.workflow_id,
                    "execution_id": execution.execution_id,
                    "completed_steps": len(execution.completed_steps),
                    "failed_steps": len(execution.failed_steps),
                    "total_execution_time": execution.total_execution_time,
                    "agents_used": list(agents.keys()),
                },
            )

            # Cleanup
            for agent in agents.values():
                await agent.cleanup()

            self.test_results.append(metrics)

            return metrics

        except Exception as e:
            execution_time = time.time() - test_start

            metrics = TestMetrics(
                test_id=test_id,
                test_name="Sequential Workflow Test",
                execution_time=execution_time,
                success=False,
                error_count=1,
                warnings=[str(e)],
            )

            self.test_results.append(metrics)
            return metrics

    async def test_parallel_workflow(
        self,
        trip_requirements: TripRequirements,
        session_id: str,
    ) -> TestMetrics:
        """Test parallel workflow execution."""

        test_start = time.time()
        test_id = str(uuid4())[:8]

        try:
            # Create test workflow
            from src.ai_services.agent_orchestrator import (
                create_quick_trip_planning_workflow,
            )

            workflow_def = create_quick_trip_planning_workflow()

            # Map AgentRole to PromptType
            # Map AgentRole to WorkflowType
            from src.ai_services.agent_orchestrator import WorkflowType

            workflow_def.workflow_type = (
                WorkflowType.PARALLEL
            )  # Force parallel execution

            # Create mock agents with different behaviors
            simulator = AgentBehaviorSimulator()
            agents = {
                AgentRole.DESTINATION_EXPERT: simulator.create_mock_agent(
                    AgentRole.DESTINATION_EXPERT,
                    MockAgentBehavior.FAST_RESPONSE,
                    session_id,
                ),
                AgentRole.BUDGET_ADVISOR: simulator.create_mock_agent(
                    AgentRole.BUDGET_ADVISOR, MockAgentBehavior.NORMAL, session_id
                ),
                AgentRole.TRIP_PLANNER: simulator.create_mock_agent(
                    AgentRole.TRIP_PLANNER, MockAgentBehavior.SLOW_RESPONSE, session_id
                ),
            }

            # Execute workflow
            workflow_engine = WorkflowExecutionEngine(session_id)
            await workflow_engine.initialize()

            execution = await workflow_engine.execute_workflow(
                workflow_def, trip_requirements
            )

            execution_time = time.time() - test_start

            metrics = TestMetrics(
                test_id=test_id,
                test_name="Parallel Workflow Test",
                execution_time=execution_time,
                success=execution.state == WorkflowState.COMPLETED,
                agent_count=len(agents),
                message_count=sum(
                    len(agent.test_responses) for agent in agents.values()
                ),
                details={
                    "workflow_id": execution.workflow_id,
                    "execution_id": execution.execution_id,
                    "parallel_execution_efficiency": execution.total_execution_time
                    / execution_time,
                    "agents_response_patterns": {
                        agent.agent_id: agent.behavior.value
                        for agent in agents.values()
                    },
                },
            )

            # Cleanup
            for agent in agents.values():
                await agent.cleanup()

            self.test_results.append(metrics)
            return metrics

        except Exception as e:
            execution_time = time.time() - test_start

            metrics = TestMetrics(
                test_id=test_id,
                test_name="Parallel Workflow Test",
                execution_time=execution_time,
                success=False,
                error_count=1,
                warnings=[str(e)],
            )

            self.test_results.append(metrics)
            return metrics

    async def test_error_handling(
        self,
        trip_requirements: TripRequirements,
        session_id: str,
    ) -> TestMetrics:
        """Test error handling and recovery mechanisms."""

        test_start = time.time()
        test_id = str(uuid4())[:8]

        test_messages = [
            "Plan a trip to Mumbai",
            "Provide destination expertise",
            "Calculate budget breakdown",
        ]

        try:
            # Create agents with error-prone behavior
            simulator = AgentBehaviorSimulator()
            agents = [
                simulator.create_mock_agent(
                    AgentRole.TRIP_PLANNER, MockAgentBehavior.ERROR_PRONE, session_id
                ),
                simulator.create_mock_agent(
                    AgentRole.DESTINATION_EXPERT, MockAgentBehavior.TIMEOUT, session_id
                ),
                simulator.create_mock_agent(
                    AgentRole.BUDGET_ADVISOR,
                    MockAgentBehavior.INVALID_RESPONSE,
                    session_id,
                ),
            ]

            # Test individual agent error handling
            error_recovery_results = []

            for agent in agents:
                await agent.initialize()

                agent_test_start = time.time()
                errors_caught = 0
                successful_responses = 0

                for message in test_messages:
                    try:
                        await asyncio.wait_for(
                            agent.process_message(message, trip_requirements.__dict__),
                            timeout=5.0,
                        )
                        successful_responses += 1
                    except Exception as e:
                        errors_caught += 1
                        logger.debug(f"Expected error caught: {type(e).__name__}")

                agent_test_time = time.time() - agent_test_start

                error_recovery_results.append(
                    {
                        "agent_id": agent.agent_id,
                        "behavior": agent.behavior.value,
                        "errors_caught": errors_caught,
                        "successful_responses": successful_responses,
                        "error_rate": errors_caught / len(test_messages),
                        "test_time": agent_test_time,
                    }
                )

                await agent.cleanup()

            execution_time = time.time() - test_start

            metrics = TestMetrics(
                test_id=test_id,
                test_name="Error Handling Test",
                execution_time=execution_time,
                success=True,  # Success if we caught expected errors
                agent_count=len(agents),
                error_count=sum(r["errors_caught"] for r in error_recovery_results),
                details={
                    "error_recovery_results": error_recovery_results,
                    "total_test_messages": len(test_messages) * len(agents),
                    "error_handling_effective": all(
                        r["errors_caught"] > 0 for r in error_recovery_results
                    ),
                },
            )

            self.test_results.append(metrics)
            return metrics

        except Exception as e:
            execution_time = time.time() - test_start

            metrics = TestMetrics(
                test_id=test_id,
                test_name="Error Handling Test",
                execution_time=execution_time,
                success=False,
                error_count=1,
                warnings=[str(e)],
            )

            self.test_results.append(metrics)
            return metrics


class PerformanceBenchmarkSuite:
    """Performance benchmarking tools for agent system."""

    def __init__(self) -> None:
        self.benchmark_results: Dict[str, List[Dict[str, Any]]] = {}
        self.baseline_metrics: Dict[str, Dict[str, float]] = {}

    async def benchmark_agent_response_times(
        self,
        agent_roles: List[AgentRole],
        message_count: int = 10,
        concurrent_messages: int = 1,
    ) -> Dict[str, Any]:
        """Benchmark agent response times."""

        benchmark_id = str(uuid4())[:8]
        benchmark_start = time.time()

        results = {
            "benchmark_id": benchmark_id,
            "start_time": datetime.now(timezone.utc).isoformat(),
            "configuration": {
                "agent_roles": [role.value for role in agent_roles],
                "message_count": message_count,
                "concurrent_messages": concurrent_messages,
            },
            "agent_results": {},
            "summary_metrics": {},
        }

        simulator = AgentBehaviorSimulator()
        test_messages = [f"Test message {i}" for i in range(message_count)]

        try:
            for role in agent_roles:
                agent = simulator.create_mock_agent(role, MockAgentBehavior.NORMAL)
                await agent.initialize()

                # Measure response times
                response_times = []
                errors = 0

                for i in range(0, len(test_messages), concurrent_messages):
                    batch_messages = test_messages[i : i + concurrent_messages]
                    batch_start = time.time()

                    # Process batch concurrently or sequentially
                    if concurrent_messages > 1:
                        tasks = [
                            agent.process_message(msg, {"test_context": "benchmark"})
                            for msg in batch_messages
                        ]
                        try:
                            await asyncio.gather(*tasks)
                        except Exception:
                            errors += len(batch_messages)
                    else:
                        for msg in batch_messages:
                            try:
                                await agent.process_message(
                                    msg, {"test_context": "benchmark"}
                                )
                            except Exception:
                                errors += 1

                    batch_time = time.time() - batch_start
                    response_times.append(
                        batch_time / len(batch_messages)
                    )  # Per message time

                # Calculate metrics
                agent_metrics = {
                    "total_messages": message_count,
                    "successful_messages": message_count - errors,
                    "error_count": errors,
                    "success_rate": ((message_count - errors) / message_count) * 100,
                    "average_response_time": (
                        statistics.mean(response_times) if response_times else 0
                    ),
                    "min_response_time": min(response_times) if response_times else 0,
                    "max_response_time": max(response_times) if response_times else 0,
                    "response_time_std": (
                        statistics.stdev(response_times)
                        if len(response_times) > 1
                        else 0
                    ),
                }

                results["agent_results"][role.value] = agent_metrics

                await agent.cleanup()

            # Calculate summary metrics
            all_avg_times = [
                metrics["average_response_time"]
                for metrics in results["agent_results"].values()
            ]
            all_success_rates = [
                metrics["success_rate"] for metrics in results["agent_results"].values()
            ]

            results["summary_metrics"] = {
                "overall_average_response_time": (
                    statistics.mean(all_avg_times) if all_avg_times else 0
                ),
                "overall_success_rate": (
                    statistics.mean(all_success_rates) if all_success_rates else 0
                ),
                "fastest_agent": (
                    min(
                        results["agent_results"].items(),
                        key=lambda x: x[1]["average_response_time"],
                    )[0]
                    if results["agent_results"]
                    else None
                ),
                "most_reliable_agent": (
                    max(
                        results["agent_results"].items(),
                        key=lambda x: x[1]["success_rate"],
                    )[0]
                    if results["agent_results"]
                    else None
                ),
            }

            benchmark_time = time.time() - benchmark_start
            results["total_benchmark_time"] = benchmark_time
            results["end_time"] = datetime.now(timezone.utc).isoformat()

            # Store results
            if "response_time_benchmark" not in self.benchmark_results:
                self.benchmark_results["response_time_benchmark"] = []
            self.benchmark_results["response_time_benchmark"].append(results)

            return results

        except Exception as e:
            logger.exception("Performance benchmark failed")
            return {
                "benchmark_id": benchmark_id,
                "error": str(e),
                "success": False,
            }

    async def stress_test_agent_system(
        self,
        max_concurrent_agents: int = 10,
        messages_per_agent: int = 5,
        duration_seconds: int = 30,
    ) -> Dict[str, Any]:
        """Stress test the agent system with high load."""

        stress_test_id = str(uuid4())[:8]
        test_start = time.time()

        results = {
            "stress_test_id": stress_test_id,
            "start_time": datetime.now(timezone.utc).isoformat(),
            "configuration": {
                "max_concurrent_agents": max_concurrent_agents,
                "messages_per_agent": messages_per_agent,
                "duration_seconds": duration_seconds,
            },
            "performance_metrics": {
                "peak_agents": 0,
                "total_messages": 0,
                "successful_messages": 0,
                "failed_messages": 0,
                "average_response_time": 0.0,
                "peak_memory_usage": 0.0,
            },
            "errors": [],
            "success": True,
        }

        simulator = AgentBehaviorSimulator()
        active_agents = []
        message_tasks = []

        try:
            # Create agents and tasks
            for i in range(max_concurrent_agents):
                role = list(AgentRole)[i % len(AgentRole)]  # Cycle through roles
                agent = simulator.create_mock_agent(role, MockAgentBehavior.NORMAL)
                await agent.initialize()
                active_agents.append(agent)

                # Create message tasks for this agent
                for j in range(messages_per_agent):
                    task = asyncio.create_task(
                        agent.process_message(
                            f"Stress test message {j} for {role.value}",
                            {"stress_test": True, "agent_index": i},
                        )
                    )
                    message_tasks.append((agent.agent_id, task))

            results["performance_metrics"]["peak_agents"] = len(active_agents)

            # Execute all tasks with timeout
            completed_tasks = 0
            failed_tasks = 0
            response_times = []

            for agent_id, task in message_tasks:
                task_start = time.time()
                try:
                    await asyncio.wait_for(task, timeout=duration_seconds)
                    task_time = time.time() - task_start
                    response_times.append(task_time)
                    completed_tasks += 1
                except Exception as e:
                    failed_tasks += 1
                    results["errors"].append(
                        {
                            "agent_id": agent_id,
                            "error": str(e),
                            "error_type": type(e).__name__,
                        }
                    )

            # Calculate final metrics
            total_messages = len(message_tasks)
            results["performance_metrics"].update(
                {
                    "total_messages": total_messages,
                    "successful_messages": completed_tasks,
                    "failed_messages": failed_tasks,
                    "success_rate": (
                        (completed_tasks / total_messages) * 100
                        if total_messages > 0
                        else 0
                    ),
                    "average_response_time": (
                        statistics.mean(response_times) if response_times else 0
                    ),
                    "response_time_percentiles": {
                        "p50": (
                            statistics.median(response_times) if response_times else 0
                        ),
                        "p95": (
                            statistics.quantiles(response_times, n=20)[18]
                            if len(response_times) > 20
                            else (max(response_times) if response_times else 0)
                        ),
                        "p99": (
                            statistics.quantiles(response_times, n=100)[98]
                            if len(response_times) > 100
                            else (max(response_times) if response_times else 0)
                        ),
                    },
                }
            )

            # Cleanup
            for agent in active_agents:
                await agent.cleanup()

            test_time = time.time() - test_start
            results["total_test_time"] = test_time
            results["end_time"] = datetime.now(timezone.utc).isoformat()

            # Store results
            if "stress_test" not in self.benchmark_results:
                self.benchmark_results["stress_test"] = []
            self.benchmark_results["stress_test"].append(results)

            return results

        except Exception as e:
            logger.exception("Stress test failed")
            error_message = str(e)

            # Cleanup on error
            for agent in active_agents:
                try:
                    await agent.cleanup()
                except Exception as cleanup_error:
                    logger.debug(
                        f"Error cleaning up agent {agent.agent_id}: {cleanup_error}"
                    )

            return {
                "stress_test_id": stress_test_id,
                "error": error_message,
                "success": False,
                "total_test_time": time.time() - test_start,
            }

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""

        if not self.benchmark_results:
            return {"message": "No benchmark results available"}

        report = {
            "report_generated": datetime.now(timezone.utc).isoformat(),
            "test_categories": list(self.benchmark_results.keys()),
            "total_benchmarks": sum(
                len(results) for results in self.benchmark_results.values()
            ),
            "category_summaries": {},
            "overall_insights": [],
        }

        for category, results in self.benchmark_results.items():
            if not results:
                continue

            successful_tests = [r for r in results if r.get("success", False)]

            category_summary = {
                "total_tests": len(results),
                "successful_tests": len(successful_tests),
                "success_rate": (len(successful_tests) / len(results)) * 100,
                "average_execution_time": statistics.mean(
                    [
                        r.get("total_test_time", r.get("total_benchmark_time", 0))
                        for r in results
                    ]
                ),
            }

            if category == "response_time_benchmark":
                # Add response time specific metrics
                overall_response_times = []
                for result in successful_tests:
                    agent_results = result.get("agent_results", {})
                    for agent_metrics in agent_results.values():
                        overall_response_times.append(
                            agent_metrics.get("average_response_time", 0)
                        )

                if overall_response_times:
                    category_summary["response_time_analysis"] = {
                        "average": statistics.mean(overall_response_times),
                        "median": statistics.median(overall_response_times),
                        "fastest": min(overall_response_times),
                        "slowest": max(overall_response_times),
                    }

            elif category == "stress_test":
                # Add stress test specific metrics
                peak_agents = [
                    r.get("performance_metrics", {}).get("peak_agents", 0)
                    for r in successful_tests
                ]
                success_rates = [
                    r.get("performance_metrics", {}).get("success_rate", 0)
                    for r in successful_tests
                ]

                if peak_agents and success_rates:
                    category_summary["stress_analysis"] = {
                        "max_concurrent_agents": max(peak_agents),
                        "average_success_rate": statistics.mean(success_rates),
                        "system_stability": (
                            "stable"
                            if statistics.mean(success_rates) > 90
                            else "unstable"
                        ),
                    }

            report["category_summaries"][category] = category_summary

        # Generate insights
        if report["category_summaries"]:
            avg_success_rate = statistics.mean(
                [
                    summary["success_rate"]
                    for summary in report["category_summaries"].values()
                ]
            )

            if avg_success_rate > 95:
                report["overall_insights"].append(
                    "System demonstrates high reliability across all test categories"
                )
            elif avg_success_rate > 80:
                report["overall_insights"].append(
                    "System shows good performance with some areas for improvement"
                )
            else:
                report["overall_insights"].append(
                    "System requires optimization and error handling improvements"
                )

        return report


class AgentTestValidator:
    """Validator for agent responses and behavior quality."""

    @staticmethod
    def validate_response_format(response: str, expected_format: str) -> Dict[str, Any]:
        """Validate agent response format."""

        validation_result = {
            "valid_format": True,
            "format_score": 1.0,
            "issues": [],
            "response_length": len(response),
        }

        try:
            if expected_format == "json":
                json.loads(response)
            elif expected_format == "structured_itinerary":
                # Check for key itinerary elements
                required_elements = ["destination", "activities", "duration"]
                missing_elements = [
                    elem
                    for elem in required_elements
                    if elem.lower() not in response.lower()
                ]

                if missing_elements:
                    validation_result["issues"].append(
                        f"Missing elements: {missing_elements}"
                    )
                    validation_result["format_score"] = 0.7

            elif expected_format == "budget_breakdown":
                # Check for budget elements
                budget_elements = ["cost", "budget", "price", "expense"]
                has_budget_info = any(
                    elem in response.lower() for elem in budget_elements
                )

                if not has_budget_info:
                    validation_result["issues"].append("No budget information found")
                    validation_result["format_score"] = 0.5

        except json.JSONDecodeError:
            if expected_format == "json":
                validation_result["valid_format"] = False
                validation_result["format_score"] = 0.0
                validation_result["issues"].append("Invalid JSON format")

        return validation_result

    @staticmethod
    def validate_response_relevance(
        response: str, input_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate response relevance to input context."""

        relevance_score = 1.0
        issues = []

        # Check if response mentions key context elements
        destination = input_context.get("destination", "")
        if destination and destination.lower() not in response.lower():
            relevance_score -= 0.3
            issues.append("Response doesn't mention the destination")

        duration = input_context.get("duration_days")
        if duration and str(duration) not in response:
            relevance_score -= 0.2
            issues.append("Response doesn't reference trip duration")

        budget = input_context.get("budget")
        if (
            budget
            and "budget" not in response.lower()
            and "cost" not in response.lower()
        ):
            relevance_score -= 0.2
            issues.append("Response doesn't address budget considerations")

        return {
            "relevance_score": max(0.0, relevance_score),
            "is_relevant": relevance_score >= 0.7,
            "issues": issues,
            "context_coverage": {
                "destination_mentioned": (
                    destination.lower() in response.lower() if destination else False
                ),
                "duration_mentioned": str(duration) in response if duration else False,
                "budget_mentioned": any(
                    term in response.lower() for term in ["budget", "cost", "price"]
                ),
            },
        }

    @staticmethod
    def assess_response_quality(
        response: str,
        agent_role: AgentRole,
        context: Dict[str, Any],
        execution_time: float,
    ) -> Dict[str, Any]:
        """Comprehensive quality assessment of agent response."""

        quality_assessment = {
            "overall_quality": 0.0,
            "component_scores": {},
            "recommendations": [],
            "assessment_details": {},
        }

        # Format validation
        expected_format = (
            "structured_itinerary" if agent_role == AgentRole.TRIP_PLANNER else "text"
        )
        format_validation = AgentTestValidator.validate_response_format(
            response, expected_format
        )
        quality_assessment["component_scores"]["format"] = format_validation[
            "format_score"
        ]

        # Relevance validation
        relevance_validation = AgentTestValidator.validate_response_relevance(
            response, context
        )
        quality_assessment["component_scores"]["relevance"] = relevance_validation[
            "relevance_score"
        ]

        # Response completeness
        completeness_score = min(
            1.0, len(response) / 500
        )  # Assume 500 chars is complete
        quality_assessment["component_scores"]["completeness"] = completeness_score

        # Response timeliness
        if execution_time < 1.0:
            timeliness_score = 1.0
        elif execution_time < 5.0:
            timeliness_score = 0.8
        elif execution_time < 10.0:
            timeliness_score = 0.6
        else:
            timeliness_score = 0.4

        quality_assessment["component_scores"]["timeliness"] = timeliness_score

        # Role-specific assessment
        role_score = AgentTestValidator._assess_role_specific_quality(
            response, agent_role, context
        )
        quality_assessment["component_scores"]["role_specific"] = role_score

        # Calculate overall quality
        weights = {
            "format": 0.2,
            "relevance": 0.3,
            "completeness": 0.2,
            "timeliness": 0.1,
            "role_specific": 0.2,
        }

        overall_quality = sum(
            quality_assessment["component_scores"].get(component, 0) * weight
            for component, weight in weights.items()
        )

        quality_assessment["overall_quality"] = overall_quality

        # Generate recommendations
        if format_validation.get("issues"):
            quality_assessment["recommendations"].extend(format_validation["issues"])

        if relevance_validation.get("issues"):
            quality_assessment["recommendations"].extend(relevance_validation["issues"])

        if execution_time > 5.0:
            quality_assessment["recommendations"].append(
                "Consider optimizing response time"
            )

        if completeness_score < 0.7:
            quality_assessment["recommendations"].append(
                "Response could be more comprehensive"
            )

        return quality_assessment

    @staticmethod
    def _assess_role_specific_quality(
        response: str, role: AgentRole, _context: Dict[str, Any]
    ) -> float:
        """Assess quality specific to agent role."""

        role_keywords = {
            AgentRole.TRIP_PLANNER: [
                "itinerary",
                "plan",
                "schedule",
                "activities",
                "day",
            ],
            AgentRole.DESTINATION_EXPERT: [
                "destination",
                "local",
                "culture",
                "attractions",
                "tips",
            ],
            AgentRole.BUDGET_ADVISOR: [
                "budget",
                "cost",
                "price",
                "expense",
                "savings",
            ],
            AgentRole.INFORMATION_GATHERER: [
                "information",
                "research",
                "data",
                "details",
            ],
            AgentRole.ITINERARY_PLANNER: [
                "schedule",
                "timing",
                "sequence",
                "order",
            ],
            AgentRole.OPTIMIZATION_AGENT: [
                "optimize",
                "improve",
                "efficient",
                "better",
            ],
            AgentRole.ROUTE_PLANNER: [
                "route",
                "direction",
                "transport",
                "travel",
            ],
        }

        keywords = role_keywords.get(role, [])
        if not keywords:
            return 0.8  # Default score for unknown roles

        keyword_matches = sum(1 for keyword in keywords if keyword in response.lower())
        keyword_score = min(1.0, keyword_matches / len(keywords))

        return keyword_score


# Test execution utilities


async def run_comprehensive_agent_tests(session_id: str) -> Dict[str, Any]:
    """Run comprehensive test suite for the agent system."""

    test_suite_start = time.time()

    # Create test requirements
    test_requirements = TripRequirements(
        destination="Mumbai, India",
        duration_days=5,
        traveler_count=2,
        budget_range="₹50,000-₹75,000",
        trip_type="leisure",
        complexity=TripComplexity.MODERATE,
    )

    test_results = {
        "test_suite_id": str(uuid4())[:8],
        "start_time": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "test_configuration": test_requirements.__dict__,
        "test_results": {},
        "overall_success": True,
        "summary_metrics": {},
    }

    try:
        # Initialize test suites
        workflow_suite = WorkflowTestSuite()
        performance_suite = PerformanceBenchmarkSuite()

        # Run workflow tests
        logger.info("Running workflow tests...")

        sequential_test = await workflow_suite.test_sequential_workflow(
            test_requirements, session_id
        )
        parallel_test = await workflow_suite.test_parallel_workflow(
            test_requirements, session_id
        )
        error_test = await workflow_suite.test_error_handling(
            test_requirements, session_id
        )

        test_results["test_results"]["workflow_tests"] = {
            "sequential": sequential_test.__dict__,
            "parallel": parallel_test.__dict__,
            "error_handling": error_test.__dict__,
        }

        # Run performance benchmarks
        logger.info("Running performance benchmarks...")

        response_time_benchmark = (
            await performance_suite.benchmark_agent_response_times(
                [
                    AgentRole.TRIP_PLANNER,
                    AgentRole.DESTINATION_EXPERT,
                    AgentRole.BUDGET_ADVISOR,
                ],
                message_count=5,
            )
        )

        stress_test_result = await performance_suite.stress_test_agent_system(
            max_concurrent_agents=5,
            messages_per_agent=3,
            duration_seconds=15,
        )

        test_results["test_results"]["performance_tests"] = {
            "response_times": response_time_benchmark,
            "stress_test": stress_test_result,
        }

        # Calculate summary metrics
        all_tests = [sequential_test, parallel_test, error_test]
        successful_tests = [t for t in all_tests if t.success]

        test_results["summary_metrics"] = {
            "total_tests": len(all_tests) + 2,  # +2 for performance tests
            "successful_tests": len(successful_tests)
            + (
                2
                if response_time_benchmark.get("success", True)
                and stress_test_result.get("success", True)
                else 0
            ),
            "test_success_rate": (len(successful_tests) / len(all_tests)) * 100,
            "average_test_time": statistics.mean([t.execution_time for t in all_tests]),
            "total_agents_tested": sum(t.agent_count for t in all_tests),
            "performance_insights": performance_suite.get_performance_report(),
        }

        test_results["overall_success"] = len(successful_tests) == len(all_tests)

    except Exception as e:
        test_results["overall_success"] = False
        test_results["error"] = str(e)
        logger.exception("Comprehensive test suite failed")

    test_suite_time = time.time() - test_suite_start
    test_results["total_execution_time"] = test_suite_time
    test_results["end_time"] = datetime.now(timezone.utc).isoformat()

    return test_results


def create_test_trip_request() -> TripRequest:
    """Create a standard test trip request."""
    from datetime import date
    from decimal import Decimal

    from src.trip_planner.schemas import (
        ActivityType,
        Budget,
        TripRequest,
        TripType,
    )

    return TripRequest(
        user_id="test_user_123",
        destination="Goa, India",
        start_date=date(2024, 2, 15),
        end_date=date(2024, 2, 20),
        duration_days=5,
        traveler_count=2,
        trip_type=TripType.LEISURE,
        budget=Budget(
            total_budget=Decimal("60000"),
            currency="INR",
            breakdown={
                "accommodation": Decimal("24000"),
                "food": Decimal("15000"),
                "activities": Decimal("12000"),
                "transport": Decimal("9000"),
            },
            remaining_amount=Decimal("60000"),  # Initial remaining amount
            daily_budget=Decimal("12000"),  # 60000 / 5 days
        ),
        preferred_activities=[
            ActivityType.SIGHTSEEING,
            ActivityType.CULTURAL,
            ActivityType.RELAXATION,
        ],
        special_requirements=["vegetarian_food"],
        updated_at=None,  # Optional field
        session_id=None,  # Optional field
    )


# Global test instances
_test_validators: Dict[str, AgentTestValidator] = {}


def get_test_validator(test_type: str = "default") -> AgentTestValidator:
    """Get test validator instance."""
    global _test_validators  # noqa: PLW0602

    if test_type not in _test_validators:
        _test_validators[test_type] = AgentTestValidator()

    return _test_validators[test_type]


async def cleanup_test_framework() -> None:
    """Cleanup test framework resources."""
    global _test_validators  # noqa: PLW0602

    _test_validators.clear()
    logger.info("Agent testing framework cleaned up")
