"""Agent Orchestrator for AI-Powered Trip Planner Backend - Google ADK Multi-Agent System.

This module provides the main trip planning orchestrator using Google ADK patterns
(SequentialAgent, ParallelAgent, LoopAgent) with multi-agent workflow coordination,
state management, error handling, and performance monitoring.
"""

import asyncio
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from src.ai_services.exceptions import (
    AgentError,
)
from src.ai_services.function_tools import get_tool_registry
from src.ai_services.gemini_agents import AgentRole, BaseAgent
from src.ai_services.session_manager import get_session_manager
from src.core.logging import get_logger

logger = get_logger(__name__)


class WorkflowType(str, Enum):
    """Types of agent workflows."""

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    LOOP = "loop"
    CONDITIONAL = "conditional"
    HYBRID = "hybrid"


class WorkflowState(str, Enum):
    """Workflow execution states."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentExecutionMode(str, Enum):
    """Agent execution modes in workflows."""

    SEQUENTIAL = "sequential"  # One after another
    PARALLEL = "parallel"  # All at once
    CONDITIONAL = "conditional"  # Based on conditions
    LOOP = "loop"  # Iterative execution


@dataclass
class AgentResult:
    """Result from agent execution."""

    agent_id: str
    agent_role: AgentRole
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    tokens_used: int = 0
    function_calls: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class WorkflowStep:
    """Individual step in a workflow."""

    step_id: str
    agent_roles: List[AgentRole]
    execution_mode: AgentExecutionMode
    dependencies: List[str] = field(default_factory=list)
    conditions: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 3
    timeout_seconds: int = 300
    required: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class WorkflowDefinition(BaseModel):
    """Definition of a multi-agent workflow."""

    workflow_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: str
    workflow_type: WorkflowType
    steps: List[WorkflowStep] = Field(default_factory=list)
    global_timeout: int = Field(default=1800)  # 30 minutes
    max_iterations: int = Field(default=5)  # For loop workflows
    success_criteria: Dict[str, Any] = Field(default_factory=dict)
    failure_handling: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class WorkflowExecution(BaseModel):
    """Runtime execution state of a workflow."""

    execution_id: str = Field(default_factory=lambda: str(uuid4()))
    workflow_id: str
    session_id: Optional[str] = None
    state: WorkflowState = WorkflowState.PENDING
    current_step: Optional[str] = None
    completed_steps: List[str] = Field(default_factory=list)
    failed_steps: List[str] = Field(default_factory=list)
    agent_results: Dict[str, AgentResult] = Field(default_factory=dict)
    iteration_count: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    total_execution_time: float = 0.0
    total_tokens: int = 0
    error_log: List[str] = Field(default_factory=list)
    context: Dict[str, Any] = Field(default_factory=dict)


class BaseWorkflowAgent(ABC):
    """Abstract base class for Google ADK workflow agents."""

    def __init__(
        self,
        agent_id: str,
        workflow_type: WorkflowType,
        session_id: Optional[str] = None,
    ) -> None:
        self.agent_id = agent_id
        self.workflow_type = workflow_type
        self.session_id = session_id
        self.agents: Dict[str, BaseAgent] = {}
        self.execution_stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "average_execution_time": 0.0,
            "total_tokens_used": 0,
        }

    @abstractmethod
    async def execute(
        self, workflow_definition: WorkflowDefinition, initial_context: Dict[str, Any]
    ) -> WorkflowExecution:
        """Execute the workflow."""

    def add_agent(self, agent: BaseAgent) -> None:
        """Add agent to the workflow."""
        self.agents[agent.agent_id] = agent

    def get_agent_by_role(self, role: AgentRole) -> Optional[BaseAgent]:
        """Get agent by role."""
        for agent in self.agents.values():
            if agent.capabilities.role == role:
                return agent
        return None

    def _check_dependencies(
        self, step: WorkflowStep, completed_steps: List[str]
    ) -> bool:
        """Check if step dependencies are satisfied."""
        return all(dep in completed_steps for dep in step.dependencies)

    async def _execute_step(
        self, step: WorkflowStep, context: Dict[str, Any]
    ) -> List[AgentResult]:
        """Execute a workflow step."""
        results = []

        for role in step.agent_roles:
            agent = self.get_agent_by_role(role)
            if not agent:
                logger.warning(f"No agent found for role {role.value}")
                continue

            start_time = datetime.now(timezone.utc)

            try:
                # Prepare message based on agent role and context
                message = self._prepare_agent_message(role, context, step)

                # Execute agent
                response = await asyncio.wait_for(
                    agent.process_message(message, context),
                    timeout=step.timeout_seconds,
                )

                execution_time = (
                    datetime.now(timezone.utc) - start_time
                ).total_seconds()

                result = AgentResult(
                    agent_id=agent.agent_id,
                    agent_role=role,
                    success=True,
                    result=response,
                    execution_time=execution_time,
                    tokens_used=getattr(agent, "last_token_count", 0),
                    function_calls=getattr(agent, "last_function_calls", []),
                )

                results.append(result)

            except asyncio.TimeoutError:
                execution_time = step.timeout_seconds
                result = AgentResult(
                    agent_id=agent.agent_id,
                    agent_role=role,
                    success=False,
                    error=f"Agent execution timeout after {step.timeout_seconds}s",
                    execution_time=execution_time,
                )
                results.append(result)

            except Exception as e:
                execution_time = (
                    datetime.now(timezone.utc) - start_time
                ).total_seconds()
                result = AgentResult(
                    agent_id=agent.agent_id,
                    agent_role=role,
                    success=False,
                    error=str(e),
                    execution_time=execution_time,
                )
                results.append(result)

        return results

    def _prepare_agent_message(
        self, role: AgentRole, context: Dict[str, Any], step: WorkflowStep
    ) -> str:
        """Prepare message for agent based on role and context."""
        base_context = {
            "step_id": step.step_id,
            "workflow_context": context,
            "step_metadata": step.metadata,
        }

        if role == AgentRole.TRIP_PLANNER:
            return f"""Plan a trip based on the following context:
            
Destination: {context.get("destination", "Not specified")}
Travel Dates: {context.get("travel_dates", "Not specified")}
Budget: {context.get("budget", "Not specified")}
Traveler Count: {context.get("traveler_count", 1)}
Preferences: {context.get("preferences", {})}

Please create a comprehensive trip plan including activities, accommodations, and transportation."""

        elif role == AgentRole.DESTINATION_EXPERT:
            return f"""Provide expert information about the destination:
            
Destination: {context.get("destination", "Not specified")}

Please provide detailed information about attractions, local customs, weather, best times to visit, and insider tips."""

        elif role == AgentRole.BUDGET_ADVISOR:
            return f"""Analyze and optimize the budget for this trip:
            
Budget: {context.get("budget", "Not specified")}
Destination: {context.get("destination", "Not specified")}
Duration: {context.get("duration_days", "Not specified")} days
Traveler Count: {context.get("traveler_count", 1)}

Please provide budget breakdown and cost optimization suggestions."""

        else:
            return f"Process the following context for {role.value}: {json.dumps(base_context, indent=2)}"

    def _update_stats(self, execution: WorkflowExecution, success: bool) -> None:
        """Update execution statistics."""
        self.execution_stats["total_executions"] += 1

        if success:
            self.execution_stats["successful_executions"] += 1
        else:
            self.execution_stats["failed_executions"] += 1

        # Update average execution time
        if execution.total_execution_time > 0:
            current_avg = self.execution_stats["average_execution_time"]
            total_execs = self.execution_stats["total_executions"]

            new_avg = (
                (current_avg * (total_execs - 1)) + execution.total_execution_time
            ) / total_execs
            self.execution_stats["average_execution_time"] = new_avg

        self.execution_stats["total_tokens_used"] += execution.total_tokens


class SequentialWorkflowAgent(BaseWorkflowAgent):
    """Google ADK SequentialAgent pattern implementation."""

    def __init__(self, agent_id: str, session_id: Optional[str] = None) -> None:
        super().__init__(agent_id, WorkflowType.SEQUENTIAL, session_id)

    async def execute(
        self, workflow_definition: WorkflowDefinition, initial_context: Dict[str, Any]
    ) -> WorkflowExecution:
        """Execute workflow steps sequentially."""
        execution = WorkflowExecution(
            workflow_id=workflow_definition.workflow_id,
            session_id=self.session_id,
            state=WorkflowState.RUNNING,
            start_time=datetime.now(timezone.utc),
            context=initial_context.copy(),
        )

        try:
            logger.info(
                "Starting sequential workflow execution",
                workflow_id=workflow_definition.workflow_id,
                execution_id=execution.execution_id,
                steps_count=len(workflow_definition.steps),
            )

            for step in workflow_definition.steps:
                execution.current_step = step.step_id

                # Check dependencies
                if not self._check_dependencies(step, execution.completed_steps):
                    if step.required:
                        raise AgentError(
                            f"Dependencies not met for step {step.step_id}"
                        )
                    logger.warning(
                        f"Skipping optional step {step.step_id} - dependencies not met"
                    )
                    continue

                # Execute step
                step_results = await self._execute_step(step, execution.context)

                # Process results
                for result in step_results:
                    execution.agent_results[result.agent_id] = result
                    execution.total_tokens += result.tokens_used

                    if not result.success:
                        execution.failed_steps.append(step.step_id)
                        if step.required:
                            raise AgentError(
                                f"Required step {step.step_id} failed: {result.error}"
                            )
                    # Update context with results
                    elif isinstance(result.result, dict):
                        execution.context.update(result.result)

                execution.completed_steps.append(step.step_id)

                logger.debug(
                    "Sequential step completed",
                    step_id=step.step_id,
                    execution_id=execution.execution_id,
                )

            execution.state = WorkflowState.COMPLETED
            execution.end_time = datetime.now(timezone.utc)

            # Safe datetime calculation
            if execution.start_time is not None and execution.end_time is not None:
                execution.total_execution_time = (
                    execution.end_time - execution.start_time
                ).total_seconds()

            self._update_stats(execution, success=True)

            logger.info(
                "Sequential workflow completed successfully",
                workflow_id=workflow_definition.workflow_id,
                execution_id=execution.execution_id,
                execution_time=execution.total_execution_time,
            )

            return execution

        except Exception as e:
            execution.state = WorkflowState.FAILED
            execution.end_time = datetime.now(timezone.utc)
            execution.error_log.append(str(e))

            # Safe datetime calculation
            if execution.start_time is not None and execution.end_time is not None:
                execution.total_execution_time = (
                    execution.end_time - execution.start_time
                ).total_seconds()

            self._update_stats(execution, success=False)

            logger.error(
                "Sequential workflow failed",
                workflow_id=workflow_definition.workflow_id,
                execution_id=execution.execution_id,
                error=str(e),
                exc_info=True,
            )

            return execution


class ParallelWorkflowAgent(BaseWorkflowAgent):
    """Google ADK ParallelAgent pattern implementation."""

    def __init__(self, agent_id: str, session_id: Optional[str] = None) -> None:
        super().__init__(agent_id, WorkflowType.PARALLEL, session_id)

    async def execute(
        self, workflow_definition: WorkflowDefinition, initial_context: Dict[str, Any]
    ) -> WorkflowExecution:
        """Execute workflow steps in parallel."""
        execution = WorkflowExecution(
            workflow_id=workflow_definition.workflow_id,
            session_id=self.session_id,
            state=WorkflowState.RUNNING,
            start_time=datetime.now(timezone.utc),
            context=initial_context.copy(),
        )

        try:
            logger.info(
                "Starting parallel workflow execution",
                workflow_id=workflow_definition.workflow_id,
                execution_id=execution.execution_id,
                steps_count=len(workflow_definition.steps),
            )

            # Execute all steps in parallel
            tasks = []
            for step in workflow_definition.steps:
                task = asyncio.create_task(
                    self._execute_step_with_timeout(step, execution.context)
                )
                tasks.append((step, task))

            # Wait for all tasks to complete
            for step, task in tasks:
                try:
                    step_results = await task

                    for result in step_results:
                        execution.agent_results[result.agent_id] = result
                        execution.total_tokens += result.tokens_used

                        if result.success:
                            execution.completed_steps.append(step.step_id)
                            if isinstance(result.result, dict):
                                execution.context.update(result.result)
                        else:
                            execution.failed_steps.append(step.step_id)
                            execution.error_log.append(
                                f"Step {step.step_id}: {result.error}"
                            )

                except Exception as e:
                    execution.failed_steps.append(step.step_id)
                    execution.error_log.append(f"Step {step.step_id}: {e!s}")

            # Determine final state
            if execution.failed_steps:
                required_failed = any(
                    step.required
                    for step in workflow_definition.steps
                    if step.step_id in execution.failed_steps
                )
                execution.state = (
                    WorkflowState.FAILED if required_failed else WorkflowState.COMPLETED
                )
            else:
                execution.state = WorkflowState.COMPLETED

            execution.end_time = datetime.now(timezone.utc)

            # Safe datetime calculation
            if execution.start_time is not None and execution.end_time is not None:
                execution.total_execution_time = (
                    execution.end_time - execution.start_time
                ).total_seconds()

            self._update_stats(
                execution, success=execution.state == WorkflowState.COMPLETED
            )

            logger.info(
                "Parallel workflow completed",
                workflow_id=workflow_definition.workflow_id,
                execution_id=execution.execution_id,
                state=execution.state.value,
                execution_time=execution.total_execution_time,
            )

            return execution

        except Exception as e:
            execution.state = WorkflowState.FAILED
            execution.end_time = datetime.now(timezone.utc)
            execution.error_log.append(str(e))

            # Safe datetime calculation
            if execution.start_time is not None and execution.end_time is not None:
                execution.total_execution_time = (
                    execution.end_time - execution.start_time
                ).total_seconds()

            self._update_stats(execution, success=False)

            logger.error(
                "Parallel workflow failed",
                workflow_id=workflow_definition.workflow_id,
                execution_id=execution.execution_id,
                error=str(e),
                exc_info=True,
            )

            return execution

    async def _execute_step_with_timeout(
        self, step: WorkflowStep, context: Dict[str, Any]
    ) -> List[AgentResult]:
        """Execute step with timeout handling."""
        try:
            return await asyncio.wait_for(
                self._execute_step(step, context), timeout=step.timeout_seconds
            )
        except asyncio.TimeoutError:
            return [
                AgentResult(
                    agent_id=f"timeout_{step.step_id}",
                    agent_role=role,
                    success=False,
                    error=f"Step timeout after {step.timeout_seconds}s",
                    execution_time=step.timeout_seconds,
                )
                for role in step.agent_roles
            ]


class LoopWorkflowAgent(BaseWorkflowAgent):
    """Google ADK LoopAgent pattern implementation."""

    def __init__(self, agent_id: str, session_id: Optional[str] = None) -> None:
        super().__init__(agent_id, WorkflowType.LOOP, session_id)

    async def execute(
        self, workflow_definition: WorkflowDefinition, initial_context: Dict[str, Any]
    ) -> WorkflowExecution:
        """Execute workflow steps in iterative loops."""
        execution = WorkflowExecution(
            workflow_id=workflow_definition.workflow_id,
            session_id=self.session_id,
            state=WorkflowState.RUNNING,
            start_time=datetime.now(timezone.utc),
            context=initial_context.copy(),
        )

        try:
            logger.info(
                "Starting loop workflow execution",
                workflow_id=workflow_definition.workflow_id,
                execution_id=execution.execution_id,
                max_iterations=workflow_definition.max_iterations,
            )

            while execution.iteration_count < workflow_definition.max_iterations:
                execution.iteration_count += 1
                iteration_successful = True

                logger.debug(
                    f"Starting iteration {execution.iteration_count}",
                    execution_id=execution.execution_id,
                )

                # Execute all steps in sequence for this iteration
                for step in workflow_definition.steps:
                    execution.current_step = step.step_id

                    step_results = await self._execute_step(step, execution.context)

                    for result in step_results:
                        result_key = (
                            f"{result.agent_id}_iter_{execution.iteration_count}"
                        )
                        execution.agent_results[result_key] = result
                        execution.total_tokens += result.tokens_used

                        if not result.success:
                            iteration_successful = False
                            execution.error_log.append(
                                f"Iteration {execution.iteration_count}, Step {step.step_id}: {result.error}"
                            )
                        # Update context with results
                        elif isinstance(result.result, dict):
                            execution.context.update(result.result)

                # Check success criteria
                if self._check_success_criteria(
                    workflow_definition.success_criteria, execution.context
                ):
                    logger.info(
                        f"Success criteria met after {execution.iteration_count} iterations",
                        execution_id=execution.execution_id,
                    )
                    break

                # Check if we should continue based on results
                if (
                    not iteration_successful
                    and workflow_definition.failure_handling.get(
                        "stop_on_failure", False
                    )
                ):
                    logger.warning(
                        "Stopping loop due to iteration failure",
                        execution_id=execution.execution_id,
                        iteration=execution.iteration_count,
                    )
                    break

            execution.state = WorkflowState.COMPLETED
            execution.end_time = datetime.now(timezone.utc)

            # Safe datetime calculation
            if execution.start_time is not None and execution.end_time is not None:
                execution.total_execution_time = (
                    execution.end_time - execution.start_time
                ).total_seconds()

            self._update_stats(execution, success=True)

            logger.info(
                "Loop workflow completed",
                workflow_id=workflow_definition.workflow_id,
                execution_id=execution.execution_id,
                iterations=execution.iteration_count,
                execution_time=execution.total_execution_time,
            )

            return execution

        except Exception as e:
            execution.state = WorkflowState.FAILED
            execution.end_time = datetime.now(timezone.utc)
            execution.error_log.append(str(e))

            # Safe datetime calculation
            if execution.start_time is not None and execution.end_time is not None:
                execution.total_execution_time = (
                    execution.end_time - execution.start_time
                ).total_seconds()

            self._update_stats(execution, success=False)

            logger.error(
                "Loop workflow failed",
                workflow_id=workflow_definition.workflow_id,
                execution_id=execution.execution_id,
                error=str(e),
                exc_info=True,
            )

            return execution

    def _check_success_criteria(
        self, criteria: Dict[str, Any], context: Dict[str, Any]
    ) -> bool:
        """Check if success criteria are met."""
        if not criteria:
            return False

        for criterion, expected_value in criteria.items():
            if criterion not in context:
                return False

            if context[criterion] != expected_value:
                return False

        return True


class TripPlannerOrchestrator:
    """Main orchestrator for AI-powered trip planning using multi-agent workflows."""

    def __init__(self, session_id: Optional[str] = None) -> None:
        self.session_id = session_id
        self.session_manager = get_session_manager()
        self.tool_registry = get_tool_registry()

        # Workflow agents
        self.sequential_agent = SequentialWorkflowAgent("seq_orchestrator", session_id)
        self.parallel_agent = ParallelWorkflowAgent("par_orchestrator", session_id)
        self.loop_agent = LoopWorkflowAgent("loop_orchestrator", session_id)

        # Active executions
        self.active_executions: Dict[str, WorkflowExecution] = {}

        # Performance metrics
        self.metrics = {
            "total_workflows": 0,
            "successful_workflows": 0,
            "failed_workflows": 0,
            "average_execution_time": 0.0,
            "total_tokens_consumed": 0,
        }

    async def register_agent(self, agent: BaseAgent) -> None:
        """Register agent with all workflow orchestrators."""
        self.sequential_agent.add_agent(agent)
        self.parallel_agent.add_agent(agent)
        self.loop_agent.add_agent(agent)

        logger.info(
            "Agent registered with orchestrator",
            agent_id=agent.agent_id,
            role=agent.capabilities.role.value,
        )

    async def execute_workflow(
        self, workflow_definition: WorkflowDefinition, initial_context: Dict[str, Any]
    ) -> WorkflowExecution:
        """Execute a workflow using the appropriate agent pattern."""

        # Select appropriate workflow agent
        if workflow_definition.workflow_type == WorkflowType.SEQUENTIAL:
            workflow_agent = self.sequential_agent
        elif workflow_definition.workflow_type == WorkflowType.PARALLEL:
            workflow_agent = self.parallel_agent
        elif workflow_definition.workflow_type == WorkflowType.LOOP:
            workflow_agent = self.loop_agent
        else:
            raise AgentError(
                f"Unsupported workflow type: {workflow_definition.workflow_type}"
            )

        # Execute workflow
        execution = await workflow_agent.execute(workflow_definition, initial_context)

        # Track execution
        self.active_executions[execution.execution_id] = execution

        # Update metrics
        self._update_orchestrator_metrics(execution)

        # Persist execution state if session is available
        if self.session_id:
            await self._persist_execution_state(execution)

        return execution

    def _update_orchestrator_metrics(self, execution: WorkflowExecution) -> None:
        """Update orchestrator performance metrics."""
        self.metrics["total_workflows"] += 1

        if execution.state == WorkflowState.COMPLETED:
            self.metrics["successful_workflows"] += 1
        elif execution.state == WorkflowState.FAILED:
            self.metrics["failed_workflows"] += 1

        if execution.total_execution_time > 0:
            current_avg = self.metrics["average_execution_time"]
            total_workflows = self.metrics["total_workflows"]

            new_avg = (
                (current_avg * (total_workflows - 1)) + execution.total_execution_time
            ) / total_workflows
            self.metrics["average_execution_time"] = new_avg

        self.metrics["total_tokens_consumed"] += execution.total_tokens

    async def _persist_execution_state(self, execution: WorkflowExecution) -> None:
        """Persist workflow execution state to session."""
        try:
            if not self.session_id:
                return

            session = await self.session_manager.get_session(self.session_id)

            # Store execution state in session context
            execution_data = {
                "execution_id": execution.execution_id,
                "workflow_id": execution.workflow_id,
                "state": execution.state.value,
                "completed_steps": execution.completed_steps,
                "failed_steps": execution.failed_steps,
                "total_execution_time": execution.total_execution_time,
                "total_tokens": execution.total_tokens,
                "context": execution.context,
            }

            session.update_context(workflow_execution=execution_data)
            await self.session_manager.update_session(session)

        except Exception as e:
            logger.warning(
                "Failed to persist execution state",
                execution_id=execution.execution_id,
                error=str(e),
            )

    def get_execution_status(self, execution_id: str) -> Optional[WorkflowExecution]:
        """Get status of a workflow execution."""
        return self.active_executions.get(execution_id)

    def get_orchestrator_metrics(self) -> Dict[str, Any]:
        """Get orchestrator performance metrics."""
        return {
            **self.metrics,
            "active_executions": len(self.active_executions),
            "workflow_agent_stats": {
                "sequential": self.sequential_agent.execution_stats,
                "parallel": self.parallel_agent.execution_stats,
                "loop": self.loop_agent.execution_stats,
            },
        }

    async def cleanup_completed_executions(self) -> None:
        """Clean up completed workflow executions from memory."""
        completed_executions = [
            exec_id
            for exec_id, execution in self.active_executions.items()
            if execution.state
            in [WorkflowState.COMPLETED, WorkflowState.FAILED, WorkflowState.CANCELLED]
        ]

        for exec_id in completed_executions:
            del self.active_executions[exec_id]

        logger.info(f"Cleaned up {len(completed_executions)} completed executions")


# Global orchestrator instance
_orchestrator: Optional[TripPlannerOrchestrator] = None


def get_trip_planner_orchestrator(
    session_id: Optional[str] = None,
) -> TripPlannerOrchestrator:
    """Get or create trip planner orchestrator instance."""
    global _orchestrator

    if _orchestrator is None or (
        _orchestrator.session_id != session_id and session_id is not None
    ):
        _orchestrator = TripPlannerOrchestrator(session_id)

    return _orchestrator


# Pre-defined workflow definitions for common trip planning scenarios


def create_comprehensive_trip_planning_workflow() -> WorkflowDefinition:
    """Create a comprehensive sequential trip planning workflow."""
    return WorkflowDefinition(
        name="Comprehensive Trip Planning",
        description="Complete trip planning workflow using sequential agent coordination",
        workflow_type=WorkflowType.SEQUENTIAL,
        steps=[
            WorkflowStep(
                step_id="destination_research",
                agent_roles=[AgentRole.DESTINATION_EXPERT],
                execution_mode=AgentExecutionMode.SEQUENTIAL,
                timeout_seconds=120,
                required=True,
                metadata={"priority": "high", "category": "research"},
            ),
            WorkflowStep(
                step_id="budget_analysis",
                agent_roles=[AgentRole.BUDGET_ADVISOR],
                execution_mode=AgentExecutionMode.SEQUENTIAL,
                dependencies=["destination_research"],
                timeout_seconds=90,
                required=True,
                metadata={"priority": "high", "category": "planning"},
            ),
            WorkflowStep(
                step_id="itinerary_creation",
                agent_roles=[AgentRole.TRIP_PLANNER],
                execution_mode=AgentExecutionMode.SEQUENTIAL,
                dependencies=["destination_research", "budget_analysis"],
                timeout_seconds=300,
                required=True,
                metadata={"priority": "critical", "category": "planning"},
            ),
            WorkflowStep(
                step_id="optimization",
                agent_roles=[AgentRole.ITINERARY_OPTIMIZER],
                execution_mode=AgentExecutionMode.SEQUENTIAL,
                dependencies=["itinerary_creation"],
                timeout_seconds=180,
                required=False,
                metadata={"priority": "medium", "category": "optimization"},
            ),
        ],
        global_timeout=900,  # 15 minutes total
        success_criteria={"trip_plan_created": True},
        failure_handling={"stop_on_failure": True},
        metadata={"workflow_version": "1.0", "category": "trip_planning"},
    )


def create_quick_trip_planning_workflow() -> WorkflowDefinition:
    """Create a quick parallel trip planning workflow."""
    return WorkflowDefinition(
        name="Quick Trip Planning",
        description="Fast trip planning using parallel agent coordination",
        workflow_type=WorkflowType.PARALLEL,
        steps=[
            WorkflowStep(
                step_id="parallel_research",
                agent_roles=[AgentRole.DESTINATION_EXPERT, AgentRole.BUDGET_ADVISOR],
                execution_mode=AgentExecutionMode.PARALLEL,
                timeout_seconds=60,
                required=True,
                metadata={"priority": "high", "category": "research"},
            ),
            WorkflowStep(
                step_id="quick_planning",
                agent_roles=[AgentRole.TRIP_PLANNER],
                execution_mode=AgentExecutionMode.SEQUENTIAL,
                timeout_seconds=120,
                required=True,
                metadata={"priority": "critical", "category": "planning"},
            ),
        ],
        global_timeout=300,  # 5 minutes total
        success_criteria={"basic_plan_created": True},
        failure_handling={"stop_on_failure": False},
        metadata={"workflow_version": "1.0", "category": "quick_planning"},
    )


async def cleanup_orchestrator() -> None:
    """Cleanup global orchestrator instance."""
    global _orchestrator

    if _orchestrator:
        await _orchestrator.cleanup_completed_executions()
        _orchestrator = None
