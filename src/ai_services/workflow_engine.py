"""Workflow Engine for AI-Powered Trip Planner Backend - Google ADK Multi-Agent System.

This module provides workflow execution engine with sequential workflow execution for
step-by-step planning, parallel workflow execution for independent tasks, loop workflows
for iterative refinement, conditional workflows based on user inputs and preferences,
and workflow state persistence and resumption capabilities.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from src.ai_services.agent_communication import (
    get_communication_manager,
)
from src.ai_services.agent_factory import (
    TripRequirements,
    get_agent_factory,
)
from src.ai_services.agent_orchestrator import (
    TripPlannerOrchestrator,
    WorkflowDefinition,
    WorkflowExecution,
    WorkflowState,
    WorkflowType,
)
from src.ai_services.exceptions import (
    AgentError,
)
from src.ai_services.session_manager import get_session_manager
from src.core.logging import get_logger

logger = get_logger(__name__)


class WorkflowPriority(str, Enum):
    """Workflow execution priority levels."""

    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    BACKGROUND = "background"


class WorkflowTrigger(str, Enum):
    """Workflow trigger types."""

    MANUAL = "manual"
    SCHEDULED = "scheduled"
    EVENT_DRIVEN = "event_driven"
    CONDITIONAL = "conditional"
    CHAIN_REACTION = "chain_reaction"


class ConditionalOperator(str, Enum):
    """Operators for conditional workflow logic."""

    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    CONTAINS = "contains"
    EXISTS = "exists"
    AND = "and"
    OR = "or"


@dataclass
class WorkflowCondition:
    """Condition for conditional workflow execution."""

    field_name: str
    operator: ConditionalOperator
    expected_value: Any = None
    sub_conditions: List["WorkflowCondition"] = field(default_factory=list)

    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Evaluate condition against context."""
        try:
            if self.operator == ConditionalOperator.AND:
                return all(cond.evaluate(context) for cond in self.sub_conditions)
            elif self.operator == ConditionalOperator.OR:
                return any(cond.evaluate(context) for cond in self.sub_conditions)

            field_value = context.get(self.field_name)

            if self.operator == ConditionalOperator.EXISTS:
                return field_value is not None
            elif self.operator == ConditionalOperator.EQUALS:
                return field_value == self.expected_value
            elif self.operator == ConditionalOperator.NOT_EQUALS:
                return field_value != self.expected_value
            elif self.operator == ConditionalOperator.GREATER_THAN:
                return field_value > self.expected_value
            elif self.operator == ConditionalOperator.LESS_THAN:
                return field_value < self.expected_value
            elif self.operator == ConditionalOperator.CONTAINS:
                return self.expected_value in str(field_value)

            return False

        except Exception as e:
            logger.warning(f"Condition evaluation failed: {e}")
            return False


@dataclass
class WorkflowTemplate:
    """Template for creating workflow instances."""

    template_id: str
    name: str
    description: str
    workflow_type: WorkflowType
    default_priority: WorkflowPriority = WorkflowPriority.NORMAL
    required_context: List[str] = field(default_factory=list)
    optional_context: List[str] = field(default_factory=list)
    estimated_duration: int = 300  # seconds
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    conditions: List[WorkflowCondition] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    def can_execute(self, context: Dict[str, Any]) -> bool:
        """Check if template can execute with given context."""
        # Check required context
        if not all(key in context for key in self.required_context):
            return False

        # Evaluate conditions
        return all(condition.evaluate(context) for condition in self.conditions)


class WorkflowExecutionEngine:
    """Engine for executing workflows with state persistence and recovery."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.session_manager = get_session_manager()
        self.communication_manager = get_communication_manager(session_id)
        self.agent_factory = get_agent_factory()

        # Execution state
        self._active_workflows: Dict[str, WorkflowExecution] = {}
        self._workflow_queue: List[Dict[str, Any]] = []
        self._execution_stats = {
            "total_executed": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "average_execution_time": 0.0,
            "workflows_by_type": {},
        }

        # Recovery and persistence
        self._checkpoint_interval = 30  # seconds
        self._last_checkpoint = datetime.now(timezone.utc)

    async def initialize(self) -> None:
        """Initialize workflow engine."""
        await self.communication_manager.initialize()

        # Load persisted workflow state
        await self._load_persisted_workflows()

        logger.info(
            "Workflow execution engine initialized",
            session_id=self.session_id,
            active_workflows=len(self._active_workflows),
        )

    async def execute_workflow(
        self,
        workflow_definition: WorkflowDefinition,
        trip_requirements: TripRequirements,
        priority: WorkflowPriority = WorkflowPriority.NORMAL,
    ) -> WorkflowExecution:
        """Execute a workflow with agent creation and coordination."""

        execution_id = str(uuid4())

        try:
            logger.info(
                "Starting workflow execution",
                execution_id=execution_id,
                workflow_type=workflow_definition.workflow_type.value,
                priority=priority.value,
            )

            # Create required agents for workflow
            agent_team = self.agent_factory.create_agent_team_for_trip(
                trip_requirements, self.session_id
            )

            # Initialize orchestrator with agents
            orchestrator = TripPlannerOrchestrator(self.session_id)
            for agent in agent_team.values():
                await orchestrator.register_agent(agent)

            # Prepare initial context
            initial_context = {
                "destination": trip_requirements.destination,
                "duration_days": trip_requirements.duration_days,
                "traveler_count": trip_requirements.traveler_count,
                "budget_range": trip_requirements.budget_range,
                "trip_type": trip_requirements.trip_type,
                "special_requirements": trip_requirements.special_requirements,
                "execution_id": execution_id,
                "priority": priority.value,
                "agents_available": list(agent_team.keys()),
            }

            # Execute workflow based on type
            if workflow_definition.workflow_type == WorkflowType.SEQUENTIAL:
                execution = await self._execute_sequential_workflow(
                    workflow_definition, initial_context, orchestrator
                )
            elif workflow_definition.workflow_type == WorkflowType.PARALLEL:
                execution = await self._execute_parallel_workflow(
                    workflow_definition, initial_context, orchestrator
                )
            elif workflow_definition.workflow_type == WorkflowType.LOOP:
                execution = await self._execute_loop_workflow(
                    workflow_definition, initial_context, orchestrator
                )
            else:
                raise AgentError(
                    f"Unsupported workflow type: {workflow_definition.workflow_type}"
                )

            # Track execution
            self._active_workflows[execution.execution_id] = execution
            self._update_execution_stats(execution)

            # Persist execution state
            await self._persist_workflow_state(execution)

            logger.info(
                "Workflow execution completed",
                execution_id=execution.execution_id,
                state=execution.state.value,
                execution_time=execution.total_execution_time,
            )

            return execution

        except Exception as e:
            logger.error(
                "Workflow execution failed",
                execution_id=execution_id,
                error=str(e),
                exc_info=True,
            )

            # Create failed execution record
            failed_execution = WorkflowExecution(
                execution_id=execution_id,
                workflow_id=workflow_definition.workflow_id,
                session_id=self.session_id,
                state=WorkflowState.FAILED,
                start_time=datetime.now(timezone.utc),
                end_time=datetime.now(timezone.utc),
                error_log=[str(e)],
            )

            return failed_execution

    async def _execute_sequential_workflow(
        self,
        workflow_definition: WorkflowDefinition,
        initial_context: Dict[str, Any],
        orchestrator: TripPlannerOrchestrator,
    ) -> WorkflowExecution:
        """Execute sequential workflow with step-by-step coordination."""

        execution = await orchestrator.execute_workflow(
            workflow_definition, initial_context
        )

        # Add communication coordination
        if len(orchestrator.sequential_agent.agents) > 1:
            agent_ids = list(orchestrator.sequential_agent.agents.keys())
            await self.communication_manager.create_handoff_coordination(
                agent_ids, initial_context
            )

        return execution

    async def _execute_parallel_workflow(
        self,
        workflow_definition: WorkflowDefinition,
        initial_context: Dict[str, Any],
        orchestrator: TripPlannerOrchestrator,
    ) -> WorkflowExecution:
        """Execute parallel workflow with concurrent agent coordination."""

        execution = await orchestrator.execute_workflow(
            workflow_definition, initial_context
        )

        # Add collaboration coordination for parallel execution
        if len(orchestrator.parallel_agent.agents) > 1:
            agent_ids = list(orchestrator.parallel_agent.agents.keys())
            await self.communication_manager.create_collaboration_coordination(
                agent_ids, initial_context
            )

        return execution

    async def _execute_loop_workflow(
        self,
        workflow_definition: WorkflowDefinition,
        initial_context: Dict[str, Any],
        orchestrator: TripPlannerOrchestrator,
    ) -> WorkflowExecution:
        """Execute loop workflow with iterative refinement."""

        execution = await orchestrator.execute_workflow(
            workflow_definition, initial_context
        )

        # Add iterative communication patterns
        agent_ids = list(orchestrator.loop_agent.agents.keys())
        if agent_ids:
            # Create feedback loop between agents
            for i in range(execution.iteration_count):
                await self.communication_manager.shared_context.update_context(
                    "loop_coordinator", {f"iteration_{i}_context": execution.context}
                )

        return execution

    async def execute_conditional_workflow(
        self,
        workflow_templates: List[WorkflowTemplate],
        trip_requirements: TripRequirements,
        context: Dict[str, Any],
    ) -> Optional[WorkflowExecution]:
        """Execute workflow based on conditions."""

        # Find matching template
        matching_template = None
        for template in workflow_templates:
            if template.can_execute(context):
                matching_template = template
                break

        if not matching_template:
            logger.warning("No matching workflow template found for conditions")
            return None

        # Create workflow definition from template
        workflow_definition = WorkflowDefinition(
            name=matching_template.name,
            description=matching_template.description,
            workflow_type=matching_template.workflow_type,
            metadata={"template_id": matching_template.template_id},
        )

        # Execute with appropriate priority
        return await self.execute_workflow(
            workflow_definition,
            trip_requirements,
            matching_template.default_priority,
        )

    async def queue_workflow(
        self,
        workflow_definition: WorkflowDefinition,
        trip_requirements: TripRequirements,
        priority: WorkflowPriority = WorkflowPriority.NORMAL,
        scheduled_time: Optional[datetime] = None,
    ) -> str:
        """Queue workflow for later execution."""

        queue_item = {
            "queue_id": str(uuid4()),
            "workflow_definition": workflow_definition.model_dump(),
            "trip_requirements": trip_requirements.__dict__,
            "priority": priority.value,
            "scheduled_time": scheduled_time.isoformat() if scheduled_time else None,
            "queued_at": datetime.now(timezone.utc).isoformat(),
            "status": "queued",
        }

        # Insert based on priority
        if priority == WorkflowPriority.CRITICAL:
            self._workflow_queue.insert(0, queue_item)
        else:
            self._workflow_queue.append(queue_item)

        logger.info(
            "Workflow queued",
            queue_id=queue_item["queue_id"],
            priority=priority.value,
            scheduled_time=(
                scheduled_time.isoformat() if scheduled_time else "immediate"
            ),
        )

        return queue_item["queue_id"]

    async def process_workflow_queue(self) -> Dict[str, Any]:
        """Process queued workflows."""

        processed_count = 0
        successful_count = 0
        failed_count = 0

        processing_results = {
            "start_time": datetime.now(timezone.utc).isoformat(),
            "processed_workflows": [],
            "statistics": {},
        }

        try:
            # Process queue items
            current_time = datetime.now(timezone.utc)

            for queue_item in self._workflow_queue.copy():
                # Check if scheduled time has arrived
                scheduled_time = queue_item.get("scheduled_time")
                if scheduled_time:
                    scheduled_dt = datetime.fromisoformat(
                        scheduled_time.replace("Z", "+00:00")
                    )
                    if current_time < scheduled_dt:
                        continue

                # Remove from queue
                self._workflow_queue.remove(queue_item)
                processed_count += 1

                try:
                    # Reconstruct objects
                    workflow_def = WorkflowDefinition(
                        **queue_item["workflow_definition"]
                    )
                    trip_req = TripRequirements(**queue_item["trip_requirements"])
                    priority = WorkflowPriority(queue_item["priority"])

                    # Execute workflow
                    execution = await self.execute_workflow(
                        workflow_def, trip_req, priority
                    )

                    if execution.state == WorkflowState.COMPLETED:
                        successful_count += 1
                    else:
                        failed_count += 1

                    processing_results["processed_workflows"].append(
                        {
                            "queue_id": queue_item["queue_id"],
                            "execution_id": execution.execution_id,
                            "success": execution.state == WorkflowState.COMPLETED,
                            "execution_time": execution.total_execution_time,
                        }
                    )

                except Exception as e:
                    failed_count += 1
                    processing_results["processed_workflows"].append(
                        {
                            "queue_id": queue_item["queue_id"],
                            "success": False,
                            "error": str(e),
                        }
                    )

                    logger.exception("Queued workflow execution failed")

            processing_results["statistics"] = {
                "processed_count": processed_count,
                "successful_count": successful_count,
                "failed_count": failed_count,
                "remaining_queue_size": len(self._workflow_queue),
            }

            processing_results["end_time"] = datetime.now(timezone.utc).isoformat()

            logger.info(
                "Workflow queue processed",
                processed=processed_count,
                successful=successful_count,
                failed=failed_count,
                remaining=len(self._workflow_queue),
            )

        except Exception as e:
            processing_results["error"] = str(e)
            logger.exception("Workflow queue processing failed")

        return processing_results

    async def pause_workflow(self, execution_id: str) -> bool:
        """Pause an active workflow execution."""
        if execution_id in self._active_workflows:
            execution = self._active_workflows[execution_id]
            execution.state = WorkflowState.PAUSED

            await self._persist_workflow_state(execution)

            logger.info(f"Workflow paused: {execution_id}")
            return True

        return False

    async def resume_workflow(self, execution_id: str) -> Optional[WorkflowExecution]:
        """Resume a paused workflow execution."""
        if execution_id in self._active_workflows:
            execution = self._active_workflows[execution_id]

            if execution.state == WorkflowState.PAUSED:
                execution.state = WorkflowState.RUNNING

                # Continue execution from last checkpoint
                # This is a simplified implementation - in production, you'd need
                # more sophisticated state recovery

                await self._persist_workflow_state(execution)

                logger.info(f"Workflow resumed: {execution_id}")
                return execution

        return None

    async def cancel_workflow(self, execution_id: str) -> bool:
        """Cancel an active workflow execution."""
        if execution_id in self._active_workflows:
            execution = self._active_workflows[execution_id]
            execution.state = WorkflowState.CANCELLED
            execution.end_time = datetime.now(timezone.utc)

            await self._persist_workflow_state(execution)

            # Clean up from active workflows
            del self._active_workflows[execution_id]

            logger.info(f"Workflow cancelled: {execution_id}")
            return True

        return False

    async def _load_persisted_workflows(self) -> None:
        """Load persisted workflow states from session."""
        try:
            session = await self.session_manager.get_session(self.session_id)
            persisted_workflows = session.context.session_metadata.get(
                "active_workflows", {}
            )

            for exec_id, workflow_data in persisted_workflows.items():
                try:
                    # Reconstruct workflow execution
                    execution = WorkflowExecution(**workflow_data)

                    # Only load if not completed or failed
                    if execution.state in [WorkflowState.RUNNING, WorkflowState.PAUSED]:
                        self._active_workflows[exec_id] = execution

                        logger.debug(f"Loaded persisted workflow: {exec_id}")

                except Exception as e:
                    logger.warning(f"Failed to load persisted workflow {exec_id}: {e}")

            logger.info(f"Loaded {len(self._active_workflows)} persisted workflows")

        except Exception as e:
            logger.warning(f"Failed to load persisted workflows: {e}")

    async def _persist_workflow_state(self, execution: WorkflowExecution) -> None:
        """Persist workflow execution state."""
        try:
            session = await self.session_manager.get_session(self.session_id)

            # Get current active workflows
            active_workflows = session.context.session_metadata.get(
                "active_workflows", {}
            )

            # Update with current execution
            active_workflows[execution.execution_id] = execution.model_dump(mode="json")

            # Clean up completed/failed workflows older than 1 hour
            current_time = datetime.now(timezone.utc)
            cleaned_workflows = {}

            for exec_id, workflow_data in active_workflows.items():
                try:
                    end_time = workflow_data.get("end_time")
                    if end_time:
                        end_dt = datetime.fromisoformat(end_time.replace("Z", "+00:00"))
                        if current_time - end_dt < timedelta(hours=1):
                            cleaned_workflows[exec_id] = workflow_data
                    else:
                        cleaned_workflows[exec_id] = workflow_data
                except Exception:
                    # Keep workflow if date parsing fails
                    cleaned_workflows[exec_id] = workflow_data

            # Update session
            session.update_context(active_workflows=cleaned_workflows)
            await self.session_manager.update_session(session)

        except Exception as e:
            logger.warning(f"Failed to persist workflow state: {e}")

    async def _checkpoint_active_workflows(self) -> None:
        """Create checkpoint for all active workflows."""
        current_time = datetime.now(timezone.utc)

        if current_time - self._last_checkpoint > timedelta(
            seconds=self._checkpoint_interval
        ):
            try:
                for execution in self._active_workflows.values():
                    if execution.state == WorkflowState.RUNNING:
                        await self._persist_workflow_state(execution)

                self._last_checkpoint = current_time

                logger.debug(
                    f"Checkpointed {len(self._active_workflows)} active workflows"
                )

            except Exception:
                logger.exception("Workflow checkpoint failed")

    def _update_execution_stats(self, execution: WorkflowExecution) -> None:
        """Update workflow execution statistics."""
        self._execution_stats["total_executed"] += 1

        if execution.state == WorkflowState.COMPLETED:
            self._execution_stats["successful_executions"] += 1
        elif execution.state == WorkflowState.FAILED:
            self._execution_stats["failed_executions"] += 1

        # Update average execution time
        if execution.total_execution_time > 0:
            current_avg = self._execution_stats["average_execution_time"]
            total_execs = self._execution_stats["total_executed"]

            new_avg = (
                (current_avg * (total_execs - 1)) + execution.total_execution_time
            ) / total_execs
            self._execution_stats["average_execution_time"] = new_avg

        # Update workflow type statistics
        workflow_type = "unknown"
        if execution.workflow_id in [
            wf.workflow_id for wf in self._get_workflow_definitions()
        ]:
            # In production, you'd track this more systematically
            workflow_type = "trip_planning"

        type_count = self._execution_stats["workflows_by_type"].get(workflow_type, 0)
        self._execution_stats["workflows_by_type"][workflow_type] = type_count + 1

    def _get_workflow_definitions(self) -> List[WorkflowDefinition]:
        """Get available workflow definitions (placeholder for registry)."""
        # In production, this would come from a workflow registry
        return []

    def get_engine_stats(self) -> Dict[str, Any]:
        """Get workflow engine statistics."""
        return {
            **self._execution_stats,
            "active_workflows": len(self._active_workflows),
            "queued_workflows": len(self._workflow_queue),
            "session_id": self.session_id,
            "last_checkpoint": self._last_checkpoint.isoformat(),
            "communication_stats": self.communication_manager.get_communication_stats(),
        }

    def get_active_workflows(self) -> Dict[str, WorkflowExecution]:
        """Get currently active workflow executions."""
        return self._active_workflows.copy()

    async def cleanup(self) -> None:
        """Cleanup workflow engine resources."""
        # Final checkpoint
        await self._checkpoint_active_workflows()

        # Clean up agent factory resources
        await self.agent_factory.cleanup_all_agents()

        logger.info(f"Workflow engine cleaned up for session {self.session_id}")


class WorkflowTemplateRegistry:
    """Registry for workflow templates and definitions."""

    def __init__(self) -> None:
        self._templates: Dict[str, WorkflowTemplate] = {}
        self._register_default_templates()

    def _register_default_templates(self) -> None:
        """Register default workflow templates."""

        # Simple trip planning template
        simple_template = WorkflowTemplate(
            template_id="simple_trip_planning",
            name="Simple Trip Planning",
            description="Basic trip planning for simple destinations",
            workflow_type=WorkflowType.SEQUENTIAL,
            required_context=["destination", "duration_days"],
            optional_context=["budget_range", "traveler_count"],
            estimated_duration=300,
            conditions=[
                WorkflowCondition("duration_days", ConditionalOperator.LESS_THAN, 5),
                WorkflowCondition("complexity", ConditionalOperator.EQUALS, "simple"),
            ],
            tags=["simple", "basic", "quick"],
        )

        # Complex trip planning template
        complex_template = WorkflowTemplate(
            template_id="complex_trip_planning",
            name="Complex Trip Planning",
            description="Comprehensive planning for complex trips",
            workflow_type=WorkflowType.PARALLEL,
            default_priority=WorkflowPriority.HIGH,
            required_context=["destination", "duration_days", "budget_range"],
            optional_context=["special_requirements", "preferred_activities"],
            estimated_duration=900,
            conditions=[
                WorkflowCondition(
                    field_name="duration_days",
                    operator=ConditionalOperator.GREATER_THAN,
                    expected_value=7,
                ),
            ],
            tags=["complex", "comprehensive", "detailed"],
        )

        # Iterative optimization template
        optimization_template = WorkflowTemplate(
            template_id="iterative_optimization",
            name="Iterative Trip Optimization",
            description="Iterative refinement of trip plans",
            workflow_type=WorkflowType.LOOP,
            default_priority=WorkflowPriority.NORMAL,
            required_context=["existing_itinerary", "optimization_criteria"],
            estimated_duration=600,
            conditions=[
                WorkflowCondition(
                    "optimization_requested", ConditionalOperator.EQUALS, True
                ),
            ],
            tags=["optimization", "iterative", "refinement"],
        )

        self.register_template(simple_template)
        self.register_template(complex_template)
        self.register_template(optimization_template)

    def register_template(self, template: WorkflowTemplate) -> None:
        """Register a workflow template."""
        self._templates[template.template_id] = template
        logger.debug(f"Workflow template registered: {template.template_id}")

    def get_template(self, template_id: str) -> Optional[WorkflowTemplate]:
        """Get workflow template by ID."""
        return self._templates.get(template_id)

    def find_matching_templates(
        self, context: Dict[str, Any]
    ) -> List[WorkflowTemplate]:
        """Find templates that match the given context."""
        matching_templates = []

        for template in self._templates.values():
            if template.can_execute(context):
                matching_templates.append(template)

        # Sort by priority and estimated duration
        matching_templates.sort(
            key=lambda t: (t.default_priority.value, t.estimated_duration)
        )

        return matching_templates

    def get_all_templates(self) -> List[WorkflowTemplate]:
        """Get all registered templates."""
        return list(self._templates.values())


# Global instances
_workflow_engines: Dict[str, WorkflowExecutionEngine] = {}
_template_registry: Optional[WorkflowTemplateRegistry] = None


def get_workflow_engine(session_id: str) -> WorkflowExecutionEngine:
    """Get workflow execution engine for a session."""
    global _workflow_engines  # noqa: PLW0602

    if session_id not in _workflow_engines:
        _workflow_engines[session_id] = WorkflowExecutionEngine(session_id)

    return _workflow_engines[session_id]


def get_workflow_template_registry() -> WorkflowTemplateRegistry:
    """Get global workflow template registry."""
    global _template_registry

    if _template_registry is None:
        _template_registry = WorkflowTemplateRegistry()

    return _template_registry


async def cleanup_workflow_engine(session_id: str) -> None:
    """Cleanup workflow engine for a session."""
    global _workflow_engines  # noqa: PLW0602

    if session_id in _workflow_engines:
        engine = _workflow_engines[session_id]
        await engine.cleanup()
        del _workflow_engines[session_id]

        logger.info(f"Workflow engine cleaned up for session {session_id}")


async def cleanup_all_workflow_engines() -> None:
    """Cleanup all workflow engines."""
    global _workflow_engines  # noqa: PLW0602

    for session_id in list(_workflow_engines.keys()):
        await cleanup_workflow_engine(session_id)

    _workflow_engines.clear()
    logger.info("All workflow engines cleaned up")


# Convenience functions for common workflow patterns


async def execute_simple_trip_workflow(
    destination: str,
    duration_days: int,
    session_id: str,
    budget: Optional[str] = None,
) -> WorkflowExecution:
    """Execute simple trip planning workflow."""

    from src.ai_services.agent_orchestrator import create_quick_trip_planning_workflow

    workflow_def = create_quick_trip_planning_workflow()
    trip_req = TripRequirements(
        destination=destination,
        duration_days=duration_days,
        budget_range=budget,
    )

    engine = get_workflow_engine(session_id)
    await engine.initialize()

    return await engine.execute_workflow(workflow_def, trip_req)


async def execute_comprehensive_trip_workflow(
    destination: str,
    duration_days: int,
    traveler_count: int,
    session_id: str,
    budget: Optional[str] = None,
    special_requirements: Optional[List[str]] = None,
) -> WorkflowExecution:
    """Execute comprehensive trip planning workflow."""

    from src.ai_services.agent_orchestrator import (
        create_comprehensive_trip_planning_workflow,
    )

    workflow_def = create_comprehensive_trip_planning_workflow()
    trip_req = TripRequirements(
        destination=destination,
        duration_days=duration_days,
        traveler_count=traveler_count,
        budget_range=budget,
        special_requirements=special_requirements or [],
    )

    engine = get_workflow_engine(session_id)
    await engine.initialize()

    return await engine.execute_workflow(workflow_def, trip_req, WorkflowPriority.HIGH)
