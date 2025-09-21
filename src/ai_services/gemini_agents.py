"""Basic Gemini Agent Foundation for AI-Powered Trip Planner Backend.

This module provides base agent class with Google ADK integration, basic LLM agent
for Gemini model interaction, agent communication patterns, state management,
and function tool integration framework.
"""

import asyncio
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from src.ai_services.exceptions import (
    AgentCommunicationError,
    AgentError,
    AgentStateError,
    FunctionCallError,
    ModelResponseError,
)
from src.ai_services.function_tools import get_tool_registry
from src.ai_services.model_config import AsyncVertexAIClient, get_async_client
from src.ai_services.prompt_templates import PromptType, get_context_aware_prompt
from src.ai_services.session_manager import (
    MessageRole,
    get_session_manager,
)
from src.core.logging import get_logger

logger = get_logger(__name__)


class AgentState(str, Enum):
    """Agent states."""

    INITIALIZING = "initializing"
    READY = "ready"
    PROCESSING = "processing"
    WAITING_FOR_TOOL = "waiting_for_tool"
    ERROR = "error"
    TERMINATED = "terminated"


class AgentRole(str, Enum):
    """Agent roles in the system."""

    TRIP_PLANNER = "trip_planner"
    DESTINATION_EXPERT = "destination_expert"
    ACTIVITY_RECOMMENDER = "activity_recommender"
    BUDGET_ADVISOR = "budget_advisor"
    ITINERARY_OPTIMIZER = "itinerary_optimizer"
    LOCAL_GUIDE = "local_guide"
    SAFETY_ADVISOR = "safety_advisor"
    WEATHER_ANALYST = "weather_analyst"
    TRANSPORT_PLANNER = "transport_planner"
    ACCOMMODATION_FINDER = "accommodation_finder"
    INFORMATION_GATHERER = "information_gatherer"
    ITINERARY_PLANNER = "itinerary_planner"
    OPTIMIZATION_AGENT = "optimization_agent"
    ROUTE_PLANNER = "route_planner"


@dataclass
class AgentMessage:
    """Message between agents."""

    sender_id: str
    receiver_id: str
    content: str
    message_type: str = "text"
    metadata: Optional[Dict[str, Any]] = None
    timestamp: Optional[datetime] = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)


class AgentCapabilities(BaseModel):
    """Agent capabilities definition."""

    role: AgentRole
    prompt_type: PromptType
    supported_functions: List[str] = Field(default_factory=list)
    can_delegate: bool = False
    max_iterations: int = 10
    timeout_seconds: int = 300
    requires_context: List[str] = Field(default_factory=list)
    output_format: str = "text"

    def supports_function(self, function_name: str) -> bool:
        """Check if agent supports a function."""
        return function_name in self.supported_functions or not self.supported_functions

    def validate_context(self, context: Dict[str, Any]) -> bool:
        """Validate if required context is available."""
        return all(key in context for key in self.requires_context)


class BaseAgent(ABC):
    """Abstract base class for all AI agents."""

    def __init__(
        self,
        agent_id: str,
        capabilities: AgentCapabilities,
        session_id: Optional[str] = None,
    ) -> None:
        """Initialize base agent.

        Args:
            agent_id: Unique agent identifier
            capabilities: Agent capabilities
            session_id: Optional session ID for conversation context
        """
        self.agent_id = agent_id
        self.capabilities = capabilities
        self.session_id = session_id
        self.state = AgentState.INITIALIZING
        self.created_at = datetime.now(timezone.utc)
        self.last_activity = self.created_at
        self.error_count = 0
        self.message_count = 0

        # Components
        self.session_manager = get_session_manager()
        self.tool_registry = get_tool_registry()

        # State tracking
        self.current_task: Optional[str] = None
        self.context: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {}

        logger.debug(
            "Agent initialized",
            agent_id=self.agent_id,
            role=self.capabilities.role.value,
            session_id=self.session_id,
        )

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize agent resources."""

    @abstractmethod
    async def process_message(
        self, message: str, context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Process incoming message and generate response."""

    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup agent resources."""

    def update_state(self, new_state: AgentState, error: Optional[str] = None) -> None:
        """Update agent state.

        Args:
            new_state: New agent state
            error: Optional error message if state is ERROR
        """
        old_state = self.state
        self.state = new_state
        self.last_activity = datetime.now(timezone.utc)

        if new_state == AgentState.ERROR:
            self.error_count += 1
            if error:
                self.metadata["last_error"] = error

        logger.debug(
            "Agent state updated",
            agent_id=self.agent_id,
            old_state=old_state.value,
            new_state=new_state.value,
            error=error,
        )

    def update_context(self, **context_updates) -> None:
        """Update agent context.

        Args:
            **context_updates: Context updates
        """
        self.context.update(context_updates)
        self.last_activity = datetime.now(timezone.utc)

    async def execute_function(
        self, function_name: str, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute function tool.

        Args:
            function_name: Name of function to execute
            parameters: Function parameters

        Returns:
            Dict[str, Any]: Function result

        Raises:
            FunctionCallError: If function execution fails
        """
        if not self.capabilities.supports_function(function_name):
            raise FunctionCallError(
                f"Agent {self.agent_id} does not support function {function_name}",
                function_name=function_name,
                function_args=parameters,
            )

        try:
            self.update_state(AgentState.WAITING_FOR_TOOL)

            result = await self.tool_registry.execute_tool(function_name, **parameters)

            self.update_state(AgentState.PROCESSING)

            logger.debug(
                "Function executed by agent",
                agent_id=self.agent_id,
                function_name=function_name,
                execution_time=result.get("execution_time"),
            )

            return result

        except Exception as e:
            self.update_state(AgentState.ERROR, str(e))
            raise FunctionCallError(
                f"Function execution failed: {e}",
                function_name=function_name,
                function_args=parameters,
            ) from e

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics.

        Returns:
            Dict[str, Any]: Agent statistics
        """
        uptime = datetime.now(timezone.utc) - self.created_at

        return {
            "agent_id": self.agent_id,
            "role": self.capabilities.role.value,
            "state": self.state.value,
            "session_id": self.session_id,
            "uptime_seconds": int(uptime.total_seconds()),
            "message_count": self.message_count,
            "error_count": self.error_count,
            "current_task": self.current_task,
            "last_activity": self.last_activity.isoformat(),
            "context_keys": list(self.context.keys()),
        }


class GeminiAgent(BaseAgent):
    """Gemini-powered LLM agent."""

    def __init__(
        self,
        agent_id: str,
        capabilities: AgentCapabilities,
        session_id: Optional[str] = None,
        model_client: Optional[AsyncVertexAIClient] = None,
    ) -> None:
        """Initialize Gemini agent.

        Args:
            agent_id: Unique agent identifier
            capabilities: Agent capabilities
            session_id: Optional session ID
            model_client: Optional model client (will create if not provided)
        """
        super().__init__(agent_id, capabilities, session_id)
        self.model_client = model_client or get_async_client()
        self.system_prompt: Optional[str] = None
        self.conversation_history: List[Dict[str, Any]] = []
        self.function_call_history: List[Dict[str, Any]] = []

    async def initialize(self) -> None:
        """Initialize Gemini agent."""
        try:
            # Initialize model client
            await self.model_client.initialize()

            # Generate system prompt based on role and context
            if self.session_id:
                try:
                    session = await self.session_manager.get_session(self.session_id)
                    user_context = {
                        "user_id": session.user_id,
                        "user_profile": session.context.user_profile,
                        "preferences": session.context.preferences,
                    }
                    trip_context = session.context.trip_context

                    self.system_prompt = get_context_aware_prompt(
                        self.capabilities.prompt_type,
                        user_context,
                        trip_context,
                        **self.context,
                    )

                    # Load existing conversation history
                    self.conversation_history = session.get_conversation_history()

                except Exception as e:
                    logger.warning(
                        "Failed to load session context, using default prompt",
                        agent_id=self.agent_id,
                        session_id=self.session_id,
                        error=str(e),
                    )

            # Test model connectivity
            health_ok = await self.model_client.health_check()
            if not health_ok:
                raise AgentError("Model health check failed")

            self.update_state(AgentState.READY)

            logger.info(
                "Gemini agent initialized successfully",
                agent_id=self.agent_id,
                role=self.capabilities.role.value,
                has_system_prompt=bool(self.system_prompt),
                conversation_history_length=len(self.conversation_history),
            )

        except Exception as e:
            self.update_state(AgentState.ERROR, str(e))
            raise AgentError(f"Failed to initialize Gemini agent: {e}") from e

    async def process_message(
        self, message: str, context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Process message using Gemini model.

        Args:
            message: Input message
            context: Optional additional context

        Returns:
            str: Agent response

        Raises:
            AgentCommunicationError: If processing fails
        """
        if self.state not in {AgentState.READY, AgentState.PROCESSING}:
            raise AgentStateError(
                f"Agent {self.agent_id} not ready for processing (state: {self.state})",
                agent_id=self.agent_id,
                agent_state=self.state.value,
            )

        try:
            self.update_state(AgentState.PROCESSING)
            self.message_count += 1

            # Update context if provided
            if context:
                self.update_context(**context)

            # Build prompt with conversation history
            full_prompt = self._build_conversation_prompt(message)

            # Generate response
            response = await self._generate_response(full_prompt)

            # Process potential function calls
            if self._contains_function_call(response):
                response = await self._handle_function_calls(response)

            # Update conversation history
            await self._update_conversation_history(message, response)

            self.update_state(AgentState.READY)

            logger.debug(
                "Message processed by Gemini agent",
                agent_id=self.agent_id,
                message_length=len(message),
                response_length=len(response),
            )

            return response

        except Exception as e:
            self.update_state(AgentState.ERROR, str(e))
            raise AgentCommunicationError(
                f"Failed to process message: {e}", agent_id=self.agent_id
            ) from e

    async def process_streaming(
        self, message: str, context: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[str, None]:
        """Process message with streaming response.

        Args:
            message: Input message
            context: Optional additional context

        Yields:
            str: Response chunks
        """
        if self.state != AgentState.READY:
            raise AgentStateError(
                f"Agent {self.agent_id} not ready for streaming",
                agent_id=self.agent_id,
                agent_state=self.state.value,
            )

        try:
            self.update_state(AgentState.PROCESSING)

            # Update context
            if context:
                self.update_context(**context)

            # Build prompt
            full_prompt = self._build_conversation_prompt(message)

            # Generate streaming response
            stream = await self.model_client.generate_content_stream(full_prompt)

            complete_response = ""
            async for chunk in stream:
                if hasattr(chunk, "text") and chunk.text:
                    complete_response += chunk.text
                    yield chunk.text

            # Update conversation history with complete response
            await self._update_conversation_history(message, complete_response)

            self.update_state(AgentState.READY)

        except Exception as e:
            self.update_state(AgentState.ERROR, str(e))
            raise AgentCommunicationError(
                f"Streaming processing failed: {e}", agent_id=self.agent_id
            ) from e

    def _build_conversation_prompt(self, current_message: str) -> str:
        """Build conversation prompt with history and context.

        Args:
            current_message: Current user message

        Returns:
            str: Complete conversation prompt
        """
        prompt_parts = []

        # Add system prompt
        if self.system_prompt:
            prompt_parts.append(f"SYSTEM: {self.system_prompt}")

        # Add conversation history (limited to recent messages)
        max_history = 10
        recent_history = self.conversation_history[-max_history:]

        for msg in recent_history:
            role = msg.get("role", "user").upper()
            content = msg.get("content", "")
            prompt_parts.append(f"{role}: {content}")

        # Add current message
        prompt_parts.append(f"USER: {current_message}")
        prompt_parts.append("ASSISTANT:")

        return "\n\n".join(prompt_parts)

    async def _generate_response(self, prompt: str) -> str:
        """Generate response using Gemini model.

        Args:
            prompt: Input prompt

        Returns:
            str: Model response

        Raises:
            ModelResponseError: If generation fails
        """
        try:
            response = await self.model_client.generate_content(prompt)

            if not response or not response.strip():
                raise ModelResponseError("Empty response from model")

            return response.strip()

        except Exception as e:
            raise ModelResponseError(f"Response generation failed: {e}") from e

    def _contains_function_call(self, response: str) -> bool:
        """Check if response contains function calls.

        Args:
            response: Model response

        Returns:
            bool: True if contains function calls
        """
        # Simple heuristic - look for function call patterns
        function_patterns = [
            "```function_call",
            "FUNCTION_CALL:",
            "call_function(",
            "use_tool(",
        ]

        return any(pattern in response for pattern in function_patterns)

    async def _handle_function_calls(self, response: str) -> str:
        """Handle function calls in response.

        Args:
            response: Response with function calls

        Returns:
            str: Response with function results
        """
        try:
            # Extract function calls (simplified implementation)
            # In a real implementation, you'd parse structured function calls
            function_calls = self._extract_function_calls(response)

            results = []
            for func_call in function_calls:
                try:
                    result = await self.execute_function(
                        func_call["name"], func_call["parameters"]
                    )

                    self.function_call_history.append(
                        {
                            "function": func_call["name"],
                            "parameters": func_call["parameters"],
                            "result": result,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                    )

                    results.append(
                        {"function": func_call["name"], "result": result["result"]}
                    )

                except Exception as e:
                    logger.exception(
                        "Function call failed",
                        agent_id=self.agent_id,
                        function=func_call["name"],
                    )
                    results.append({"function": func_call["name"], "error": str(e)})

            # Combine original response with function results
            function_results_text = self._format_function_results(results)

            return f"{response}\n\n{function_results_text}"

        except Exception:
            logger.exception("Error handling function calls", agent_id=self.agent_id)
            return response

    def _extract_function_calls(self, response: str) -> List[Dict[str, Any]]:
        """Extract function calls from response.

        Args:
            response: Model response

        Returns:
            List[Dict[str, Any]]: Extracted function calls
        """
        # Simplified extraction - in production, use proper parsing
        function_calls = []

        # Look for patterns like "call_function(name, {params})"
        import re

        pattern = r"call_function\(([^,]+),\s*(\{[^}]+\})\)"
        matches = re.findall(pattern, response)

        for match in matches:
            try:
                function_name = match[0].strip().strip("\"'")
                parameters = json.loads(match[1])

                function_calls.append({"name": function_name, "parameters": parameters})

            except (json.JSONDecodeError, IndexError):
                continue

        return function_calls

    def _format_function_results(self, results: List[Dict[str, Any]]) -> str:
        """Format function execution results.

        Args:
            results: Function execution results

        Returns:
            str: Formatted results
        """
        if not results:
            return ""

        formatted_parts = ["**Function Execution Results:**"]

        for result in results:
            function_name = result.get("function", "unknown")

            if "error" in result:
                formatted_parts.append(f"- {function_name}: Error - {result['error']}")
            else:
                result_data = result.get("result", "No result")
                formatted_parts.append(f"- {function_name}: {result_data}")

        return "\n".join(formatted_parts)

    async def _update_conversation_history(
        self, user_message: str, assistant_response: str
    ) -> None:
        """Update conversation history in session.

        Args:
            user_message: User message
            assistant_response: Assistant response
        """
        try:
            if self.session_id:
                # Add messages to session
                await self.session_manager.add_message(
                    self.session_id,
                    MessageRole.USER,
                    user_message,
                    agent_id=self.agent_id,
                )

                await self.session_manager.add_message(
                    self.session_id,
                    MessageRole.ASSISTANT,
                    assistant_response,
                    agent_id=self.agent_id,
                )

            # Update local history
            self.conversation_history.extend(
                [
                    {
                        "role": "user",
                        "content": user_message,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                    {
                        "role": "assistant",
                        "content": assistant_response,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                ]
            )

            # Limit history size
            max_history = 50
            if len(self.conversation_history) > max_history:
                # Keep system messages and recent messages
                system_msgs = [
                    m for m in self.conversation_history if m.get("role") == "system"
                ]
                recent_msgs = [
                    m for m in self.conversation_history if m.get("role") != "system"
                ][-max_history + len(system_msgs) :]
                self.conversation_history = system_msgs + recent_msgs

        except Exception as e:
            logger.warning(
                "Failed to update conversation history",
                agent_id=self.agent_id,
                session_id=self.session_id,
                error=str(e),
            )

    async def cleanup(self) -> None:
        """Cleanup agent resources."""
        try:
            # Save any pending state
            if self.session_id and self.conversation_history:
                try:
                    session = await self.session_manager.get_session(self.session_id)
                    session.update_context(
                        agent_conversation_history=self.conversation_history,
                        agent_function_calls=self.function_call_history,
                    )
                    await self.session_manager.update_session(session)
                except Exception as e:
                    logger.warning(
                        "Failed to save agent state to session",
                        agent_id=self.agent_id,
                        error=str(e),
                    )

            self.update_state(AgentState.TERMINATED)

            logger.info(
                "Gemini agent cleanup completed",
                agent_id=self.agent_id,
                message_count=self.message_count,
                function_calls=len(self.function_call_history),
            )

        except Exception as e:
            logger.error(
                "Error during agent cleanup",
                agent_id=self.agent_id,
                error=str(e),
                exc_info=True,
            )


class AgentFactory:
    """Factory for creating specialized agents."""

    @staticmethod
    def create_trip_planner_agent(
        user_id: str, session_id: Optional[str] = None, **context
    ) -> GeminiAgent:
        """Create trip planner agent.

        Args:
            user_id: User identifier
            session_id: Optional session ID
            **context: Additional context

        Returns:
            GeminiAgent: Trip planner agent
        """
        capabilities = AgentCapabilities(
            role=AgentRole.TRIP_PLANNER,
            prompt_type=PromptType.TRIP_PLANNER,
            supported_functions=[
                "get_current_datetime",
                "convert_currency",
                "validate_email",
            ],
            can_delegate=True,
            max_iterations=15,
            requires_context=["destination", "travel_dates"],
            output_format="structured_itinerary",
        )

        agent_id = f"trip_planner_{user_id}_{int(datetime.now().timestamp())}"
        agent = GeminiAgent(agent_id, capabilities, session_id)
        agent.update_context(**context)

        return agent

    @staticmethod
    def create_destination_expert_agent(
        destination: str, session_id: Optional[str] = None, **context
    ) -> GeminiAgent:
        """Create destination expert agent.

        Args:
            destination: Destination name
            session_id: Optional session ID
            **context: Additional context

        Returns:
            GeminiAgent: Destination expert agent
        """
        capabilities = AgentCapabilities(
            role=AgentRole.DESTINATION_EXPERT,
            prompt_type=PromptType.DESTINATION_EXPERT,
            supported_functions=[
                "get_current_datetime",
            ],
            requires_context=["destination"],
        )

        agent_id = f"dest_expert_{destination.lower().replace(' ', '_')}_{int(datetime.now().timestamp())}"
        agent = GeminiAgent(agent_id, capabilities, session_id)
        agent.update_context(destination=destination, **context)

        return agent

    @staticmethod
    def create_budget_advisor_agent(
        budget_range: str,
        currency: str = "INR",
        session_id: Optional[str] = None,
        **context,
    ) -> GeminiAgent:
        """Create budget advisor agent.

        Args:
            budget_range: Budget range
            currency: Currency code
            session_id: Optional session ID
            **context: Additional context

        Returns:
            GeminiAgent: Budget advisor agent
        """
        capabilities = AgentCapabilities(
            role=AgentRole.BUDGET_ADVISOR,
            prompt_type=PromptType.BUDGET_ADVISOR,
            supported_functions=[
                "convert_currency",
                "get_current_datetime",
            ],
            requires_context=["budget_range", "currency"],
        )

        agent_id = f"budget_advisor_{int(datetime.now().timestamp())}"
        agent = GeminiAgent(agent_id, capabilities, session_id)
        agent.update_context(budget_range=budget_range, currency=currency, **context)

        return agent

    @staticmethod
    def create_information_gathering_agent(
        destination: str,
        session_id: Optional[str] = None,
        **context,
    ) -> GeminiAgent:
        """Create information gathering agent for parallel data collection.

        This agent specializes in collecting destination info, weather, events,
        and other travel-related data using Maps and external APIs.
        """
        capabilities = AgentCapabilities(
            role=AgentRole.INFORMATION_GATHERER,
            prompt_type=PromptType.DESTINATION_EXPERT,
            supported_functions=[
                "find_places",
                "geocode_location",
                "find_nearby_places",
                "get_place_details",
                "validate_location",
                "get_current_datetime",
            ],
            can_delegate=False,
            max_iterations=10,
            requires_context=["destination"],
            output_format="structured_data",
        )

        agent_id = f"info_gatherer_{destination.lower().replace(' ', '_')}_{int(datetime.now().timestamp())}"
        agent = GeminiAgent(agent_id, capabilities, session_id)
        agent.update_context(
            destination=destination, role="information_gathering", **context
        )

        return agent

    @staticmethod
    def create_itinerary_planning_agent(
        trip_duration: int,
        session_id: Optional[str] = None,
        **context,
    ) -> GeminiAgent:
        """Create itinerary planning agent for structured daily plans.

        This agent creates day-by-day trip plans using place details and routing.
        """
        capabilities = AgentCapabilities(
            role=AgentRole.ITINERARY_PLANNER,
            prompt_type=PromptType.TRIP_PLANNER,
            supported_functions=[
                "find_places",
                "get_directions",
                "get_travel_time",
                "get_place_details",
                "find_nearby_places",
                "get_current_datetime",
            ],
            can_delegate=True,
            max_iterations=20,
            requires_context=["destination", "trip_duration", "preferences"],
            output_format="structured_itinerary",
        )

        agent_id = (
            f"itinerary_planner_{trip_duration}days_{int(datetime.now().timestamp())}"
        )
        agent = GeminiAgent(agent_id, capabilities, session_id)
        agent.update_context(
            trip_duration=trip_duration, role="itinerary_planning", **context
        )

        return agent

    @staticmethod
    def create_optimization_agent(
        optimization_criteria: List[str],
        session_id: Optional[str] = None,
        **context,
    ) -> GeminiAgent:
        """Create optimization agent for balancing budget, time, and preferences.

        This agent optimizes itineraries for multiple criteria using parallel processing.
        """
        capabilities = AgentCapabilities(
            role=AgentRole.OPTIMIZATION_AGENT,
            prompt_type=PromptType.ITINERARY_OPTIMIZER,
            supported_functions=[
                "get_travel_time",
                "get_directions",
                "convert_currency",
                "find_nearby_places",
                "get_current_datetime",
            ],
            can_delegate=False,
            max_iterations=15,
            requires_context=["itinerary_data", "optimization_criteria"],
            output_format="optimized_itinerary",
        )

        agent_id = f"optimizer_{len(optimization_criteria)}criteria_{int(datetime.now().timestamp())}"
        agent = GeminiAgent(agent_id, capabilities, session_id)
        agent.update_context(
            optimization_criteria=optimization_criteria, role="optimization", **context
        )

        return agent

    @staticmethod
    def create_route_planning_agent(
        transportation_modes: List[str],
        session_id: Optional[str] = None,
        **context,
    ) -> GeminiAgent:
        """Create route planning agent for transportation optimization.

        This agent optimizes travel routes and transportation between destinations.
        """
        capabilities = AgentCapabilities(
            role=AgentRole.ROUTE_PLANNER,
            prompt_type=PromptType.TRANSPORT_PLANNER,
            supported_functions=[
                "get_directions",
                "get_travel_time",
                "find_nearby_places",
                "geocode_location",
                "validate_location",
                "get_current_datetime",
            ],
            can_delegate=False,
            max_iterations=12,
            requires_context=["locations", "transportation_modes"],
            output_format="route_plan",
        )

        agent_id = f"route_planner_{len(transportation_modes)}modes_{int(datetime.now().timestamp())}"
        agent = GeminiAgent(agent_id, capabilities, session_id)
        agent.update_context(
            transportation_modes=transportation_modes, role="route_planning", **context
        )

        return agent

    @staticmethod
    def create_enhanced_destination_expert_agent(
        destination: str,
        expertise_areas: List[str],
        session_id: Optional[str] = None,
        **context,
    ) -> GeminiAgent:
        """Create enhanced destination expert with specialized knowledge areas.

        This agent provides deep destination insights with Maps integration.
        """
        capabilities = AgentCapabilities(
            role=AgentRole.DESTINATION_EXPERT,
            prompt_type=PromptType.DESTINATION_EXPERT,
            supported_functions=[
                "find_places",
                "find_nearby_places",
                "get_place_details",
                "geocode_location",
                "validate_location",
                "get_current_datetime",
            ],
            can_delegate=False,
            max_iterations=8,
            requires_context=["destination", "expertise_areas"],
            output_format="destination_insights",
        )

        agent_id = f"dest_expert_enhanced_{destination.lower().replace(' ', '_')}_{int(datetime.now().timestamp())}"
        agent = GeminiAgent(agent_id, capabilities, session_id)
        agent.update_context(
            destination=destination,
            expertise_areas=expertise_areas,
            role="destination_expert_enhanced",
            **context,
        )

        return agent

    @staticmethod
    def create_enhanced_budget_advisor_agent(
        budget_components: Dict[str, Any],
        currency: str = "INR",
        session_id: Optional[str] = None,
        **context,
    ) -> GeminiAgent:
        """Create enhanced budget advisor with detailed cost analysis.

        This agent provides comprehensive budget optimization and tracking.
        """
        capabilities = AgentCapabilities(
            role=AgentRole.BUDGET_ADVISOR,
            prompt_type=PromptType.BUDGET_ADVISOR,
            supported_functions=[
                "convert_currency",
                "get_current_datetime",
                "find_places",  # For price research
                "get_place_details",  # For cost information
            ],
            can_delegate=False,
            max_iterations=10,
            requires_context=["budget_components", "currency"],
            output_format="budget_analysis",
        )

        agent_id = (
            f"budget_advisor_enhanced_{currency}_{int(datetime.now().timestamp())}"
        )
        agent = GeminiAgent(agent_id, capabilities, session_id)
        agent.update_context(
            budget_components=budget_components,
            currency=currency,
            role="budget_advisor_enhanced",
            **context,
        )

        return agent


class EnhancedAgentOrchestrator:
    """Enhanced orchestrator for managing specialized trip planning agents with ADK integration."""

    def __init__(self) -> None:
        """Initialize enhanced agent orchestrator."""
        self.agents: Dict[str, BaseAgent] = {}
        self.agent_communications: List[AgentMessage] = []
        self.session_manager = get_session_manager()

        # Agent performance tracking
        self.agent_metrics: Dict[str, Dict[str, Any]] = {}

        # Execution history
        self.execution_history: List[Dict[str, Any]] = []

        # Agent pools by role
        self.agent_pools: Dict[AgentRole, List[BaseAgent]] = {}

    async def register_agent(self, agent: BaseAgent) -> None:
        """Register agent with enhanced tracking and pool management."""
        await agent.initialize()

        self.agents[agent.agent_id] = agent

        # Add to role-based pool
        role = agent.capabilities.role
        if role not in self.agent_pools:
            self.agent_pools[role] = []
        self.agent_pools[role].append(agent)

        # Initialize metrics tracking
        self.agent_metrics[agent.agent_id] = {
            "total_messages": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "average_response_time": 0.0,
            "last_active": datetime.now(timezone.utc).isoformat(),
            "function_calls": 0,
        }

        logger.info(
            "Agent registered with enhanced orchestrator",
            agent_id=agent.agent_id,
            role=role.value,
            pool_size=len(self.agent_pools.get(role, [])),
        )

    async def execute_multi_agent_workflow(
        self,
        workflow_steps: List[Dict[str, Any]],
        context: Dict[str, Any],
        execution_mode: str = "sequential",
    ) -> Dict[str, Any]:
        """Execute a multi-agent workflow with different execution patterns.

        Args:
            workflow_steps: List of workflow step definitions
            context: Execution context
            execution_mode: "sequential", "parallel", or "hybrid"

        Returns:
            Dict containing workflow results and metrics
        """
        start_time = datetime.now(timezone.utc)
        execution_id = str(uuid4())

        logger.info(
            "Starting multi-agent workflow execution",
            execution_id=execution_id,
            steps=len(workflow_steps),
            mode=execution_mode,
        )

        workflow_results = {
            "execution_id": execution_id,
            "start_time": start_time.isoformat(),
            "execution_mode": execution_mode,
            "step_results": [],
            "context_updates": {},
            "metrics": {
                "total_steps": len(workflow_steps),
                "successful_steps": 0,
                "failed_steps": 0,
                "total_execution_time": 0.0,
                "total_tokens": 0,
                "agents_used": set(),
            },
        }

        try:
            if execution_mode == "sequential":
                await self._execute_sequential_workflow(
                    workflow_steps, context, workflow_results
                )
            elif execution_mode == "parallel":
                await self._execute_parallel_workflow(
                    workflow_steps, context, workflow_results
                )
            elif execution_mode == "hybrid":
                await self._execute_hybrid_workflow(
                    workflow_steps, context, workflow_results
                )
            else:
                raise AgentError(f"Unsupported execution mode: {execution_mode}")

            # Calculate final metrics
            end_time = datetime.now(timezone.utc)
            workflow_results["end_time"] = end_time.isoformat()
            workflow_results["metrics"]["total_execution_time"] = (
                end_time - start_time
            ).total_seconds()
            workflow_results["metrics"]["agents_used"] = list(
                workflow_results["metrics"]["agents_used"]
            )

            # Store execution history
            self.execution_history.append(
                {
                    "execution_id": execution_id,
                    "timestamp": start_time.isoformat(),
                    "mode": execution_mode,
                    "success": workflow_results["metrics"]["failed_steps"] == 0,
                    "duration": workflow_results["metrics"]["total_execution_time"],
                    "steps_count": len(workflow_steps),
                }
            )

            logger.info(
                "Multi-agent workflow completed",
                execution_id=execution_id,
                success=workflow_results["metrics"]["failed_steps"] == 0,
                duration=workflow_results["metrics"]["total_execution_time"],
            )

            return workflow_results

        except Exception as e:
            workflow_results["error"] = str(e)
            workflow_results["end_time"] = datetime.now(timezone.utc).isoformat()

            logger.error(
                "Multi-agent workflow failed",
                execution_id=execution_id,
                error=str(e),
                exc_info=True,
            )

            return workflow_results

    async def _execute_sequential_workflow(
        self,
        steps: List[Dict[str, Any]],
        context: Dict[str, Any],
        results: Dict[str, Any],
    ) -> None:
        """Execute workflow steps sequentially."""
        current_context = context.copy()

        for i, step in enumerate(steps):
            step_start = datetime.now(timezone.utc)
            step_result = {
                "step_index": i,
                "step_id": step.get("id", f"step_{i}"),
                "agent_role": step.get("agent_role"),
                "start_time": step_start.isoformat(),
                "success": False,
            }

            try:
                # Get agent for this step
                agent = await self._get_agent_for_step(step)
                if not agent:
                    raise AgentError(f"No agent available for step {step.get('id', i)}")

                # Prepare step message
                message = self._prepare_step_message(step, current_context)

                # Execute step
                response = await agent.process_message(message, current_context)

                # Process response
                step_result.update(
                    {
                        "success": True,
                        "agent_id": agent.agent_id,
                        "response": response,
                        "execution_time": (
                            datetime.now(timezone.utc) - step_start
                        ).total_seconds(),
                    }
                )

                # Update context with response - handle both dict and string responses safely
                response_data = self._process_agent_response(response)
                if response_data:
                    current_context.update(response_data)
                    results["context_updates"].update(response_data)

                results["metrics"]["successful_steps"] += 1
                results["metrics"]["agents_used"].add(agent.agent_id)

            except Exception as e:
                step_result.update(
                    {
                        "error": str(e),
                        "execution_time": (
                            datetime.now(timezone.utc) - step_start
                        ).total_seconds(),
                    }
                )
                results["metrics"]["failed_steps"] += 1

                logger.exception(f"Step {step.get('id', i)} failed")

                # Check if step is required
                if step.get("required", True):
                    raise AgentError(
                        f"Required step {step.get('id', i)} failed: {e}"
                    ) from e

            step_result["end_time"] = datetime.now(timezone.utc).isoformat()
            results["step_results"].append(step_result)

    async def _execute_parallel_workflow(
        self,
        steps: List[Dict[str, Any]],
        context: Dict[str, Any],
        results: Dict[str, Any],
    ) -> None:
        """Execute workflow steps in parallel."""
        # Create tasks for all steps
        tasks = []
        for i, step in enumerate(steps):
            task = asyncio.create_task(self._execute_single_step(step, context, i))
            tasks.append(task)

        # Wait for all tasks to complete
        step_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results - handle both successful results and exceptions
        for i, result in enumerate(step_results):
            if isinstance(result, Exception):
                error_result = {
                    "step_index": i,
                    "step_id": steps[i].get("id", f"step_{i}"),
                    "success": False,
                    "error": str(result),
                    "execution_time": 0.0,
                }
                results["metrics"]["failed_steps"] += 1
                results["step_results"].append(error_result)
            elif isinstance(result, dict):
                # Handle successful step result
                if result.get("success", False):
                    results["metrics"]["successful_steps"] += 1
                    agent_id = result.get("agent_id")
                    if agent_id:
                        results["metrics"]["agents_used"].add(agent_id)

                    # Update context with response
                    response = result.get("response")
                    if response:
                        response_data = self._process_agent_response(response)
                        if response_data:
                            results["context_updates"].update(response_data)
                else:
                    results["metrics"]["failed_steps"] += 1

                results["step_results"].append(result)
            else:
                # Handle unexpected result type
                error_result = {
                    "step_index": i,
                    "step_id": steps[i].get("id", f"step_{i}"),
                    "success": False,
                    "error": f"Unexpected result type: {type(result)}",
                    "execution_time": 0.0,
                }
                results["metrics"]["failed_steps"] += 1
                results["step_results"].append(error_result)

    async def _execute_hybrid_workflow(
        self,
        steps: List[Dict[str, Any]],
        context: Dict[str, Any],
        results: Dict[str, Any],
    ) -> None:
        """Execute workflow with mixed sequential and parallel execution."""
        current_context = context.copy()
        i = 0

        while i < len(steps):
            # Find consecutive parallel steps
            parallel_steps = []

            # Collect steps that can run in parallel
            while i < len(steps) and steps[i].get("execution_mode") == "parallel":
                parallel_steps.append((i, steps[i]))
                i += 1

            # Execute parallel steps
            if parallel_steps:
                tasks = []
                for step_index, step in parallel_steps:
                    task = asyncio.create_task(
                        self._execute_single_step(step, current_context, step_index)
                    )
                    tasks.append(task)

                parallel_results = await asyncio.gather(*tasks, return_exceptions=True)

                # Process parallel results
                for j, result in enumerate(parallel_results):
                    if isinstance(result, Exception):
                        error_result = {
                            "step_index": parallel_steps[j][0],
                            "step_id": parallel_steps[j][1].get(
                                "id", f"step_{parallel_steps[j][0]}"
                            ),
                            "success": False,
                            "error": str(result),
                        }
                        results["metrics"]["failed_steps"] += 1
                        results["step_results"].append(error_result)
                    elif isinstance(result, dict):
                        if result.get("success", False):
                            results["metrics"]["successful_steps"] += 1
                            agent_id = result.get("agent_id")
                            if agent_id:
                                results["metrics"]["agents_used"].add(agent_id)

                            # Update context
                            response = result.get("response")
                            if response:
                                response_data = self._process_agent_response(response)
                                if response_data:
                                    current_context.update(response_data)
                                    results["context_updates"].update(response_data)
                        else:
                            results["metrics"]["failed_steps"] += 1

                        results["step_results"].append(result)

            # Execute sequential steps
            while (
                i < len(steps)
                and steps[i].get("execution_mode", "sequential") == "sequential"
            ):
                step = steps[i]
                step_start = datetime.now(timezone.utc)
                step_result = {
                    "step_index": i,
                    "step_id": step.get("id", f"step_{i}"),
                    "agent_role": step.get("agent_role"),
                    "start_time": step_start.isoformat(),
                    "success": False,
                }

                try:
                    agent = await self._get_agent_for_step(step)
                    if not agent:
                        raise AgentError(
                            f"No agent available for step {step.get('id', i)}"
                        )

                    message = self._prepare_step_message(step, current_context)
                    response = await agent.process_message(message, current_context)

                    step_result.update(
                        {
                            "success": True,
                            "agent_id": agent.agent_id,
                            "response": response,
                            "execution_time": (
                                datetime.now(timezone.utc) - step_start
                            ).total_seconds(),
                        }
                    )

                    # Update context
                    response_data = self._process_agent_response(response)
                    if response_data:
                        current_context.update(response_data)
                        results["context_updates"].update(response_data)

                    results["metrics"]["successful_steps"] += 1
                    results["metrics"]["agents_used"].add(agent.agent_id)

                except Exception as e:
                    step_result.update(
                        {
                            "error": str(e),
                            "execution_time": (
                                datetime.now(timezone.utc) - step_start
                            ).total_seconds(),
                        }
                    )
                    results["metrics"]["failed_steps"] += 1

                    if step.get("required", True):
                        raise AgentError(
                            f"Required step {step.get('id', i)} failed: {e}"
                        ) from e

                step_result["end_time"] = datetime.now(timezone.utc).isoformat()
                results["step_results"].append(step_result)
                i += 1

    async def _execute_single_step(
        self, step: Dict[str, Any], context: Dict[str, Any], step_index: int
    ) -> Dict[str, Any]:
        """Execute a single workflow step."""
        step_start = datetime.now(timezone.utc)
        step_result = {
            "step_index": step_index,
            "step_id": step.get("id", f"step_{step_index}"),
            "agent_role": step.get("agent_role"),
            "start_time": step_start.isoformat(),
            "success": False,
        }

        try:
            agent = await self._get_agent_for_step(step)
            if not agent:
                raise AgentError(
                    f"No agent available for step {step.get('id', step_index)}"
                )

            message = self._prepare_step_message(step, context)
            response = await agent.process_message(message, context)

            step_result.update(
                {
                    "success": True,
                    "agent_id": agent.agent_id,
                    "response": response,
                    "execution_time": (
                        datetime.now(timezone.utc) - step_start
                    ).total_seconds(),
                }
            )

        except Exception as e:
            step_result.update(
                {
                    "error": str(e),
                    "execution_time": (
                        datetime.now(timezone.utc) - step_start
                    ).total_seconds(),
                }
            )

        step_result["end_time"] = datetime.now(timezone.utc).isoformat()
        return step_result

    async def _get_agent_for_step(self, step: Dict[str, Any]) -> Optional[BaseAgent]:
        """Get appropriate agent for a workflow step."""
        agent_role_str = step.get("agent_role")
        if not agent_role_str:
            return None

        try:
            agent_role = AgentRole(agent_role_str)
        except ValueError:
            logger.warning(f"Unknown agent role: {agent_role_str}")
            return None

        # Get agent from pool
        if self.agent_pools.get(agent_role):
            # Use round-robin or load balancing logic here
            return self.agent_pools[agent_role][0]

        return None

    def _prepare_step_message(
        self, step: Dict[str, Any], context: Dict[str, Any]
    ) -> str:
        """Prepare message for a workflow step based on context and step requirements."""
        step_instruction = step.get("instruction", "")
        step_context = step.get("context", {})

        # Merge context
        merged_context = {**context, **step_context}

        # Create context-aware message
        if step_instruction:
            message = (
                f"{step_instruction}\n\nContext: {json.dumps(merged_context, indent=2)}"
            )
        else:
            message = (
                f"Process the following context: {json.dumps(merged_context, indent=2)}"
            )

        return message

    def _process_agent_response(self, response: Any) -> Optional[Dict[str, Any]]:
        """Process agent response and convert to dict format for context updates."""
        if isinstance(response, dict):
            return response
        elif isinstance(response, str):
            # Try to parse as JSON first
            try:
                parsed = json.loads(response)
                if isinstance(parsed, dict):
                    return parsed
                else:
                    return {"last_response": response, "parsed_data": parsed}
            except (json.JSONDecodeError, TypeError):
                return {"last_response": response}
        else:
            # Handle other types
            return {"last_response": str(response)}

    def get_enhanced_stats(self) -> Dict[str, Any]:
        """Get comprehensive orchestrator statistics."""
        total_agents = len(self.agents)
        active_agents = sum(
            1 for agent in self.agents.values() if agent.state == AgentState.READY
        )

        role_distribution = {}
        for role, agents_list in self.agent_pools.items():
            role_distribution[role.value] = len(agents_list)

        return {
            "total_agents": total_agents,
            "active_agents": active_agents,
            "agents_by_role": role_distribution,
            "total_executions": len(self.execution_history),
            "recent_executions": (
                self.execution_history[-10:] if self.execution_history else []
            ),
            "agent_metrics": self.agent_metrics,
            "communication_messages": len(self.agent_communications),
        }

    async def cleanup_all_agents(self) -> None:
        """Cleanup all registered agents."""
        for agent_id, agent in self.agents.items():
            try:
                await agent.cleanup()
                logger.debug("Agent cleaned up", agent_id=agent_id)
            except Exception:
                logger.exception("Error cleaning up agent", agent_id=agent_id)

        self.agents.clear()
        self.agent_pools.clear()
        logger.info("All enhanced agents cleaned up")


# Global orchestrator instance
_orchestrator: Optional[EnhancedAgentOrchestrator] = None


def get_agent_orchestrator() -> EnhancedAgentOrchestrator:
    """Get global enhanced agent orchestrator instance.

    Returns:
        EnhancedAgentOrchestrator: Orchestrator instance
    """
    global _orchestrator

    if _orchestrator is None:
        _orchestrator = EnhancedAgentOrchestrator()

    return _orchestrator


async def cleanup_agents() -> None:
    """Cleanup global enhanced agent orchestrator."""
    global _orchestrator

    if _orchestrator:
        await _orchestrator.cleanup_all_agents()
        _orchestrator = None
