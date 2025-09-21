"""Agent Factory System for AI-Powered Trip Planner Backend - Google ADK Multi-Agent System.

This module provides factory patterns for creating specialized agents, agent configuration
and customization based on trip requirements, dynamic agent selection based on user
preferences and trip complexity, agent lifecycle management and resource allocation,
and agent capabilities registry and discovery.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from src.ai_services.exceptions import AgentError
from src.ai_services.function_tools import get_tool_registry
from src.ai_services.gemini_agents import (
    AgentCapabilities,
    AgentRole,
    BaseAgent,
    GeminiAgent,
)
from src.ai_services.prompt_templates import PromptType
from src.core.logging import get_logger

logger = get_logger(__name__)


class TripComplexity(str, Enum):
    """Trip complexity levels for agent selection."""

    SIMPLE = "simple"  # Single destination, short duration
    MODERATE = "moderate"  # Multiple destinations or longer duration
    COMPLEX = "complex"  # Multiple destinations, long duration, special requirements
    ENTERPRISE = "enterprise"  # Business travel with specific needs


class AgentPriority(str, Enum):
    """Agent execution priority levels."""

    CRITICAL = "critical"  # Must execute successfully
    HIGH = "high"  # Important for trip quality
    MEDIUM = "medium"  # Nice to have
    LOW = "low"  # Optional enhancements


class ResourceTier(str, Enum):
    """Resource allocation tiers for agents."""

    PREMIUM = "premium"  # High-performance resources
    STANDARD = "standard"  # Normal resource allocation
    ECONOMY = "economy"  # Limited resource allocation


@dataclass
class TripRequirements:
    """Trip requirements for agent selection and configuration."""

    destination: str
    duration_days: int
    traveler_count: int = 1
    budget_range: Optional[str] = None
    trip_type: str = "leisure"
    complexity: TripComplexity = TripComplexity.SIMPLE
    special_requirements: List[str] = field(default_factory=list)
    preferred_activities: List[str] = field(default_factory=list)
    transportation_modes: List[str] = field(default_factory=list)
    accommodation_preferences: Dict[str, Any] = field(default_factory=dict)
    dietary_restrictions: List[str] = field(default_factory=list)
    accessibility_needs: List[str] = field(default_factory=list)
    language_preferences: List[str] = field(default_factory=lambda: ["en"])


@dataclass
class AgentConfiguration:
    """Configuration for agent creation and behavior."""

    agent_type: AgentRole
    priority: AgentPriority
    resource_tier: ResourceTier
    supported_functions: List[str] = field(default_factory=list)
    max_iterations: int = 10
    timeout_seconds: int = 300
    requires_context: List[str] = field(default_factory=list)
    output_format: str = "structured"
    custom_parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[AgentRole] = field(default_factory=list)


class AgentCapabilityRegistry:
    """Registry for agent capabilities and specializations."""

    def __init__(self) -> None:
        self._capabilities: Dict[AgentRole, Dict[str, Any]] = {}
        self._specializations: Dict[str, List[AgentRole]] = {}
        self._function_mappings: Dict[str, List[AgentRole]] = {}
        self._register_default_capabilities()

    def _register_default_capabilities(self) -> None:
        """Register default agent capabilities."""

        # Trip Planner capabilities
        self.register_capability(
            AgentRole.TRIP_PLANNER,
            functions=[
                "find_places",
                "get_directions",
                "get_travel_time",
                "convert_currency",
            ],
            specializations=["comprehensive_planning", "multi_day_itinerary"],
            complexity_levels=[TripComplexity.MODERATE, TripComplexity.COMPLEX],
            context_requirements=["destination", "duration", "budget"],
            output_formats=["structured_itinerary", "daily_schedule"],
        )

        # Information Gatherer capabilities
        self.register_capability(
            AgentRole.INFORMATION_GATHERER,
            functions=[
                "find_places",
                "find_nearby_places",
                "get_place_details",
                "geocode_location",
            ],
            specializations=[
                "destination_research",
                "activity_discovery",
                "venue_analysis",
            ],
            complexity_levels=[
                TripComplexity.SIMPLE,
                TripComplexity.MODERATE,
                TripComplexity.COMPLEX,
            ],
            context_requirements=["destination"],
            output_formats=["structured_data", "research_report"],
        )

        # Itinerary Planner capabilities
        self.register_capability(
            AgentRole.ITINERARY_PLANNER,
            functions=[
                "find_places",
                "get_directions",
                "get_travel_time",
                "get_place_details",
            ],
            specializations=[
                "daily_scheduling",
                "route_optimization",
                "timing_coordination",
            ],
            complexity_levels=[TripComplexity.MODERATE, TripComplexity.COMPLEX],
            context_requirements=["destination", "activities", "transportation"],
            output_formats=["daily_itinerary", "timeline_schedule"],
        )

        # Optimization Agent capabilities
        self.register_capability(
            AgentRole.OPTIMIZATION_AGENT,
            functions=["get_travel_time", "get_directions", "convert_currency"],
            specializations=[
                "cost_optimization",
                "time_optimization",
                "route_optimization",
            ],
            complexity_levels=[TripComplexity.COMPLEX, TripComplexity.ENTERPRISE],
            context_requirements=["itinerary_data", "constraints"],
            output_formats=["optimized_plan", "efficiency_report"],
        )

        # Destination Expert capabilities
        self.register_capability(
            AgentRole.DESTINATION_EXPERT,
            functions=["find_places", "find_nearby_places", "get_place_details"],
            specializations=["local_knowledge", "cultural_insights", "seasonal_advice"],
            complexity_levels=[
                TripComplexity.SIMPLE,
                TripComplexity.MODERATE,
                TripComplexity.COMPLEX,
            ],
            context_requirements=["destination"],
            output_formats=["expert_insights", "recommendation_report"],
        )

        # Budget Advisor capabilities
        self.register_capability(
            AgentRole.BUDGET_ADVISOR,
            functions=["convert_currency", "find_places"],
            specializations=["cost_analysis", "budget_planning", "expense_tracking"],
            complexity_levels=[
                TripComplexity.MODERATE,
                TripComplexity.COMPLEX,
                TripComplexity.ENTERPRISE,
            ],
            context_requirements=["budget", "destination", "activities"],
            output_formats=["budget_breakdown", "cost_analysis"],
        )

        # Route Planner capabilities
        self.register_capability(
            AgentRole.ROUTE_PLANNER,
            functions=["get_directions", "get_travel_time", "geocode_location"],
            specializations=[
                "transportation_optimization",
                "multi_modal_routing",
                "logistics_planning",
            ],
            complexity_levels=[
                TripComplexity.MODERATE,
                TripComplexity.COMPLEX,
                TripComplexity.ENTERPRISE,
            ],
            context_requirements=["locations", "transportation_modes"],
            output_formats=["route_plan", "logistics_schedule"],
        )

    def register_capability(
        self,
        agent_role: AgentRole,
        functions: List[str],
        specializations: List[str],
        complexity_levels: List[TripComplexity],
        context_requirements: List[str],
        output_formats: List[str],
    ) -> None:
        """Register capabilities for an agent role."""

        self._capabilities[agent_role] = {
            "functions": functions,
            "specializations": specializations,
            "complexity_levels": complexity_levels,
            "context_requirements": context_requirements,
            "output_formats": output_formats,
        }

        # Update specialization mappings
        for spec in specializations:
            if spec not in self._specializations:
                self._specializations[spec] = []
            self._specializations[spec].append(agent_role)

        # Update function mappings
        for func in functions:
            if func not in self._function_mappings:
                self._function_mappings[func] = []
            self._function_mappings[func].append(agent_role)

    def get_agents_by_specialization(self, specialization: str) -> List[AgentRole]:
        """Get agent roles that support a specific specialization."""
        return self._specializations.get(specialization, [])

    def get_agents_by_function(self, function_name: str) -> List[AgentRole]:
        """Get agent roles that support a specific function."""
        return self._function_mappings.get(function_name, [])

    def get_agents_by_complexity(self, complexity: TripComplexity) -> List[AgentRole]:
        """Get agent roles suitable for a specific complexity level."""
        suitable_agents = []
        for role, capabilities in self._capabilities.items():
            if complexity in capabilities.get("complexity_levels", []):
                suitable_agents.append(role)
        return suitable_agents

    def get_agent_capabilities(self, agent_role: AgentRole) -> Optional[Dict[str, Any]]:
        """Get capabilities for a specific agent role."""
        return self._capabilities.get(agent_role)


class AgentConfigurationBuilder:
    """Builder for creating agent configurations based on requirements."""

    def __init__(self, capabilities_registry: AgentCapabilityRegistry) -> None:
        self.registry = capabilities_registry
        self._config = AgentConfiguration(
            agent_type=AgentRole.TRIP_PLANNER,
            priority=AgentPriority.MEDIUM,
            resource_tier=ResourceTier.STANDARD,
        )

    def for_role(self, role: AgentRole) -> "AgentConfigurationBuilder":
        """Set the agent role."""
        self._config.agent_type = role

        # Set defaults from registry
        capabilities = self.registry.get_agent_capabilities(role)
        if capabilities:
            self._config.supported_functions = capabilities.get("functions", [])
            self._config.requires_context = capabilities.get("context_requirements", [])

        return self

    def with_priority(self, priority: AgentPriority) -> "AgentConfigurationBuilder":
        """Set agent priority."""
        self._config.priority = priority
        return self

    def with_resource_tier(self, tier: ResourceTier) -> "AgentConfigurationBuilder":
        """Set resource allocation tier."""
        self._config.resource_tier = tier

        # Adjust timeouts and iterations based on tier
        if tier == ResourceTier.PREMIUM:
            self._config.timeout_seconds = 600
            self._config.max_iterations = 20
        elif tier == ResourceTier.ECONOMY:
            self._config.timeout_seconds = 120
            self._config.max_iterations = 5

        return self

    def with_functions(self, functions: List[str]) -> "AgentConfigurationBuilder":
        """Add supported functions."""
        self._config.supported_functions.extend(functions)
        return self

    def with_context_requirements(
        self, requirements: List[str]
    ) -> "AgentConfigurationBuilder":
        """Add context requirements."""
        self._config.requires_context.extend(requirements)
        return self

    def with_custom_parameters(
        self, parameters: Dict[str, Any]
    ) -> "AgentConfigurationBuilder":
        """Add custom parameters."""
        self._config.custom_parameters.update(parameters)
        return self

    def with_dependencies(
        self, dependencies: List[AgentRole]
    ) -> "AgentConfigurationBuilder":
        """Add agent dependencies."""
        self._config.dependencies = dependencies
        return self

    def build(self) -> AgentConfiguration:
        """Build the agent configuration."""
        return self._config


class TripPlanningAgentFactory:
    """Advanced factory for creating trip planning agents with dynamic configuration."""

    def __init__(self) -> None:
        self.capabilities_registry = AgentCapabilityRegistry()
        self.tool_registry = get_tool_registry()
        self._agent_instances: Dict[str, BaseAgent] = {}
        self._agent_pools: Dict[AgentRole, List[BaseAgent]] = {}
        self._creation_stats = {
            "total_created": 0,
            "active_agents": 0,
            "agents_by_role": {},
            "agents_by_complexity": {},
        }

    def create_agent_for_trip(
        self,
        trip_requirements: TripRequirements,
        agent_role: AgentRole,
        session_id: Optional[str] = None,
        custom_config: Optional[AgentConfiguration] = None,
    ) -> GeminiAgent:
        """Create a specialized agent based on trip requirements."""

        try:
            # Determine configuration
            if custom_config:
                config = custom_config
            else:
                config = self._build_config_for_trip(agent_role, trip_requirements)

            # Create agent based on role and configuration
            agent = self._create_specialized_agent(
                agent_role, trip_requirements, config, session_id
            )

            # Register and track agent
            self._register_agent_instance(agent, trip_requirements.complexity)

            logger.info(
                "Agent created for trip",
                agent_id=agent.agent_id,
                role=agent_role.value,
                complexity=trip_requirements.complexity.value,
                destination=trip_requirements.destination,
            )

            return agent

        except Exception as e:
            logger.error(
                "Failed to create agent for trip",
                role=agent_role.value,
                destination=trip_requirements.destination,
                error=str(e),
                exc_info=True,
            )
            raise AgentError(f"Agent creation failed: {e}") from e

    def create_agent_team_for_trip(
        self,
        trip_requirements: TripRequirements,
        session_id: Optional[str] = None,
    ) -> Dict[AgentRole, GeminiAgent]:
        """Create a complete team of agents for a trip based on complexity."""

        # Determine required agents based on trip complexity
        required_roles = self._get_required_agents(trip_requirements)

        agent_team = {}

        for role in required_roles:
            try:
                agent = self.create_agent_for_trip(trip_requirements, role, session_id)
                agent_team[role] = agent

            except Exception as e:
                logger.warning(
                    f"Failed to create agent for role {role.value}: {e}",
                    role=role.value,
                    trip_complexity=trip_requirements.complexity.value,
                )

        logger.info(
            "Agent team created for trip",
            team_size=len(agent_team),
            roles=[role.value for role in agent_team],
            complexity=trip_requirements.complexity.value,
            destination=trip_requirements.destination,
        )

        return agent_team

    def _build_config_for_trip(
        self, agent_role: AgentRole, requirements: TripRequirements
    ) -> AgentConfiguration:
        """Build agent configuration based on trip requirements."""

        builder = AgentConfigurationBuilder(self.capabilities_registry)
        config = (
            builder.for_role(agent_role)
            .with_priority(self._determine_priority(agent_role, requirements))
            .with_resource_tier(self._determine_resource_tier(requirements))
            .build()
        )

        # Add trip-specific customizations
        config.custom_parameters.update(
            {
                "destination": requirements.destination,
                "duration_days": requirements.duration_days,
                "traveler_count": requirements.traveler_count,
                "trip_type": requirements.trip_type,
                "budget_range": requirements.budget_range,
                "special_requirements": requirements.special_requirements,
                "language_preferences": requirements.language_preferences,
            }
        )

        # Role-specific customizations
        if agent_role == AgentRole.INFORMATION_GATHERER:
            config.custom_parameters["research_depth"] = (
                "comprehensive"
                if requirements.complexity == TripComplexity.COMPLEX
                else "standard"
            )
            config.max_iterations = (
                15 if requirements.complexity == TripComplexity.COMPLEX else 10
            )

        elif agent_role == AgentRole.ITINERARY_PLANNER:
            config.custom_parameters["optimization_level"] = (
                requirements.complexity.value
            )
            config.custom_parameters["preferred_activities"] = (
                requirements.preferred_activities
            )
            config.timeout_seconds = 450 if requirements.duration_days > 7 else 300

        elif agent_role == AgentRole.BUDGET_ADVISOR:
            config.custom_parameters["budget_analysis_detail"] = (
                "detailed" if requirements.budget_range else "estimated"
            )
            config.supported_functions.append("find_places")  # For cost research

        elif agent_role == AgentRole.ROUTE_PLANNER:
            config.custom_parameters["transportation_modes"] = (
                requirements.transportation_modes
            )
            config.custom_parameters["multi_destination"] = requirements.complexity in [
                TripComplexity.COMPLEX,
                TripComplexity.ENTERPRISE,
            ]

        return config

    def _create_specialized_agent(
        self,
        role: AgentRole,
        requirements: TripRequirements,
        config: AgentConfiguration,
        session_id: Optional[str] = None,
    ) -> GeminiAgent:
        """Create a specialized agent instance."""

        # Create agent capabilities
        capabilities = AgentCapabilities(
            role=role,
            prompt_type=self._get_prompt_type_for_role(role),
            supported_functions=config.supported_functions,
            can_delegate=role in [AgentRole.TRIP_PLANNER, AgentRole.ITINERARY_PLANNER],
            max_iterations=config.max_iterations,
            timeout_seconds=config.timeout_seconds,
            requires_context=config.requires_context,
            output_format=config.output_format,
        )

        # Generate unique agent ID
        agent_id = f"{role.value}_{requirements.destination.lower().replace(' ', '_')}_{int(datetime.now(timezone.utc).timestamp())}"

        # Create agent instance
        agent = GeminiAgent(agent_id, capabilities, session_id)

        # Apply custom configuration
        agent.update_context(**config.custom_parameters)

        return agent

    def _get_required_agents(self, requirements: TripRequirements) -> List[AgentRole]:
        """Determine required agents based on trip complexity and requirements."""

        base_agents = [AgentRole.TRIP_PLANNER]

        if requirements.complexity == TripComplexity.SIMPLE:
            # Simple trips need basic planning
            if requirements.budget_range:
                base_agents.append(AgentRole.BUDGET_ADVISOR)

        elif requirements.complexity == TripComplexity.MODERATE:
            # Moderate trips benefit from research and optimization
            base_agents.extend(
                [
                    AgentRole.INFORMATION_GATHERER,
                    AgentRole.DESTINATION_EXPERT,
                ]
            )

            if requirements.budget_range:
                base_agents.append(AgentRole.BUDGET_ADVISOR)

            if len(requirements.transportation_modes) > 1:
                base_agents.append(AgentRole.ROUTE_PLANNER)

        elif requirements.complexity in [
            TripComplexity.COMPLEX,
            TripComplexity.ENTERPRISE,
        ]:
            # Complex trips need comprehensive planning
            base_agents.extend(
                [
                    AgentRole.INFORMATION_GATHERER,
                    AgentRole.ITINERARY_PLANNER,
                    AgentRole.DESTINATION_EXPERT,
                    AgentRole.BUDGET_ADVISOR,
                    AgentRole.ROUTE_PLANNER,
                    AgentRole.OPTIMIZATION_AGENT,
                ]
            )

        # Add specialized agents based on requirements
        if requirements.accessibility_needs:
            base_agents.append(AgentRole.SAFETY_ADVISOR)

        if requirements.dietary_restrictions:
            # Could add a specialized food/dining agent
            pass

        return list(set(base_agents))  # Remove duplicates

    def _determine_priority(
        self, role: AgentRole, requirements: TripRequirements
    ) -> AgentPriority:
        """Determine agent priority based on role and trip requirements."""

        # Critical agents for all trips
        if role in [AgentRole.TRIP_PLANNER, AgentRole.DESTINATION_EXPERT]:
            return AgentPriority.CRITICAL

        # High priority for complex trips
        if requirements.complexity == TripComplexity.COMPLEX and role in [
            AgentRole.ITINERARY_PLANNER,
            AgentRole.BUDGET_ADVISOR,
        ]:
            return AgentPriority.HIGH

        # High priority for enterprise trips
        if requirements.complexity == TripComplexity.ENTERPRISE:
            return AgentPriority.HIGH

        return AgentPriority.MEDIUM

    def _determine_resource_tier(self, requirements: TripRequirements) -> ResourceTier:
        """Determine resource allocation tier based on trip requirements."""

        if requirements.complexity == TripComplexity.ENTERPRISE:
            return ResourceTier.PREMIUM
        elif (
            requirements.complexity == TripComplexity.COMPLEX
            or requirements.duration_days > 14
            or requirements.traveler_count > 6
        ):
            return ResourceTier.STANDARD
        else:
            return ResourceTier.ECONOMY

    def _get_prompt_type_for_role(self, role: AgentRole) -> PromptType:
        """Get appropriate prompt type for agent role."""

        role_to_prompt = {
            AgentRole.TRIP_PLANNER: PromptType.TRIP_PLANNER,
            AgentRole.INFORMATION_GATHERER: PromptType.DESTINATION_EXPERT,
            AgentRole.ITINERARY_PLANNER: PromptType.TRIP_PLANNER,
            AgentRole.OPTIMIZATION_AGENT: PromptType.ITINERARY_OPTIMIZER,
            AgentRole.DESTINATION_EXPERT: PromptType.DESTINATION_EXPERT,
            AgentRole.BUDGET_ADVISOR: PromptType.BUDGET_ADVISOR,
            AgentRole.ROUTE_PLANNER: PromptType.TRANSPORT_PLANNER,
            AgentRole.SAFETY_ADVISOR: PromptType.DESTINATION_EXPERT,
        }

        return role_to_prompt.get(role, PromptType.TRIP_PLANNER)

    def _register_agent_instance(
        self, agent: BaseAgent, complexity: TripComplexity
    ) -> None:
        """Register agent instance for tracking and management."""

        self._agent_instances[agent.agent_id] = agent

        # Add to role-based pool
        role = agent.capabilities.role
        if role not in self._agent_pools:
            self._agent_pools[role] = []
        self._agent_pools[role].append(agent)

        # Update statistics
        self._creation_stats["total_created"] += 1
        self._creation_stats["active_agents"] += 1

        role_count = self._creation_stats["agents_by_role"].get(role.value, 0)
        self._creation_stats["agents_by_role"][role.value] = role_count + 1

        complexity_count = self._creation_stats["agents_by_complexity"].get(
            complexity.value, 0
        )
        self._creation_stats["agents_by_complexity"][complexity.value] = (
            complexity_count + 1
        )

    def get_agent_pool(self, role: AgentRole) -> List[BaseAgent]:
        """Get agent pool for a specific role."""
        return self._agent_pools.get(role, [])

    def get_factory_stats(self) -> Dict[str, Any]:
        """Get factory statistics and metrics."""
        return {
            **self._creation_stats,
            "available_roles": [role.value for role in AgentRole],
            "supported_complexity_levels": [level.value for level in TripComplexity],
            "registered_capabilities": len(self.capabilities_registry._capabilities),
        }

    async def cleanup_agent(self, agent_id: str) -> None:
        """Cleanup a specific agent."""
        if agent_id in self._agent_instances:
            agent = self._agent_instances[agent_id]

            try:
                await agent.cleanup()

                # Remove from pools
                role = agent.capabilities.role
                if role in self._agent_pools and agent in self._agent_pools[role]:
                    self._agent_pools[role].remove(agent)

                # Remove from instances
                del self._agent_instances[agent_id]

                self._creation_stats["active_agents"] -= 1

                logger.debug("Agent cleaned up from factory", agent_id=agent_id)

            except Exception:
                logger.exception(f"Error cleaning up agent {agent_id}")

    async def cleanup_all_agents(self) -> None:
        """Cleanup all agents created by this factory."""
        for agent_id in list(self._agent_instances.keys()):
            await self.cleanup_agent(agent_id)

        self._agent_instances.clear()
        self._agent_pools.clear()

        logger.info("All factory agents cleaned up")


# Global factory instance
_agent_factory: Optional[TripPlanningAgentFactory] = None


def get_agent_factory() -> TripPlanningAgentFactory:
    """Get global agent factory instance."""
    global _agent_factory

    if _agent_factory is None:
        _agent_factory = TripPlanningAgentFactory()

    return _agent_factory


# Convenience functions for common agent creation patterns


def create_simple_trip_agents(
    destination: str,
    duration_days: int,
    budget: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Dict[AgentRole, GeminiAgent]:
    """Create agents for a simple trip."""

    requirements = TripRequirements(
        destination=destination,
        duration_days=duration_days,
        budget_range=budget,
        complexity=TripComplexity.SIMPLE,
    )

    factory = get_agent_factory()
    return factory.create_agent_team_for_trip(requirements, session_id)


def create_complex_trip_agents(
    destination: str,
    duration_days: int,
    traveler_count: int,
    budget: Optional[str] = None,
    special_requirements: Optional[List[str]] = None,
    session_id: Optional[str] = None,
) -> Dict[AgentRole, GeminiAgent]:
    """Create agents for a complex trip."""

    requirements = TripRequirements(
        destination=destination,
        duration_days=duration_days,
        traveler_count=traveler_count,
        budget_range=budget,
        complexity=TripComplexity.COMPLEX,
        special_requirements=special_requirements or [],
    )

    factory = get_agent_factory()
    return factory.create_agent_team_for_trip(requirements, session_id)


async def cleanup_agent_factory() -> None:
    """Cleanup global agent factory."""
    global _agent_factory

    if _agent_factory:
        await _agent_factory.cleanup_all_agents()
        _agent_factory = None
