"""Agent Communication Patterns for AI-Powered Trip Planner Backend - Google ADK Multi-Agent System.

This module provides inter-agent messaging and data exchange protocols, shared session
state management between agents, agent coordination patterns (handoff, delegation,
collaboration), context persistence and conversation continuity, and error propagation
and recovery mechanisms.
"""

import asyncio
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from uuid import uuid4

from src.ai_services.exceptions import (
    AgentCommunicationError,
    AgentError,
)
from src.ai_services.session_manager import get_session_manager
from src.core.logging import get_logger

logger = get_logger(__name__)


class MessageType(str, Enum):
    """Types of inter-agent messages."""

    REQUEST = "request"
    RESPONSE = "response"
    HANDOFF = "handoff"
    DELEGATION = "delegation"
    COLLABORATION = "collaboration"
    STATUS_UPDATE = "status_update"
    ERROR_REPORT = "error_report"
    CONTEXT_SHARE = "context_share"
    BROADCAST = "broadcast"


class MessagePriority(str, Enum):
    """Message priority levels."""

    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


class CommunicationPattern(str, Enum):
    """Agent communication patterns."""

    DIRECT = "direct"  # One-to-one communication
    BROADCAST = "broadcast"  # One-to-many communication
    CHAIN = "chain"  # Sequential handoff pattern
    DELEGATION = "delegation"  # Parent-child delegation
    COLLABORATION = "collaboration"  # Peer-to-peer collaboration


@dataclass
class AgentMessage:
    """Enhanced message structure for inter-agent communication."""

    message_id: str = field(default_factory=lambda: str(uuid4()))
    sender_id: str = ""
    receiver_id: Optional[str] = None
    message_type: MessageType = MessageType.REQUEST
    priority: MessagePriority = MessagePriority.NORMAL
    content: str = ""
    context_data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    reply_to: Optional[str] = None
    conversation_thread: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary format."""
        return {
            "message_id": self.message_id,
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "message_type": self.message_type.value,
            "priority": self.priority.value,
            "content": self.content,
            "context_data": self.context_data,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "reply_to": self.reply_to,
            "conversation_thread": self.conversation_thread,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentMessage":
        """Create message from dictionary."""
        # Parse datetime fields
        if "timestamp" in data and isinstance(data["timestamp"], str):
            data["timestamp"] = datetime.fromisoformat(
                data["timestamp"].replace("Z", "+00:00")
            )

        if (
            "expires_at" in data
            and data["expires_at"]
            and isinstance(data["expires_at"], str)
        ):
            data["expires_at"] = datetime.fromisoformat(
                data["expires_at"].replace("Z", "+00:00")
            )

        # Convert enum fields
        if "message_type" in data:
            data["message_type"] = MessageType(data["message_type"])
        if "priority" in data:
            data["priority"] = MessagePriority(data["priority"])

        return cls(**data)

    def is_expired(self) -> bool:
        """Check if message has expired."""
        if not self.expires_at:
            return False
        return datetime.now(timezone.utc) > self.expires_at


class SharedContext:
    """Shared context manager for multi-agent collaboration."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.session_manager = get_session_manager()
        self._context_data: Dict[str, Any] = {}
        self._context_lock = asyncio.Lock()
        self._subscribers: Dict[str, List[Callable]] = {}  # Event subscribers
        self._change_log: List[Dict[str, Any]] = []

    async def initialize(self) -> None:
        """Initialize shared context from session."""
        try:
            session = await self.session_manager.get_session(self.session_id)
            self._context_data = session.context.session_metadata.get(
                "shared_context", {}
            )

            logger.debug(
                "Shared context initialized",
                session_id=self.session_id,
                context_keys=list(self._context_data.keys()),
            )

        except Exception as e:
            logger.warning(f"Failed to initialize shared context: {e}")
            self._context_data = {}

    async def update_context(self, agent_id: str, updates: Dict[str, Any]) -> None:
        """Update shared context with thread safety."""
        async with self._context_lock:
            try:
                # Record change
                change_record = {
                    "agent_id": agent_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "updates": updates,
                    "previous_keys": list(self._context_data.keys()),
                }

                # Apply updates
                self._context_data.update(updates)

                # Log change
                self._change_log.append(change_record)

                # Keep change log manageable
                if len(self._change_log) > 100:
                    self._change_log = self._change_log[-50:]

                # Persist to session
                await self._persist_context()

                # Notify subscribers
                await self._notify_subscribers(
                    "context_updated",
                    {
                        "agent_id": agent_id,
                        "updated_keys": list(updates.keys()),
                        "context_snapshot": self._context_data.copy(),
                    },
                )

                logger.debug(
                    "Shared context updated",
                    agent_id=agent_id,
                    updated_keys=list(updates.keys()),
                    total_keys=len(self._context_data),
                )

            except Exception as e:
                logger.exception("Failed to update shared context")
                raise AgentCommunicationError(f"Context update failed: {e}") from e

    async def get_context(self, keys: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get shared context data."""
        async with self._context_lock:
            if keys:
                return {
                    k: self._context_data.get(k)
                    for k in keys
                    if k in self._context_data
                }
            return self._context_data.copy()

    async def subscribe_to_changes(self, agent_id: str, callback: Callable) -> None:
        """Subscribe to context changes."""
        if agent_id not in self._subscribers:
            self._subscribers[agent_id] = []
        self._subscribers[agent_id].append(callback)

    async def unsubscribe_from_changes(self, agent_id: str) -> None:
        """Unsubscribe from context changes."""
        if agent_id in self._subscribers:
            del self._subscribers[agent_id]

    async def _notify_subscribers(
        self, event_type: str, event_data: Dict[str, Any]
    ) -> None:
        """Notify subscribers of context changes."""
        for agent_id, callbacks in self._subscribers.items():
            for callback in callbacks:
                try:
                    await callback(event_type, event_data)
                except Exception as e:
                    logger.warning(
                        f"Subscriber notification failed for {agent_id}: {e}"
                    )

    async def _persist_context(self) -> None:
        """Persist shared context to session."""
        try:
            session = await self.session_manager.get_session(self.session_id)
            session.update_context(shared_context=self._context_data)
            await self.session_manager.update_session(session)
        except Exception as e:
            logger.warning(f"Failed to persist shared context: {e}")

    def get_change_log(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent context changes."""
        return self._change_log[-limit:]


class MessageBroker:
    """Message broker for inter-agent communication."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self._message_queues: Dict[str, List[AgentMessage]] = {}
        self._message_history: List[AgentMessage] = []
        self._subscribers: Dict[str, List[Callable]] = {}
        self._delivery_confirmations: Dict[str, bool] = {}
        self._retry_queues: Dict[str, List[AgentMessage]] = {}

    async def send_message(
        self,
        sender_id: str,
        receiver_id: Optional[str],
        content: str,
        message_type: MessageType = MessageType.REQUEST,
        priority: MessagePriority = MessagePriority.NORMAL,
        context_data: Optional[Dict[str, Any]] = None,
        expires_in_seconds: Optional[int] = None,
    ) -> str:
        """Send message between agents."""

        message = AgentMessage(
            sender_id=sender_id,
            receiver_id=receiver_id,
            content=content,
            message_type=message_type,
            priority=priority,
            context_data=context_data or {},
        )

        if expires_in_seconds:
            message.expires_at = datetime.now(timezone.utc) + timedelta(
                seconds=expires_in_seconds
            )

        try:
            if receiver_id:
                # Direct message
                await self._deliver_direct_message(message)
            else:
                # Broadcast message
                await self._deliver_broadcast_message(message)

            # Store in history
            self._message_history.append(message)

            # Keep history manageable
            if len(self._message_history) > 1000:
                self._message_history = self._message_history[-500:]

            logger.debug(
                "Message sent",
                message_id=message.message_id,
                sender=sender_id,
                receiver=receiver_id or "broadcast",
                type=message_type.value,
            )

            return message.message_id

        except Exception as e:
            logger.exception("Failed to send message")
            raise AgentCommunicationError(f"Message delivery failed: {e}") from e

    async def _deliver_direct_message(self, message: AgentMessage) -> None:
        """Deliver message to specific agent."""
        receiver_id = message.receiver_id

        if not receiver_id:
            raise AgentCommunicationError("Direct message requires receiver_id")

        if receiver_id not in self._message_queues:
            self._message_queues[receiver_id] = []

        # Insert based on priority
        if message.priority == MessagePriority.CRITICAL:
            self._message_queues[receiver_id].insert(0, message)
        elif message.priority == MessagePriority.HIGH:
            # Insert after critical messages
            critical_count = sum(
                1
                for m in self._message_queues[receiver_id]
                if m.priority == MessagePriority.CRITICAL
            )
            self._message_queues[receiver_id].insert(critical_count, message)
        else:
            self._message_queues[receiver_id].append(message)

        # Notify subscriber if any
        await self._notify_message_subscribers(receiver_id, message)

    async def _deliver_broadcast_message(self, message: AgentMessage) -> None:
        """Deliver message to all subscribed agents."""
        for agent_id in self._message_queues:
            # Create copy for each recipient
            agent_message = AgentMessage(
                sender_id=message.sender_id,
                receiver_id=agent_id,
                message_type=message.message_type,
                priority=message.priority,
                content=message.content,
                context_data=message.context_data.copy(),
                metadata=message.metadata.copy(),
            )

            await self._deliver_direct_message(agent_message)

    async def get_messages(self, agent_id: str, limit: int = 10) -> List[AgentMessage]:
        """Get messages for an agent."""
        if agent_id not in self._message_queues:
            return []

        messages = self._message_queues[agent_id][:limit]

        # Remove expired messages
        datetime.now(timezone.utc)
        valid_messages = [m for m in messages if not m.is_expired()]

        # Update queue with valid messages
        self._message_queues[agent_id] = [
            m
            for m in self._message_queues[agent_id]
            if not m.is_expired() and m not in messages
        ] + valid_messages[limit:]

        return valid_messages

    async def acknowledge_message(self, agent_id: str, message_id: str) -> None:
        """Acknowledge message receipt and processing."""
        self._delivery_confirmations[message_id] = True

        # Remove from agent's queue
        if agent_id in self._message_queues:
            self._message_queues[agent_id] = [
                m for m in self._message_queues[agent_id] if m.message_id != message_id
            ]

        logger.debug(f"Message acknowledged: {message_id} by {agent_id}")

    async def subscribe_to_messages(self, agent_id: str, callback: Callable) -> None:
        """Subscribe to incoming messages."""
        if agent_id not in self._subscribers:
            self._subscribers[agent_id] = []
        self._subscribers[agent_id].append(callback)

    async def _notify_message_subscribers(
        self, agent_id: str, message: AgentMessage
    ) -> None:
        """Notify subscribers of new messages."""
        if agent_id in self._subscribers:
            for callback in self._subscribers[agent_id]:
                try:
                    await callback(message)
                except Exception as e:
                    logger.warning(f"Message subscriber notification failed: {e}")

    def get_message_stats(self) -> Dict[str, Any]:
        """Get message broker statistics."""
        total_messages = len(self._message_history)
        confirmed_messages = len(self._delivery_confirmations)

        message_types = {}
        for message in self._message_history[-100:]:  # Recent messages
            msg_type = message.message_type.value
            message_types[msg_type] = message_types.get(msg_type, 0) + 1

        return {
            "total_messages": total_messages,
            "confirmed_deliveries": confirmed_messages,
            "pending_messages": sum(
                len(queue) for queue in self._message_queues.values()
            ),
            "active_queues": len(self._message_queues),
            "message_types": message_types,
            "delivery_rate": (confirmed_messages / max(total_messages, 1)) * 100,
        }


class CoordinationPattern(ABC):
    """Abstract base class for agent coordination patterns."""

    def __init__(self, pattern_id: str, session_id: str) -> None:
        self.pattern_id = pattern_id
        self.session_id = session_id
        self.message_broker = MessageBroker(session_id)
        self.shared_context = SharedContext(session_id)
        self._participating_agents: Set[str] = set()

    @abstractmethod
    async def execute_pattern(
        self, agents: List[str], initial_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute the coordination pattern."""

    async def add_agent(self, agent_id: str) -> None:
        """Add agent to coordination pattern."""
        self._participating_agents.add(agent_id)

        # Initialize message queue for agent
        if agent_id not in self.message_broker._message_queues:
            self.message_broker._message_queues[agent_id] = []

    async def remove_agent(self, agent_id: str) -> None:
        """Remove agent from coordination pattern."""
        self._participating_agents.discard(agent_id)


class HandoffPattern(CoordinationPattern):
    """Sequential handoff pattern for agent coordination."""

    async def execute_pattern(
        self, agents: List[str], initial_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute sequential handoff between agents."""

        await self.shared_context.initialize()
        await self.shared_context.update_context("system", initial_context)

        results = {
            "pattern_id": self.pattern_id,
            "pattern_type": "handoff",
            "agents": agents,
            "start_time": datetime.now(timezone.utc).isoformat(),
            "handoff_chain": [],
            "final_result": None,
            "success": True,
        }

        current_result = initial_context

        try:
            for i, agent_id in enumerate(agents):
                await self.add_agent(agent_id)

                # Send handoff message
                handoff_content = json.dumps(
                    {
                        "handoff_index": i,
                        "previous_agent": agents[i - 1] if i > 0 else None,
                        "next_agent": agents[i + 1] if i < len(agents) - 1 else None,
                        "current_context": current_result,
                        "handoff_instructions": "Process the context and prepare for handoff to next agent",
                    }
                )

                message_id = await self.message_broker.send_message(
                    sender_id="handoff_coordinator",
                    receiver_id=agent_id,
                    content=handoff_content,
                    message_type=MessageType.HANDOFF,
                    priority=MessagePriority.HIGH,
                    context_data=current_result,
                )

                # Record handoff
                handoff_record = {
                    "agent_id": agent_id,
                    "handoff_index": i,
                    "message_id": message_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "context_keys": (
                        list(current_result.keys())
                        if isinstance(current_result, dict)
                        else []
                    ),
                }

                results["handoff_chain"].append(handoff_record)

                # Update shared context
                await self.shared_context.update_context(
                    agent_id,
                    {
                        f"handoff_{i}_result": current_result,
                        "current_handoff_agent": agent_id,
                    },
                )

                logger.info(
                    "Handoff executed",
                    pattern_id=self.pattern_id,
                    agent_id=agent_id,
                    handoff_index=i,
                )

            results["final_result"] = current_result
            results["end_time"] = datetime.now(timezone.utc).isoformat()

        except Exception as e:
            results["success"] = False
            results["error"] = str(e)
            results["end_time"] = datetime.now(timezone.utc).isoformat()

            logger.exception("Handoff pattern failed")

        return results


class DelegationPattern(CoordinationPattern):
    """Delegation pattern for parent-child agent coordination."""

    async def execute_pattern(
        self, agents: List[str], initial_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute delegation pattern with parent and child agents."""

        if len(agents) < 2:
            raise AgentError("Delegation pattern requires at least 2 agents")

        parent_agent = agents[0]
        child_agents = agents[1:]

        await self.shared_context.initialize()
        await self.shared_context.update_context("system", initial_context)

        results = {
            "pattern_id": self.pattern_id,
            "pattern_type": "delegation",
            "parent_agent": parent_agent,
            "child_agents": child_agents,
            "start_time": datetime.now(timezone.utc).isoformat(),
            "delegations": [],
            "consolidated_result": None,
            "success": True,
        }

        try:
            # Create delegation tasks for child agents
            delegation_tasks = []

            for child_agent in child_agents:
                await self.add_agent(child_agent)

                # Send delegation message
                delegation_content = json.dumps(
                    {
                        "parent_agent": parent_agent,
                        "delegation_task": f"Process subset of context for {child_agent}",
                        "context": initial_context,
                        "expected_output": "processed_data",
                    }
                )

                message_id = await self.message_broker.send_message(
                    sender_id=parent_agent,
                    receiver_id=child_agent,
                    content=delegation_content,
                    message_type=MessageType.DELEGATION,
                    priority=MessagePriority.HIGH,
                    context_data=initial_context,
                )

                delegation_record = {
                    "child_agent": child_agent,
                    "message_id": message_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "task_assigned": True,
                }

                results["delegations"].append(delegation_record)
                delegation_tasks.append(child_agent)

            # Wait for all delegations to be acknowledged (simplified)
            await asyncio.sleep(1)  # Give time for processing

            # Collect results from shared context
            consolidated_data = await self.shared_context.get_context()
            results["consolidated_result"] = consolidated_data
            results["end_time"] = datetime.now(timezone.utc).isoformat()

            logger.info(
                "Delegation pattern completed",
                pattern_id=self.pattern_id,
                parent_agent=parent_agent,
                child_agents_count=len(child_agents),
            )

        except Exception as e:
            results["success"] = False
            results["error"] = str(e)
            results["end_time"] = datetime.now(timezone.utc).isoformat()

            logger.exception("Delegation pattern failed")

        return results


class CollaborationPattern(CoordinationPattern):
    """Collaboration pattern for peer-to-peer agent coordination."""

    async def execute_pattern(
        self, agents: List[str], initial_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute collaboration pattern between peer agents."""

        await self.shared_context.initialize()
        await self.shared_context.update_context("system", initial_context)

        results = {
            "pattern_id": self.pattern_id,
            "pattern_type": "collaboration",
            "participating_agents": agents,
            "start_time": datetime.now(timezone.utc).isoformat(),
            "collaboration_rounds": [],
            "final_consensus": None,
            "success": True,
        }

        try:
            # Set up collaboration
            for agent_id in agents:
                await self.add_agent(agent_id)

            # Create collaboration thread
            thread_id = str(uuid4())[:8]

            # Send collaboration invitation to all agents
            collaboration_content = json.dumps(
                {
                    "collaboration_type": "peer_to_peer",
                    "participating_agents": agents,
                    "shared_objective": "Collaborative trip planning",
                    "initial_context": initial_context,
                    "thread_id": thread_id,
                }
            )

            # Send to all agents
            message_ids = []
            for agent_id in agents:
                message_id = await self.message_broker.send_message(
                    sender_id="collaboration_coordinator",
                    receiver_id=agent_id,
                    content=collaboration_content,
                    message_type=MessageType.COLLABORATION,
                    priority=MessagePriority.HIGH,
                    context_data=initial_context,
                )
                message_ids.append(message_id)

            # Record collaboration round
            collaboration_round = {
                "round": 1,
                "message_ids": message_ids,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "agents_involved": agents,
                "thread_id": thread_id,
            }

            results["collaboration_rounds"].append(collaboration_round)

            # Simulate collaborative consensus building
            consensus_data = {
                "collaborative_result": "Peer agents contributed to comprehensive trip plan",
                "agent_contributions": {
                    agent_id: f"Contribution from {agent_id}" for agent_id in agents
                },
                "consensus_reached": True,
                "confidence_score": 8.5,
            }

            await self.shared_context.update_context(
                "collaboration_result", consensus_data
            )

            results["final_consensus"] = consensus_data
            results["end_time"] = datetime.now(timezone.utc).isoformat()

            logger.info(
                "Collaboration pattern completed",
                pattern_id=self.pattern_id,
                agents_count=len(agents),
                thread_id=thread_id,
            )

        except Exception as e:
            results["success"] = False
            results["error"] = str(e)
            results["end_time"] = datetime.now(timezone.utc).isoformat()

            logger.exception("Collaboration pattern failed")

        return results


class AgentCommunicationManager:
    """Main manager for agent communication and coordination."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.message_broker = MessageBroker(session_id)
        self.shared_context = SharedContext(session_id)
        self._coordination_patterns: Dict[str, CoordinationPattern] = {}
        self._error_handlers: Dict[str, Callable] = {}

    async def initialize(self) -> None:
        """Initialize communication manager."""
        await self.shared_context.initialize()

        logger.info(
            "Agent communication manager initialized",
            session_id=self.session_id,
        )

    async def create_handoff_coordination(
        self, agents: List[str], context: Dict[str, Any]
    ) -> str:
        """Create and execute handoff coordination pattern."""

        pattern_id = f"handoff_{len(self._coordination_patterns)}"
        pattern = HandoffPattern(pattern_id, self.session_id)

        self._coordination_patterns[pattern_id] = pattern

        result = await pattern.execute_pattern(agents, context)

        logger.info(
            "Handoff coordination created and executed",
            pattern_id=pattern_id,
            agents_count=len(agents),
            success=result.get("success", False),
        )

        return pattern_id

    async def create_delegation_coordination(
        self, parent_agent: str, child_agents: List[str], context: Dict[str, Any]
    ) -> str:
        """Create and execute delegation coordination pattern."""

        pattern_id = f"delegation_{len(self._coordination_patterns)}"
        pattern = DelegationPattern(pattern_id, self.session_id)

        self._coordination_patterns[pattern_id] = pattern

        agents = [parent_agent, *child_agents]
        result = await pattern.execute_pattern(agents, context)

        logger.info(
            "Delegation coordination created and executed",
            pattern_id=pattern_id,
            parent_agent=parent_agent,
            child_agents_count=len(child_agents),
            success=result.get("success", False),
        )

        return pattern_id

    async def create_collaboration_coordination(
        self, agents: List[str], context: Dict[str, Any]
    ) -> str:
        """Create and execute collaboration coordination pattern."""

        pattern_id = f"collaboration_{len(self._coordination_patterns)}"
        pattern = CollaborationPattern(pattern_id, self.session_id)

        self._coordination_patterns[pattern_id] = pattern

        result = await pattern.execute_pattern(agents, context)

        logger.info(
            "Collaboration coordination created and executed",
            pattern_id=pattern_id,
            agents_count=len(agents),
            success=result.get("success", False),
        )

        return pattern_id

    async def send_agent_message(
        self,
        sender_id: str,
        receiver_id: str,
        content: str,
        message_type: MessageType = MessageType.REQUEST,
        context_data: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Send message between agents."""
        return await self.message_broker.send_message(
            sender_id=sender_id,
            receiver_id=receiver_id,
            content=content,
            message_type=message_type,
            context_data=context_data,
        )

    async def broadcast_message(
        self,
        sender_id: str,
        content: str,
        context_data: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Broadcast message to all agents."""
        return await self.message_broker.send_message(
            sender_id=sender_id,
            receiver_id=None,
            content=content,
            message_type=MessageType.BROADCAST,
            context_data=context_data,
        )

    async def get_agent_messages(self, agent_id: str) -> List[AgentMessage]:
        """Get pending messages for an agent."""
        return await self.message_broker.get_messages(agent_id)

    async def update_shared_context(
        self, agent_id: str, updates: Dict[str, Any]
    ) -> None:
        """Update shared context from agent."""
        await self.shared_context.update_context(agent_id, updates)

    async def get_shared_context(
        self, keys: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Get shared context data."""
        return await self.shared_context.get_context(keys)

    def register_error_handler(self, error_type: str, handler: Callable) -> None:
        """Register error handler for communication failures."""
        self._error_handlers[error_type] = handler

    async def handle_communication_error(
        self, error_type: str, error_data: Dict[str, Any]
    ) -> None:
        """Handle communication errors with registered handlers."""

        if error_type in self._error_handlers:
            try:
                await self._error_handlers[error_type](error_data)
            except Exception:
                logger.exception(f"Error handler failed for {error_type}")

        # Default error handling
        logger.error(
            "Communication error occurred",
            error_type=error_type,
            error_data=error_data,
            session_id=self.session_id,
        )

    def get_communication_stats(self) -> Dict[str, Any]:
        """Get comprehensive communication statistics."""
        return {
            "session_id": self.session_id,
            "active_patterns": len(self._coordination_patterns),
            "participating_agents": len(self.message_broker._message_queues),
            "message_stats": self.message_broker.get_message_stats(),
            "context_stats": {
                "context_keys": len(self.shared_context._context_data),
                "change_log_entries": len(self.shared_context._change_log),
                "subscribers": len(self.shared_context._subscribers),
            },
            "error_handlers": list(self._error_handlers.keys()),
        }


# Global communication managers by session
_communication_managers: Dict[str, AgentCommunicationManager] = {}


def get_communication_manager(session_id: str) -> AgentCommunicationManager:
    """Get communication manager for a session."""
    global _communication_managers  # noqa: PLW0602

    if session_id not in _communication_managers:
        _communication_managers[session_id] = AgentCommunicationManager(session_id)

    return _communication_managers[session_id]


async def cleanup_communication_manager(session_id: str) -> None:
    """Cleanup communication manager for a session."""
    global _communication_managers  # noqa: PLW0602

    if session_id in _communication_managers:
        # No specific cleanup needed for current implementation
        del _communication_managers[session_id]
        logger.info(f"Communication manager cleaned up for session {session_id}")


async def cleanup_all_communication_managers() -> None:
    """Cleanup all communication managers."""
    global _communication_managers  # noqa: PLW0602

    for session_id in list(_communication_managers.keys()):
        await cleanup_communication_manager(session_id)

    _communication_managers.clear()
    logger.info("All communication managers cleaned up")
