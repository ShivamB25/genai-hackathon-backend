"""Session Management for AI-Powered Trip Planner Backend.

This module provides AI conversation session management with context persistence,
state tracking, session cleanup, timeout handling, and Firestore integration.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from src.ai_services.exceptions import (
    SessionError,
    SessionExpiredError,
    SessionNotFoundError,
    SessionStorageError,
)
from src.core.logging import get_logger
from src.database.firestore_client import get_firestore_client

logger = get_logger(__name__)


class SessionStatus(str, Enum):
    """AI session status types."""

    ACTIVE = "active"
    IDLE = "idle"
    EXPIRED = "expired"
    TERMINATED = "terminated"
    PAUSED = "paused"


class MessageRole(str, Enum):
    """Message roles in AI conversation."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    FUNCTION = "function"


class ConversationMessage(BaseModel):
    """Individual message in AI conversation."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict)
    token_count: Optional[int] = None
    function_call: Optional[Dict[str, Any]] = None
    function_response: Optional[Dict[str, Any]] = None

    model_config = {
        "json_encoders": {
            datetime: lambda v: v.isoformat(),
        }
    }


class SessionContext(BaseModel):
    """Session context information."""

    user_id: str
    user_profile: Dict[str, Any] = Field(default_factory=dict)
    trip_context: Dict[str, Any] = Field(default_factory=dict)
    preferences: Dict[str, Any] = Field(default_factory=dict)
    conversation_summary: str = ""
    active_tools: List[str] = Field(default_factory=list)
    current_task: Optional[str] = None
    session_metadata: Dict[str, Any] = Field(default_factory=dict)


class AISession(BaseModel):
    """AI conversation session model."""

    session_id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    status: SessionStatus = SessionStatus.ACTIVE
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc) + timedelta(hours=24)
    )

    # Conversation data
    messages: List[ConversationMessage] = Field(default_factory=list)
    context: SessionContext

    # Session statistics
    total_tokens: int = 0
    message_count: int = 0
    function_calls_count: int = 0

    # Configuration
    max_messages: int = Field(default=100)
    max_tokens: int = Field(default=32768)
    session_timeout_minutes: int = Field(default=60)

    model_config = {
        "json_encoders": {
            datetime: lambda v: v.isoformat(),
        }
    }

    def is_expired(self) -> bool:
        """Check if session is expired."""
        now = datetime.now(timezone.utc)
        return now > self.expires_at or self.status == SessionStatus.EXPIRED

    def is_idle(self) -> bool:
        """Check if session is idle."""
        if self.status != SessionStatus.ACTIVE:
            return False

        idle_threshold = datetime.now(timezone.utc) - timedelta(
            minutes=self.session_timeout_minutes
        )
        return self.last_activity < idle_threshold

    def add_message(self, message: ConversationMessage) -> None:
        """Add message to conversation."""
        self.messages.append(message)
        self.message_count += 1
        self.last_activity = datetime.now(timezone.utc)
        self.updated_at = self.last_activity

        if message.token_count:
            self.total_tokens += message.token_count

        if message.function_call:
            self.function_calls_count += 1

        # Trim messages if exceeding limit
        if len(self.messages) > self.max_messages:
            # Keep system messages and recent messages
            system_messages = [m for m in self.messages if m.role == MessageRole.SYSTEM]
            recent_messages = [
                m for m in self.messages if m.role != MessageRole.SYSTEM
            ][-self.max_messages + len(system_messages) :]
            self.messages = system_messages + recent_messages

    def get_conversation_history(
        self, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get conversation history formatted for AI model."""
        messages = self.messages

        if limit:
            messages = messages[-limit:]

        return [
            {
                "role": msg.role.value,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat(),
                **({"function_call": msg.function_call} if msg.function_call else {}),
                **(
                    {"function_response": msg.function_response}
                    if msg.function_response
                    else {}
                ),
            }
            for msg in messages
        ]

    def update_context(self, **context_updates) -> None:
        """Update session context."""
        for key, value in context_updates.items():
            if hasattr(self.context, key):
                setattr(self.context, key, value)
            else:
                self.context.session_metadata[key] = value

        self.updated_at = datetime.now(timezone.utc)

    def extend_expiration(self, hours: int = 24) -> None:
        """Extend session expiration time."""
        self.expires_at = datetime.now(timezone.utc) + timedelta(hours=hours)
        self.updated_at = datetime.now(timezone.utc)

    def get_stats(self) -> Dict[str, Any]:
        """Get session statistics."""
        duration = datetime.now(timezone.utc) - self.created_at

        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "status": self.status.value,
            "duration_minutes": int(duration.total_seconds() / 60),
            "message_count": self.message_count,
            "total_tokens": self.total_tokens,
            "function_calls": self.function_calls_count,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "expires_at": self.expires_at.isoformat(),
        }


class SessionManager:
    """AI session manager with Firestore persistence."""

    def __init__(self) -> None:
        """Initialize session manager."""
        self._firestore_client = get_firestore_client()
        self._collection_name = "ai_sessions"
        self._active_sessions: Dict[str, AISession] = {}
        self._cleanup_task: Optional[asyncio.Task] = None
        self._start_cleanup_task()

    def _start_cleanup_task(self) -> None:
        """Start background cleanup task."""
        if not self._cleanup_task or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_expired_sessions())

    async def _cleanup_expired_sessions(self) -> None:
        """Background task to cleanup expired sessions."""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                await self._perform_cleanup()
            except Exception as e:
                logger.error(
                    "Error in session cleanup task", error=str(e), exc_info=True
                )

    async def _perform_cleanup(self) -> None:
        """Perform session cleanup."""
        expired_sessions = []
        idle_sessions = []

        # Check active sessions in memory
        for session_id, session in self._active_sessions.copy().items():
            if session.is_expired():
                expired_sessions.append(session_id)
            elif session.is_idle():
                idle_sessions.append(session_id)

        # Clean up expired sessions
        for session_id in expired_sessions:
            try:
                await self._terminate_session(session_id, SessionStatus.EXPIRED)
                logger.info("Cleaned up expired session", session_id=session_id)
            except Exception:
                logger.exception(
                    "Error cleaning up expired session",
                    session_id=session_id,
                )

        # Mark idle sessions
        for session_id in idle_sessions:
            try:
                session = self._active_sessions[session_id]
                session.status = SessionStatus.IDLE
                await self._save_session(session)
                logger.debug("Marked session as idle", session_id=session_id)
            except Exception:
                logger.exception("Error marking session as idle", session_id=session_id)

        logger.debug(
            "Session cleanup completed",
            expired_cleaned=len(expired_sessions),
            idle_marked=len(idle_sessions),
            active_sessions=len(self._active_sessions),
        )

    async def create_session(
        self,
        user_id: str,
        initial_context: Optional[Dict[str, Any]] = None,
        **session_options,
    ) -> AISession:
        """Create new AI session.

        Args:
            user_id: User identifier
            initial_context: Initial session context
            **session_options: Additional session configuration

        Returns:
            AISession: Created session

        Raises:
            SessionStorageError: If session creation fails
        """
        try:
            # Create session context
            context_data = initial_context or {}
            context = SessionContext(
                user_id=user_id,
                user_profile=context_data.get("user_profile", {}),
                trip_context=context_data.get("trip_context", {}),
                preferences=context_data.get("preferences", {}),
                session_metadata=context_data.get("metadata", {}),
            )

            # Create session
            session = AISession(user_id=user_id, context=context, **session_options)

            # Add system message if provided
            system_prompt = context_data.get("system_prompt")
            if system_prompt:
                system_message = ConversationMessage(
                    role=MessageRole.SYSTEM, content=system_prompt
                )
                session.add_message(system_message)

            # Save to Firestore
            await self._save_session(session)

            # Cache in memory
            self._active_sessions[session.session_id] = session

            logger.info(
                "AI session created",
                session_id=session.session_id,
                user_id=user_id,
                expires_at=session.expires_at.isoformat(),
            )

            return session

        except Exception as e:
            error_msg = f"Failed to create session: {e}"
            logger.error(error_msg, user_id=user_id, exc_info=True)
            raise SessionStorageError(error_msg) from e

    async def get_session(self, session_id: str) -> AISession:
        """Get session by ID.

        Args:
            session_id: Session identifier

        Returns:
            AISession: Session instance

        Raises:
            SessionNotFoundError: If session not found
            SessionExpiredError: If session is expired
        """
        try:
            # Try memory cache first
            if session_id in self._active_sessions:
                session = self._active_sessions[session_id]

                # Check if expired
                if session.is_expired():
                    await self._terminate_session(session_id, SessionStatus.EXPIRED)
                    raise SessionExpiredError(
                        f"Session {session_id} has expired", session_id=session_id
                    )

                return session

            # Load from Firestore
            session_data = await self._firestore_client.get_document(
                self._collection_name, session_id
            )

            if not session_data:
                raise SessionNotFoundError(
                    f"Session {session_id} not found", session_id=session_id
                )

            # Parse session data
            session = self._parse_session_data(session_data)

            # Check if expired
            if session.is_expired():
                await self._terminate_session(session_id, SessionStatus.EXPIRED)
                raise SessionExpiredError(
                    f"Session {session_id} has expired", session_id=session_id
                )

            # Cache in memory
            self._active_sessions[session_id] = session

            logger.debug("Session loaded from storage", session_id=session_id)

            return session

        except (SessionNotFoundError, SessionExpiredError):
            raise
        except Exception as e:
            error_msg = f"Failed to get session: {e}"
            logger.error(error_msg, session_id=session_id, exc_info=True)
            raise SessionError(error_msg, session_id=session_id) from e

    async def update_session(self, session: AISession) -> None:
        """Update session.

        Args:
            session: Session to update

        Raises:
            SessionStorageError: If update fails
        """
        try:
            session.updated_at = datetime.now(timezone.utc)

            # Save to Firestore
            await self._save_session(session)

            # Update memory cache
            self._active_sessions[session.session_id] = session

            logger.debug("Session updated", session_id=session.session_id)

        except Exception as e:
            error_msg = f"Failed to update session: {e}"
            logger.error(error_msg, session_id=session.session_id, exc_info=True)
            raise SessionStorageError(error_msg, session_id=session.session_id) from e

    async def add_message(
        self, session_id: str, role: MessageRole, content: str, **message_metadata
    ) -> ConversationMessage:
        """Add message to session.

        Args:
            session_id: Session identifier
            role: Message role
            content: Message content
            **message_metadata: Additional message metadata

        Returns:
            ConversationMessage: Added message

        Raises:
            SessionNotFoundError: If session not found
        """
        session = await self.get_session(session_id)

        message = ConversationMessage(
            role=role,
            content=content,
            metadata=message_metadata,
            **{
                k: v
                for k, v in message_metadata.items()
                if k in ["token_count", "function_call", "function_response"]
            },
        )

        session.add_message(message)
        await self.update_session(session)

        logger.debug(
            "Message added to session",
            session_id=session_id,
            message_id=message.id,
            role=role.value,
        )

        return message

    async def get_user_sessions(
        self, user_id: str, status: Optional[SessionStatus] = None, limit: int = 10
    ) -> List[AISession]:
        """Get user's sessions.

        Args:
            user_id: User identifier
            status: Optional status filter
            limit: Maximum number of sessions

        Returns:
            List[AISession]: User's sessions
        """
        try:
            # Query filters
            filters = [("user_id", "==", user_id)]
            if status:
                filters.append(("status", "==", status.value))

            # Query Firestore
            session_docs = await self._firestore_client.query_documents(
                collection_name=self._collection_name,
                filters=filters,
                order_by=[("updated_at", "desc")],
                limit=limit,
            )

            sessions = []
            for doc_data in session_docs:
                try:
                    session = self._parse_session_data(doc_data)
                    sessions.append(session)
                except Exception as e:
                    logger.warning(
                        "Failed to parse session data",
                        session_id=doc_data.get("id"),
                        error=str(e),
                    )

            logger.debug(
                "User sessions retrieved",
                user_id=user_id,
                count=len(sessions),
                status_filter=status.value if status else None,
            )

            return sessions

        except Exception as e:
            error_msg = f"Failed to get user sessions: {e}"
            logger.error(error_msg, user_id=user_id, exc_info=True)
            raise SessionError(error_msg) from e

    async def terminate_session(self, session_id: str) -> None:
        """Terminate session.

        Args:
            session_id: Session identifier
        """
        await self._terminate_session(session_id, SessionStatus.TERMINATED)

    async def _terminate_session(self, session_id: str, status: SessionStatus) -> None:
        """Internal method to terminate session.

        Args:
            session_id: Session identifier
            status: Final status
        """
        try:
            # Update session status
            session_data = await self._firestore_client.get_document(
                self._collection_name, session_id
            )

            if session_data:
                session_data["status"] = status.value
                session_data["updated_at"] = datetime.now(timezone.utc).isoformat()

                await self._firestore_client.update_document(
                    self._collection_name, session_id, session_data
                )

            # Remove from memory cache
            if session_id in self._active_sessions:
                del self._active_sessions[session_id]

            logger.info(
                "Session terminated", session_id=session_id, status=status.value
            )

        except Exception as e:
            logger.error(
                "Failed to terminate session",
                session_id=session_id,
                error=str(e),
                exc_info=True,
            )

    async def _save_session(self, session: AISession) -> None:
        """Save session to Firestore.

        Args:
            session: Session to save
        """
        try:
            # Convert to dictionary
            session_data = session.model_dump(mode="json")

            # Handle datetime serialization
            session_data["created_at"] = session.created_at.isoformat()
            session_data["updated_at"] = session.updated_at.isoformat()
            session_data["last_activity"] = session.last_activity.isoformat()
            session_data["expires_at"] = session.expires_at.isoformat()

            # Save to Firestore
            await self._firestore_client.update_document(
                self._collection_name, session.session_id, session_data, merge=True
            )

        except Exception as e:
            error_msg = f"Failed to save session to storage: {e}"
            logger.error(error_msg, session_id=session.session_id, exc_info=True)
            raise SessionStorageError(error_msg, session_id=session.session_id) from e

    def _parse_session_data(self, data: Dict[str, Any]) -> AISession:
        """Parse session data from Firestore.

        Args:
            data: Raw session data

        Returns:
            AISession: Parsed session
        """
        try:
            # Parse datetime fields
            for field in ["created_at", "updated_at", "last_activity", "expires_at"]:
                if field in data and isinstance(data[field], str):
                    data[field] = datetime.fromisoformat(
                        data[field].replace("Z", "+00:00")
                    )

            # Parse messages
            if "messages" in data:
                messages = []
                for msg_data in data["messages"]:
                    if isinstance(msg_data["timestamp"], str):
                        msg_data["timestamp"] = datetime.fromisoformat(
                            msg_data["timestamp"].replace("Z", "+00:00")
                        )
                    messages.append(ConversationMessage(**msg_data))
                data["messages"] = messages

            # Parse context
            if "context" in data:
                data["context"] = SessionContext(**data["context"])

            return AISession(**data)

        except Exception as e:
            raise SessionError(f"Failed to parse session data: {e}") from e

    async def get_session_stats(self) -> Dict[str, Any]:
        """Get overall session statistics.

        Returns:
            Dict[str, Any]: Session statistics
        """
        try:
            # Get active sessions count
            active_count = len(self._active_sessions)

            # Query total sessions by status
            total_sessions = await self._firestore_client.query_documents(
                self._collection_name,
                limit=1000,  # For stats, we limit the query
            )

            status_counts = {}
            for session_data in total_sessions:
                status = session_data.get("status", "unknown")
                status_counts[status] = status_counts.get(status, 0) + 1

            return {
                "active_sessions_in_memory": active_count,
                "total_sessions_queried": len(total_sessions),
                "status_distribution": status_counts,
                "cleanup_task_running": self._cleanup_task
                and not self._cleanup_task.done(),
            }

        except Exception as e:
            logger.error("Failed to get session stats", error=str(e), exc_info=True)
            return {"error": str(e)}

    async def shutdown(self) -> None:
        """Shutdown session manager."""
        try:
            # Cancel cleanup task
            if self._cleanup_task and not self._cleanup_task.done():
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    pass

            # Save all active sessions
            for session in self._active_sessions.values():
                try:
                    await self._save_session(session)
                except Exception:
                    logger.exception(
                        "Failed to save session during shutdown",
                        session_id=session.session_id,
                    )

            self._active_sessions.clear()

            logger.info("Session manager shutdown completed")

        except Exception as e:
            logger.error(
                "Error during session manager shutdown", error=str(e), exc_info=True
            )


# Global session manager instance
_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """Get global session manager instance.

    Returns:
        SessionManager: Session manager instance
    """
    global _session_manager

    if _session_manager is None:
        _session_manager = SessionManager()

    return _session_manager


async def cleanup_session_manager() -> None:
    """Cleanup global session manager."""
    global _session_manager

    if _session_manager:
        await _session_manager.shutdown()
        _session_manager = None
