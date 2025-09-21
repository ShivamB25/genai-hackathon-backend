"""Firebase Firestore client wrapper for AI-Powered Trip Planner Backend.

This module provides an async Firestore client with CRUD operations,
query helpers, and comprehensive error handling.
"""

import asyncio
from datetime import datetime
from typing import Any

from google.api_core import exceptions as gcp_exceptions
from google.cloud import firestore

from src.core.config import settings
from src.core.exceptions import (
    DatabaseConnectionException,
    DatabaseException,
    DatabaseTimeoutException,
)
from src.core.logging import get_logger
from src.database.models import BaseDocument, get_collection_name

logger = get_logger(__name__)

# Type variable for document models


class FirestoreClientError(DatabaseException):
    """Firestore-specific database error."""


class DocumentNotFoundError(FirestoreClientError):
    """Raised when a document is not found."""

    def __init__(self, collection: str, document_id: str) -> None:
        super().__init__(
            message=f"Document not found: {collection}/{document_id}",
            details={"collection": collection, "document_id": document_id},
        )


class FirestoreClient:
    """Async Firestore client wrapper with enhanced functionality."""

    def __init__(self) -> None:
        self._client = None
        self._timeout = settings.database_timeout
        self._max_retries = settings.database_retry_attempts

    async def _get_client(self) -> firestore.AsyncClient:
        """Get or create Firestore client."""
        if self._client is None:
            try:
                self._client = firestore.AsyncClient(
                    project=settings.firebase_project_id,
                    database=settings.firestore_database_id,
                )
                logger.info(
                    "Firestore client initialized", project=settings.firebase_project_id
                )
            except Exception as e:
                logger.exception("Failed to initialize Firestore client")
                error_msg = "Failed to initialize Firestore client"
                raise DatabaseConnectionException(error_msg) from e

        return self._client

    async def _execute_with_retry(
        self, operation: Any, *args: Any, **kwargs: Any
    ) -> Any:
        """Execute Firestore operation with retry logic."""
        last_exception = None

        for attempt in range(self._max_retries):
            try:
                return await asyncio.wait_for(
                    operation(*args, **kwargs), timeout=self._timeout
                )
            except TimeoutError as e:
                last_exception = e
                logger.warning(
                    "Firestore operation timeout",
                    attempt=attempt + 1,
                    max_attempts=self._max_retries,
                )
                if attempt == self._max_retries - 1:
                    error_msg = "Firestore operation timed out"
                    raise DatabaseTimeoutException(error_msg) from e
                await asyncio.sleep(2**attempt)
            except gcp_exceptions.ServiceUnavailable as e:
                last_exception = e
                logger.warning(
                    "Firestore service unavailable",
                    attempt=attempt + 1,
                    max_attempts=self._max_retries,
                )
                if attempt == self._max_retries - 1:
                    error_msg = "Firestore service unavailable"
                    raise DatabaseException(error_msg) from e
                await asyncio.sleep(2**attempt)
            except Exception as e:
                logger.exception("Firestore operation failed")
                error_msg = "Firestore operation failed"
                raise FirestoreClientError(error_msg) from e

        if last_exception:
            error_msg = "All retry attempts failed"
            raise FirestoreClientError(error_msg) from last_exception
        return None

    # CRUD Operations

    async def create_document(
        self,
        collection_name: str,
        document_data: dict[str, Any],
        document_id: str | None = None,
    ) -> str:
        """Create a new document in Firestore."""
        try:
            client = await self._get_client()

            # Add timestamp fields
            from datetime import UTC

            document_data["created_at"] = datetime.now(UTC)
            document_data["updated_at"] = datetime.now(UTC)

            if document_id:
                doc_ref = client.collection(collection_name).document(document_id)
                await self._execute_with_retry(doc_ref.set, document_data)
                created_id = document_id
            else:
                collection_ref = client.collection(collection_name)
                # type: ignore[attr-defined] - Firestore add method returns tuple
                _, doc_ref = await self._execute_with_retry(
                    collection_ref.add, document_data
                )  # type: ignore[attr-defined]
                created_id = doc_ref.id  # type: ignore[attr-defined]

            logger.debug(
                "Document created", collection=collection_name, document_id=created_id
            )
            return created_id

        except Exception as e:
            logger.exception("Failed to create document", collection=collection_name)
            error_msg = "Failed to create document"
            raise FirestoreClientError(error_msg) from e

    async def get_document(
        self, collection_name: str, document_id: str
    ) -> dict[str, Any] | None:
        """Get a document from Firestore."""
        try:
            client = await self._get_client()

            doc_ref = client.collection(collection_name).document(document_id)
            doc = await self._execute_with_retry(doc_ref.get)

            # type: ignore[attr-defined] - Document snapshot has exists and to_dict methods
            if hasattr(doc, "exists") and doc.exists:  # type: ignore[attr-defined]
                data = doc.to_dict()  # type: ignore[attr-defined]
                data["id"] = doc.id  # type: ignore[attr-defined]
                logger.debug(
                    "Document retrieved",
                    collection=collection_name,
                    document_id=document_id,
                )
                return data
            logger.debug(
                "Document not found",
                collection=collection_name,
                document_id=document_id,
            )
            return None

        except Exception as e:
            logger.exception(
                "Failed to get document",
                collection=collection_name,
                document_id=document_id,
            )
            error_msg = "Failed to get document"
            raise FirestoreClientError(error_msg) from e

    async def update_document(
        self,
        collection_name: str,
        document_id: str,
        document_data: dict[str, Any],
        merge: bool = True,
    ) -> None:
        """Update a document in Firestore."""
        try:
            client = await self._get_client()

            doc_ref = client.collection(collection_name).document(document_id)

            # Check if document exists first
            doc = await self._execute_with_retry(doc_ref.get)
            # type: ignore[attr-defined] - Document snapshot has exists property
            if not (hasattr(doc, "exists") and doc.exists):  # type: ignore[attr-defined]
                raise DocumentNotFoundError(collection_name, document_id)

            # Add updated timestamp
            from datetime import UTC

            document_data["updated_at"] = datetime.now(UTC)

            await self._execute_with_retry(doc_ref.set, document_data, merge=merge)

            logger.debug(
                "Document updated", collection=collection_name, document_id=document_id
            )

        except DocumentNotFoundError:
            raise
        except Exception as e:
            logger.exception(
                "Failed to update document",
                collection=collection_name,
                document_id=document_id,
            )
            error_msg = "Failed to update document"
            raise FirestoreClientError(error_msg) from e

    async def delete_document(self, collection_name: str, document_id: str) -> bool:
        """Delete a document from Firestore."""
        try:
            client = await self._get_client()

            doc_ref = client.collection(collection_name).document(document_id)

            # Check if document exists first
            doc = await self._execute_with_retry(doc_ref.get)
            # type: ignore[attr-defined] - Document snapshot has exists property
            if not (hasattr(doc, "exists") and doc.exists):  # type: ignore[attr-defined]
                logger.debug(
                    "Document not found for deletion",
                    collection=collection_name,
                    document_id=document_id,
                )
                return False

            await self._execute_with_retry(doc_ref.delete)

            logger.debug(
                "Document deleted", collection=collection_name, document_id=document_id
            )
            return True

        except Exception as e:
            logger.exception(
                "Failed to delete document",
                collection=collection_name,
                document_id=document_id,
            )
            error_msg = "Failed to delete document"
            raise FirestoreClientError(error_msg) from e

    # Query Operations

    async def query_documents(
        self,
        collection_name: str,
        filters: list[tuple] | None = None,
        order_by: list[tuple] | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Query documents from a collection."""
        try:
            client = await self._get_client()

            collection_ref = client.collection(collection_name)
            query = collection_ref

            # Apply filters
            if filters:
                for field, operator, value in filters:
                    query = query.where(field, operator, value)

            # Apply ordering
            if order_by:
                for field, direction in order_by:
                    direction_value = (
                        firestore.Query.DESCENDING
                        if direction.lower() == "desc"
                        else firestore.Query.ASCENDING
                    )
                    query = query.order_by(field, direction=direction_value)

            # Apply limit
            if limit:
                query = query.limit(limit)

            # Execute query
            docs = query.stream()

            # Convert to list of dictionaries
            results = []
            # type: ignore[attr-defined] - Query stream returns document snapshots
            async for doc in docs:  # type: ignore[attr-defined]
                data = doc.to_dict()  # type: ignore[attr-defined]
                data["id"] = doc.id  # type: ignore[attr-defined]
                results.append(data)

            logger.debug(
                "Documents queried", collection=collection_name, count=len(results)
            )
            return results

        except Exception as e:
            logger.exception("Failed to query documents", collection=collection_name)
            error_msg = "Failed to query documents"
            raise FirestoreClientError(error_msg) from e

    # Model-based operations

    async def create_model(self, model: BaseDocument) -> str:
        """Create a document from a model instance."""
        collection_name = get_collection_name(type(model))
        data = model.to_dict()
        return await self.create_document(collection_name, data, model.id)

    async def get_model(
        self, model_class: type[BaseDocument], document_id: str
    ) -> BaseDocument | None:
        """Get a model instance by ID."""
        collection_name = get_collection_name(model_class)
        data = await self.get_document(collection_name, document_id)

        if data:
            return model_class.from_dict(data, document_id)
        return None

    async def update_model(self, model: BaseDocument) -> None:
        """Update a document from a model instance."""
        if not model.id:
            error_msg = "Model must have an ID to update"
            raise ValueError(error_msg)

        collection_name = get_collection_name(type(model))
        data = model.to_dict()
        await self.update_document(collection_name, model.id, data)

    async def delete_model(self, model: BaseDocument) -> bool:
        """Delete a document for a model instance."""
        if not model.id:
            error_msg = "Model must have an ID to delete"
            raise ValueError(error_msg)

        collection_name = get_collection_name(type(model))
        return await self.delete_document(collection_name, model.id)

    # Connection management

    async def close(self) -> None:
        """Close the Firestore client connection."""
        if self._client:
            self._client = None
            logger.info("Firestore client connection closed")

    async def health_check(self) -> bool:
        """Check Firestore connection health."""
        try:
            client = await self._get_client()

            # Try to read a simple document to test connection
            test_collection = client.collection("_health_check")
            test_doc = test_collection.document("test")

            # Try to get the document (it may not exist, but connection should work)
            await self._execute_with_retry(test_doc.get)

            logger.debug("Firestore health check passed")
            return True

        except Exception:
            logger.exception("Firestore health check failed")
            return False


# Global client instance
_firestore_client: FirestoreClient | None = None


def get_firestore_client() -> FirestoreClient:
    """Get the global Firestore client instance."""
    global _firestore_client
    if _firestore_client is None:
        _firestore_client = FirestoreClient()
    return _firestore_client


async def close_firestore_client() -> None:
    """Close the global Firestore client connection."""
    global _firestore_client
    if _firestore_client:
        await _firestore_client.close()
        _firestore_client = None
