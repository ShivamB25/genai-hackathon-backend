from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.database.firestore_client import DocumentNotFoundError, FirestoreClient


@pytest.fixture
def mock_firestore_client():
    """Fixture for a mocked FirestoreClient."""
    with patch(
        "google.cloud.firestore.AsyncClient", new_callable=AsyncMock
    ) as mock_client:
        client = FirestoreClient()
        client._client = mock_client
        yield client


@pytest.mark.asyncio
async def test_create_document(mock_firestore_client):
    """Test creating a document in Firestore."""
    mock_doc_ref = MagicMock()
    mock_doc_ref.id = "new_doc_id"

    # Mock the collection and add methods
    mock_collection_ref = MagicMock()
    mock_collection_ref.add = AsyncMock(return_value=(None, mock_doc_ref))
    mock_firestore_client._client.collection.return_value = mock_collection_ref

    doc_id = await mock_firestore_client.create_document(
        "my_collection", {"key": "value"}
    )

    assert doc_id == "new_doc_id"
    mock_firestore_client._client.collection.assert_called_with("my_collection")
    mock_collection_ref.add.assert_called_once()


@pytest.mark.asyncio
async def test_get_document_found(mock_firestore_client):
    """Test getting a document that exists."""
    mock_doc_snapshot = MagicMock()
    mock_doc_snapshot.exists = True
    mock_doc_snapshot.to_dict.return_value = {"key": "value"}
    mock_doc_snapshot.id = "existing_doc"

    mock_doc_ref = MagicMock()
    mock_doc_ref.get = AsyncMock(return_value=mock_doc_snapshot)
    mock_firestore_client._client.collection.return_value.document.return_value = (
        mock_doc_ref
    )

    doc = await mock_firestore_client.get_document("my_collection", "existing_doc")

    assert doc is not None
    assert doc["key"] == "value"
    assert doc["id"] == "existing_doc"


@pytest.mark.asyncio
async def test_get_document_not_found(mock_firestore_client):
    """Test getting a document that does not exist."""
    mock_doc_snapshot = MagicMock()
    mock_doc_snapshot.exists = False

    mock_doc_ref = MagicMock()
    mock_doc_ref.get = AsyncMock(return_value=mock_doc_snapshot)
    mock_firestore_client._client.collection.return_value.document.return_value = (
        mock_doc_ref
    )

    doc = await mock_firestore_client.get_document("my_collection", "nonexistent_doc")

    assert doc is None


@pytest.mark.asyncio
async def test_update_document_not_found(mock_firestore_client):
    """Test that updating a nonexistent document raises DocumentNotFoundError."""
    mock_doc_snapshot = MagicMock()
    mock_doc_snapshot.exists = False

    mock_doc_ref = MagicMock()
    mock_doc_ref.get = AsyncMock(return_value=mock_doc_snapshot)
    mock_firestore_client._client.collection.return_value.document.return_value = (
        mock_doc_ref
    )

    with pytest.raises(DocumentNotFoundError):
        await mock_firestore_client.update_document(
            "my_collection", "nonexistent", {"new": "data"}
        )
