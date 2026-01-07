"""Unit tests for AlibabaCloudMySQL vector store."""

from unittest.mock import MagicMock, patch

import pytest


def test_import() -> None:
    """Test that the AlibabaCloudMySQL class can be imported."""
    from langchain_alibabacloud_mysql import AlibabaCloudMySQL

    assert AlibabaCloudMySQL is not None


def test_vector_to_string() -> None:
    """Test vector to string conversion."""
    from langchain_alibabacloud_mysql.vectorstores.alibabacloud_mysql import (
        AlibabaCloudMySQL,
    )

    # Test the static-like method by accessing it via the class
    # Since it's an instance method, we need to test the format
    vector = [1.0, 2.0, 3.0]
    expected = "[1.0,2.0,3.0]"

    # Create a mock instance to test the method
    with patch.object(AlibabaCloudMySQL, "__init__", lambda self, **kwargs: None):
        store = AlibabaCloudMySQL()
        store._distance_strategy = "cosine"
        result = store._vector_to_string(vector)
        assert result == expected


def test_distance_to_similarity_cosine() -> None:
    """Test distance to similarity conversion for cosine."""
    from langchain_alibabacloud_mysql.vectorstores.alibabacloud_mysql import (
        AlibabaCloudMySQL,
    )

    with patch.object(AlibabaCloudMySQL, "__init__", lambda self, **kwargs: None):
        store = AlibabaCloudMySQL()
        store._distance_strategy = "cosine"

        # For cosine: similarity = 1 - distance
        assert store._distance_to_similarity(0.0) == 1.0
        assert store._distance_to_similarity(0.5) == 0.5
        assert store._distance_to_similarity(1.0) == 0.0


def test_distance_to_similarity_euclidean() -> None:
    """Test distance to similarity conversion for euclidean."""
    from langchain_alibabacloud_mysql.vectorstores.alibabacloud_mysql import (
        AlibabaCloudMySQL,
    )

    with patch.object(AlibabaCloudMySQL, "__init__", lambda self, **kwargs: None):
        store = AlibabaCloudMySQL()
        store._distance_strategy = "euclidean"

        # For euclidean: similarity = 1 / (1 + distance)
        assert store._distance_to_similarity(0.0) == 1.0
        assert store._distance_to_similarity(1.0) == 0.5
        assert store._distance_to_similarity(3.0) == 0.25


def test_invalid_distance_strategy() -> None:
    """Test that invalid distance strategy raises an error."""
    from langchain_alibabacloud_mysql.vectorstores.alibabacloud_mysql import (
        AlibabaCloudMySQL,
    )

    mock_embedding = MagicMock()

    with patch("mysql.connector.pooling.MySQLConnectionPool"):
        with patch.object(AlibabaCloudMySQL, "_check_vector_support"):
            with pytest.raises(ValueError, match="Invalid distance_strategy"):
                AlibabaCloudMySQL(
                    host="localhost",
                    port=3306,
                    user="test",
                    password="test",
                    database="test",
                    embedding=mock_embedding,
                    distance_strategy="invalid",  # type: ignore
                )


def test_mmr_algorithm() -> None:
    """Test the MMR algorithm implementation."""
    pytest.importorskip("numpy")

    from langchain_alibabacloud_mysql.vectorstores.alibabacloud_mysql import (
        AlibabaCloudMySQL,
    )

    # Simple test vectors
    query_embedding = [1.0, 0.0, 0.0]
    embedding_list = [
        [1.0, 0.0, 0.0],  # Most similar to query
        [0.9, 0.1, 0.0],  # Second most similar
        [0.0, 1.0, 0.0],  # Different direction (diverse)
        [0.0, 0.0, 1.0],  # Different direction (diverse)
    ]

    # Test with lambda_mult=1.0 (no diversity, pure similarity)
    indices = AlibabaCloudMySQL._maximal_marginal_relevance(
        query_embedding=query_embedding,
        embedding_list=embedding_list,
        k=2,
        lambda_mult=1.0,
    )
    # Should return the two most similar
    assert 0 in indices
    assert 1 in indices

    # Test with lambda_mult=0.0 (max diversity)
    indices = AlibabaCloudMySQL._maximal_marginal_relevance(
        query_embedding=query_embedding,
        embedding_list=embedding_list,
        k=3,
        lambda_mult=0.0,
    )
    # First should still be most similar, but others should be diverse
    assert indices[0] == 0
    assert len(indices) == 3


def test_mmr_empty_list() -> None:
    """Test MMR with empty embedding list."""
    pytest.importorskip("numpy")

    from langchain_alibabacloud_mysql.vectorstores.alibabacloud_mysql import (
        AlibabaCloudMySQL,
    )

    indices = AlibabaCloudMySQL._maximal_marginal_relevance(
        query_embedding=[1.0, 0.0, 0.0],
        embedding_list=[],
        k=4,
        lambda_mult=0.5,
    )
    assert indices == []


# =============================================================================
# Metadata Filter Tests
# =============================================================================


def test_filter_operators_defined() -> None:
    """Test that all filter operators are defined."""
    from langchain_alibabacloud_mysql.vectorstores.alibabacloud_mysql import (
        FILTER_OPERATORS,
    )

    expected_operators = [
        "$eq",
        "$ne",
        "$gt",
        "$gte",
        "$lt",
        "$lte",
        "$in",
        "$nin",
        "$like",
    ]
    for op in expected_operators:
        assert op in FILTER_OPERATORS, f"Operator {op} not found"


def test_build_filter_clause_simple() -> None:
    """Test building filter clause with simple equality filter."""
    from langchain_alibabacloud_mysql.vectorstores.alibabacloud_mysql import (
        AlibabaCloudMySQL,
    )

    with patch.object(AlibabaCloudMySQL, "__init__", lambda self, **kwargs: None):
        store = AlibabaCloudMySQL()

        where, params = store._build_filter_clause(filter={"category": "phone"})

        assert "WHERE" in where
        assert "JSON_UNQUOTE" in where
        assert "$.category" in where
        assert "= %s" in where
        assert params == ["phone"]


def test_build_filter_clause_with_eq_operator() -> None:
    """Test building filter clause with explicit $eq operator."""
    from langchain_alibabacloud_mysql.vectorstores.alibabacloud_mysql import (
        AlibabaCloudMySQL,
    )

    with patch.object(AlibabaCloudMySQL, "__init__", lambda self, **kwargs: None):
        store = AlibabaCloudMySQL()

        where, params = store._build_filter_clause(
            filter={"category": {"$eq": "phone"}}
        )

        assert "WHERE" in where
        assert "= %s" in where
        assert params == ["phone"]


def test_build_filter_clause_with_like_operator() -> None:
    """Test building filter clause with $like operator."""
    from langchain_alibabacloud_mysql.vectorstores.alibabacloud_mysql import (
        AlibabaCloudMySQL,
    )

    with patch.object(AlibabaCloudMySQL, "__init__", lambda self, **kwargs: None):
        store = AlibabaCloudMySQL()

        where, params = store._build_filter_clause(
            filter={"name": {"$like": "%phone%"}}
        )

        assert "WHERE" in where
        assert "LIKE %s" in where
        assert params == ["%phone%"]


def test_build_filter_clause_with_operators() -> None:
    """Test building filter clause with comparison operators."""
    from langchain_alibabacloud_mysql.vectorstores.alibabacloud_mysql import (
        AlibabaCloudMySQL,
    )

    with patch.object(AlibabaCloudMySQL, "__init__", lambda self, **kwargs: None):
        store = AlibabaCloudMySQL()

        # Test $gt operator with numeric value (should preserve type)
        where, params = store._build_filter_clause(filter={"price": {"$gt": 1000}})
        assert "> %s" in where
        assert params == [1000]  # Numeric value preserved
        assert "JSON_EXTRACT" in where  # Uses JSON_EXTRACT for numeric comparison

        # Test $lt operator with numeric value
        where, params = store._build_filter_clause(filter={"price": {"$lt": 500}})
        assert "< %s" in where
        assert params == [500]  # Numeric value preserved

        # Test $gte operator
        where, params = store._build_filter_clause(filter={"price": {"$gte": 100}})
        assert ">= %s" in where

        # Test $lte operator
        where, params = store._build_filter_clause(filter={"price": {"$lte": 200}})
        assert "<= %s" in where

        # Test $ne operator with string value
        where, params = store._build_filter_clause(
            filter={"status": {"$ne": "deleted"}}
        )
        assert "!= %s" in where
        assert params == ["deleted"]  # String value
        assert "JSON_UNQUOTE" in where  # Uses JSON_UNQUOTE for string comparison


def test_build_filter_clause_with_in_operator() -> None:
    """Test building filter clause with IN operator."""
    from langchain_alibabacloud_mysql.vectorstores.alibabacloud_mysql import (
        AlibabaCloudMySQL,
    )

    with patch.object(AlibabaCloudMySQL, "__init__", lambda self, **kwargs: None):
        store = AlibabaCloudMySQL()

        where, params = store._build_filter_clause(
            filter={"category": {"$in": ["phone", "tablet", "laptop"]}}
        )

        assert "IN (%s,%s,%s)" in where
        assert params == ["phone", "tablet", "laptop"]


def test_build_filter_clause_with_nin_operator() -> None:
    """Test building filter clause with NOT IN operator."""
    from langchain_alibabacloud_mysql.vectorstores.alibabacloud_mysql import (
        AlibabaCloudMySQL,
    )

    with patch.object(AlibabaCloudMySQL, "__init__", lambda self, **kwargs: None):
        store = AlibabaCloudMySQL()

        where, params = store._build_filter_clause(
            filter={"status": {"$nin": ["deleted", "archived"]}}
        )

        assert "NOT IN (%s,%s)" in where
        assert params == ["deleted", "archived"]


def test_build_filter_clause_multiple_conditions() -> None:
    """Test building filter clause with multiple conditions."""
    from langchain_alibabacloud_mysql.vectorstores.alibabacloud_mysql import (
        AlibabaCloudMySQL,
    )

    with patch.object(AlibabaCloudMySQL, "__init__", lambda self, **kwargs: None):
        store = AlibabaCloudMySQL()

        where, params = store._build_filter_clause(
            filter={"category": "phone", "price": {"$gt": 100, "$lt": 1000}}
        )

        assert "WHERE" in where
        assert "AND" in where
        # Should have 3 params: category, price $gt, price $lt
        assert len(params) == 3


def test_build_filter_clause_invalid_operator() -> None:
    """Test that invalid operator raises ValueError."""
    from langchain_alibabacloud_mysql.vectorstores.alibabacloud_mysql import (
        AlibabaCloudMySQL,
    )

    with patch.object(AlibabaCloudMySQL, "__init__", lambda self, **kwargs: None):
        store = AlibabaCloudMySQL()

        with pytest.raises(ValueError, match="Unsupported filter operator"):
            store._build_filter_clause(filter={"price": {"$invalid": 100}})


def test_build_filter_clause_in_requires_list() -> None:
    """Test that $in operator requires a list value."""
    from langchain_alibabacloud_mysql.vectorstores.alibabacloud_mysql import (
        AlibabaCloudMySQL,
    )

    with patch.object(AlibabaCloudMySQL, "__init__", lambda self, **kwargs: None):
        store = AlibabaCloudMySQL()

        with pytest.raises(ValueError, match="requires a list value"):
            store._build_filter_clause(filter={"category": {"$in": "not_a_list"}})


def test_build_filter_clause_nin_requires_list() -> None:
    """Test that $nin operator requires a list value."""
    from langchain_alibabacloud_mysql.vectorstores.alibabacloud_mysql import (
        AlibabaCloudMySQL,
    )

    with patch.object(AlibabaCloudMySQL, "__init__", lambda self, **kwargs: None):
        store = AlibabaCloudMySQL()

        with pytest.raises(ValueError, match="requires a list value"):
            store._build_filter_clause(filter={"status": {"$nin": "single_value"}})


def test_build_filter_clause_same_field_multiple_operators() -> None:
    """Test building filter clause with multiple operators on same field."""
    from langchain_alibabacloud_mysql.vectorstores.alibabacloud_mysql import (
        AlibabaCloudMySQL,
    )

    with patch.object(AlibabaCloudMySQL, "__init__", lambda self, **kwargs: None):
        store = AlibabaCloudMySQL()

        where, params = store._build_filter_clause(
            filter={"price": {"$gte": 100, "$lte": 1000}}
        )

        assert "WHERE" in where
        assert ">= %s" in where
        assert "<= %s" in where
        assert len(params) == 2
        assert 100 in params  # Numeric values preserved
        assert 1000 in params


def test_build_filter_clause_empty() -> None:
    """Test building filter clause with no filter returns empty."""
    from langchain_alibabacloud_mysql.vectorstores.alibabacloud_mysql import (
        AlibabaCloudMySQL,
    )

    with patch.object(AlibabaCloudMySQL, "__init__", lambda self, **kwargs: None):
        store = AlibabaCloudMySQL()

        where, params = store._build_filter_clause(filter=None)
        assert where == ""
        assert params == []

        where, params = store._build_filter_clause(filter={})
        assert where == ""
        assert params == []


def test_table_name_validation() -> None:
    """Test table name validation."""
    from langchain_alibabacloud_mysql.vectorstores.alibabacloud_mysql import (
        AlibabaCloudMySQL,
    )

    # Valid names
    assert AlibabaCloudMySQL._validate_table_name("valid_table") == "valid_table"
    assert AlibabaCloudMySQL._validate_table_name("Table123") == "Table123"
    assert AlibabaCloudMySQL._validate_table_name("_private") == "_private"

    # Invalid names
    with pytest.raises(ValueError, match="Invalid table name"):
        AlibabaCloudMySQL._validate_table_name("123invalid")

    with pytest.raises(ValueError, match="Invalid table name"):
        AlibabaCloudMySQL._validate_table_name("table-name")

    with pytest.raises(ValueError, match="Invalid table name"):
        AlibabaCloudMySQL._validate_table_name("table name")

    # Too long name
    with pytest.raises(ValueError, match="Table name too long"):
        AlibabaCloudMySQL._validate_table_name("a" * 65)


# =============================================================================
# add_documents Tests
# =============================================================================


def test_add_documents_extracts_content_and_metadata() -> None:
    """Test that add_documents extracts content and metadata from documents."""
    from unittest.mock import patch

    from langchain_core.documents import Document

    from langchain_alibabacloud_mysql.vectorstores.alibabacloud_mysql import (
        AlibabaCloudMySQL,
    )

    with patch.object(AlibabaCloudMySQL, "__init__", lambda self, **kwargs: None):
        store = AlibabaCloudMySQL()
        store.add_texts = MagicMock(return_value=["id1", "id2"])

        documents = [
            Document(page_content="Hello world", metadata={"source": "web"}),
            Document(page_content="LangChain is great", metadata={"source": "doc"}),
        ]

        result = store.add_documents(documents)

        # Verify add_texts was called with correct arguments
        store.add_texts.assert_called_once()
        call_kwargs = store.add_texts.call_args
        assert call_kwargs[1]["texts"] == ["Hello world", "LangChain is great"]
        assert call_kwargs[1]["metadatas"] == [{"source": "web"}, {"source": "doc"}]

        assert result == ["id1", "id2"]


def test_add_documents_uses_document_ids() -> None:
    """Test that add_documents uses document.id if provided."""
    from unittest.mock import patch

    from langchain_core.documents import Document

    from langchain_alibabacloud_mysql.vectorstores.alibabacloud_mysql import (
        AlibabaCloudMySQL,
    )

    with patch.object(AlibabaCloudMySQL, "__init__", lambda self, **kwargs: None):
        store = AlibabaCloudMySQL()
        store.add_texts = MagicMock(return_value=["doc-id-1", "doc-id-2"])

        documents = [
            Document(id="doc-id-1", page_content="Hello", metadata={}),
            Document(id="doc-id-2", page_content="World", metadata={}),
        ]

        store.add_documents(documents)

        # Verify add_texts was called with document IDs
        call_kwargs = store.add_texts.call_args
        assert call_kwargs[1]["ids"] == ["doc-id-1", "doc-id-2"]


def test_add_documents_generates_ids_when_missing() -> None:
    """Test that add_documents generates UUIDs when document.id is None."""
    from unittest.mock import patch

    from langchain_core.documents import Document

    from langchain_alibabacloud_mysql.vectorstores.alibabacloud_mysql import (
        AlibabaCloudMySQL,
    )

    with patch.object(AlibabaCloudMySQL, "__init__", lambda self, **kwargs: None):
        store = AlibabaCloudMySQL()
        store.add_texts = MagicMock(return_value=["id1", "id2"])

        documents = [
            Document(page_content="Hello", metadata={}),  # No id
            Document(page_content="World", metadata={}),  # No id
        ]

        store.add_documents(documents)

        # Verify add_texts was called with generated IDs (UUIDs)
        call_kwargs = store.add_texts.call_args
        ids = call_kwargs[1]["ids"]
        assert len(ids) == 2
        # UUIDs should be 36 characters (with hyphens)
        assert all(len(id_) == 36 for id_ in ids)


def test_add_documents_uses_provided_ids_over_document_ids() -> None:
    """Test that explicitly provided ids are used over document.id."""
    from unittest.mock import patch

    from langchain_core.documents import Document

    from langchain_alibabacloud_mysql.vectorstores.alibabacloud_mysql import (
        AlibabaCloudMySQL,
    )

    with patch.object(AlibabaCloudMySQL, "__init__", lambda self, **kwargs: None):
        store = AlibabaCloudMySQL()
        store.add_texts = MagicMock(return_value=["custom-1", "custom-2"])

        documents = [
            Document(id="doc-id-1", page_content="Hello", metadata={}),
            Document(id="doc-id-2", page_content="World", metadata={}),
        ]

        store.add_documents(documents, ids=["custom-1", "custom-2"])

        # Verify add_texts was called with explicitly provided IDs
        call_kwargs = store.add_texts.call_args
        assert call_kwargs[1]["ids"] == ["custom-1", "custom-2"]


def test_add_documents_empty_list() -> None:
    """Test that add_documents returns empty list for empty input."""
    from unittest.mock import patch

    from langchain_alibabacloud_mysql.vectorstores.alibabacloud_mysql import (
        AlibabaCloudMySQL,
    )

    with patch.object(AlibabaCloudMySQL, "__init__", lambda self, **kwargs: None):
        store = AlibabaCloudMySQL()

        result = store.add_documents([])

        assert result == []


def test_add_documents_passes_batch_size() -> None:
    """Test that add_documents passes batch_size to add_texts."""
    from unittest.mock import patch

    from langchain_core.documents import Document

    from langchain_alibabacloud_mysql.vectorstores.alibabacloud_mysql import (
        AlibabaCloudMySQL,
    )

    with patch.object(AlibabaCloudMySQL, "__init__", lambda self, **kwargs: None):
        store = AlibabaCloudMySQL()
        store.add_texts = MagicMock(return_value=["id1"])

        documents = [Document(page_content="Hello", metadata={})]

        store.add_documents(documents, batch_size=100)

        call_kwargs = store.add_texts.call_args
        assert call_kwargs[1]["batch_size"] == 100


# =============================================================================
# add_embeddings Tests
# =============================================================================


def test_add_embeddings_inserts_correctly() -> None:
    """Test that add_embeddings inserts pre-computed embeddings."""
    from unittest.mock import patch

    from langchain_alibabacloud_mysql.vectorstores.alibabacloud_mysql import (
        AlibabaCloudMySQL,
    )

    with patch.object(AlibabaCloudMySQL, "__init__", lambda self, **kwargs: None):
        store = AlibabaCloudMySQL()
        store._table_name = "test_table"
        store._create_table_if_not_exists = MagicMock()
        store._vector_to_string = lambda v: "[" + ",".join(map(str, v)) + "]"

        mock_cursor = MagicMock()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)
        store._get_cursor = MagicMock(return_value=mock_cursor)

        text_embeddings = [
            ("Hello world", [0.1, 0.2, 0.3]),
            ("LangChain", [0.4, 0.5, 0.6]),
        ]

        ids = store.add_embeddings(
            text_embeddings=text_embeddings,
            metadatas=[{"source": "web"}, {"source": "doc"}],
            ids=["id1", "id2"],
        )

        assert ids == ["id1", "id2"]
        store._create_table_if_not_exists.assert_called_once_with(3)  # dimension=3
        mock_cursor.executemany.assert_called_once()


def test_add_embeddings_generates_ids() -> None:
    """Test that add_embeddings generates UUIDs when ids not provided."""
    from unittest.mock import patch

    from langchain_alibabacloud_mysql.vectorstores.alibabacloud_mysql import (
        AlibabaCloudMySQL,
    )

    with patch.object(AlibabaCloudMySQL, "__init__", lambda self, **kwargs: None):
        store = AlibabaCloudMySQL()
        store._table_name = "test_table"
        store._create_table_if_not_exists = MagicMock()
        store._vector_to_string = lambda v: "[" + ",".join(map(str, v)) + "]"

        mock_cursor = MagicMock()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)
        store._get_cursor = MagicMock(return_value=mock_cursor)

        text_embeddings = [
            ("Hello", [0.1, 0.2, 0.3]),
            ("World", [0.4, 0.5, 0.6]),
        ]

        ids = store.add_embeddings(text_embeddings=text_embeddings)

        assert len(ids) == 2
        # UUIDs should be 36 characters
        assert all(len(id_) == 36 for id_ in ids)


def test_add_embeddings_empty_list() -> None:
    """Test that add_embeddings returns empty list for empty input."""
    from unittest.mock import patch

    from langchain_alibabacloud_mysql.vectorstores.alibabacloud_mysql import (
        AlibabaCloudMySQL,
    )

    with patch.object(AlibabaCloudMySQL, "__init__", lambda self, **kwargs: None):
        store = AlibabaCloudMySQL()

        result = store.add_embeddings(text_embeddings=[])

        assert result == []


def test_add_embeddings_validates_lengths() -> None:
    """Test that add_embeddings validates that all arrays have same length."""
    from unittest.mock import patch

    from langchain_alibabacloud_mysql.vectorstores.alibabacloud_mysql import (
        AlibabaCloudMySQL,
    )

    with patch.object(AlibabaCloudMySQL, "__init__", lambda self, **kwargs: None):
        store = AlibabaCloudMySQL()
        store._table_name = "test_table"
        store._create_table_if_not_exists = MagicMock()

        text_embeddings = [
            ("Hello", [0.1, 0.2, 0.3]),
            ("World", [0.4, 0.5, 0.6]),
        ]

        # Mismatched metadatas length
        with pytest.raises(ValueError, match="must match"):
            store.add_embeddings(
                text_embeddings=text_embeddings,
                metadatas=[{"source": "web"}],  # Only 1 metadata for 2 embeddings
            )

        # Mismatched ids length
        with pytest.raises(ValueError, match="must match"):
            store.add_embeddings(
                text_embeddings=text_embeddings,
                ids=["id1"],  # Only 1 id for 2 embeddings
            )


def test_add_embeddings_generates_default_metadata() -> None:
    """Test that add_embeddings uses empty dict when metadatas not provided."""
    import json
    from unittest.mock import patch

    from langchain_alibabacloud_mysql.vectorstores.alibabacloud_mysql import (
        AlibabaCloudMySQL,
    )

    with patch.object(AlibabaCloudMySQL, "__init__", lambda self, **kwargs: None):
        store = AlibabaCloudMySQL()
        store._table_name = "test_table"
        store._create_table_if_not_exists = MagicMock()
        store._vector_to_string = lambda v: "[" + ",".join(map(str, v)) + "]"

        mock_cursor = MagicMock()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)
        store._get_cursor = MagicMock(return_value=mock_cursor)

        text_embeddings = [("Hello", [0.1, 0.2, 0.3])]

        store.add_embeddings(text_embeddings=text_embeddings, ids=["id1"])

        # Check that empty metadata was used
        call_args = mock_cursor.executemany.call_args
        batch_values = call_args[0][1]
        assert json.loads(batch_values[0][2]) == {}


def test_add_embeddings_batch_processing() -> None:
    """Test that add_embeddings processes in batches correctly."""
    from unittest.mock import patch

    from langchain_alibabacloud_mysql.vectorstores.alibabacloud_mysql import (
        AlibabaCloudMySQL,
    )

    with patch.object(AlibabaCloudMySQL, "__init__", lambda self, **kwargs: None):
        store = AlibabaCloudMySQL()
        store._table_name = "test_table"
        store._create_table_if_not_exists = MagicMock()
        store._vector_to_string = lambda v: "[" + ",".join(map(str, v)) + "]"

        mock_cursor = MagicMock()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)
        store._get_cursor = MagicMock(return_value=mock_cursor)

        # Create 5 text embeddings
        text_embeddings = [
            (f"Text {i}", [float(i), float(i), float(i)]) for i in range(5)
        ]

        # Use batch_size=2, should result in 3 batches (2, 2, 1)
        store.add_embeddings(
            text_embeddings=text_embeddings,
            batch_size=2,
        )

        # executemany should be called 3 times
        assert mock_cursor.executemany.call_count == 3


# =============================================================================
# upsert Tests
# =============================================================================


def test_upsert_with_documents() -> None:
    """Test upsert with documents."""
    from unittest.mock import patch

    from langchain_core.documents import Document

    from langchain_alibabacloud_mysql.vectorstores.alibabacloud_mysql import (
        AlibabaCloudMySQL,
    )

    with patch.object(AlibabaCloudMySQL, "__init__", lambda self, **kwargs: None):
        store = AlibabaCloudMySQL()
        store._table_name = "test_table"
        store._embedding = MagicMock()
        store._embedding.embed_documents = MagicMock(
            return_value=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        )
        store._create_table_if_not_exists = MagicMock()
        store._vector_to_string = lambda v: "[" + ",".join(map(str, v)) + "]"

        mock_cursor = MagicMock()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)
        store._get_cursor = MagicMock(return_value=mock_cursor)

        documents = [
            Document(id="doc1", page_content="Hello", metadata={"key": "value1"}),
            Document(id="doc2", page_content="World", metadata={"key": "value2"}),
        ]

        ids = store.upsert(documents)

        assert ids == ["doc1", "doc2"]
        store._embedding.embed_documents.assert_called_once_with(["Hello", "World"])
        store._create_table_if_not_exists.assert_called_once_with(3)
        mock_cursor.executemany.assert_called_once()


def test_upsert_uses_document_ids() -> None:
    """Test that upsert uses document.id if available."""
    from unittest.mock import patch

    from langchain_core.documents import Document

    from langchain_alibabacloud_mysql.vectorstores.alibabacloud_mysql import (
        AlibabaCloudMySQL,
    )

    with patch.object(AlibabaCloudMySQL, "__init__", lambda self, **kwargs: None):
        store = AlibabaCloudMySQL()
        store._table_name = "test_table"
        store._embedding = MagicMock()
        store._embedding.embed_documents = MagicMock(return_value=[[0.1, 0.2, 0.3]])
        store._create_table_if_not_exists = MagicMock()
        store._vector_to_string = lambda v: "[" + ",".join(map(str, v)) + "]"

        mock_cursor = MagicMock()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)
        store._get_cursor = MagicMock(return_value=mock_cursor)

        documents = [
            Document(id="custom-id-123", page_content="Test", metadata={}),
        ]

        ids = store.upsert(documents)

        assert ids == ["custom-id-123"]


def test_upsert_generates_ids_when_missing() -> None:
    """Test that upsert generates UUIDs when document.id is None."""
    from unittest.mock import patch

    from langchain_core.documents import Document

    from langchain_alibabacloud_mysql.vectorstores.alibabacloud_mysql import (
        AlibabaCloudMySQL,
    )

    with patch.object(AlibabaCloudMySQL, "__init__", lambda self, **kwargs: None):
        store = AlibabaCloudMySQL()
        store._table_name = "test_table"
        store._embedding = MagicMock()
        store._embedding.embed_documents = MagicMock(
            return_value=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        )
        store._create_table_if_not_exists = MagicMock()
        store._vector_to_string = lambda v: "[" + ",".join(map(str, v)) + "]"

        mock_cursor = MagicMock()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)
        store._get_cursor = MagicMock(return_value=mock_cursor)

        documents = [
            Document(page_content="No ID 1", metadata={}),
            Document(page_content="No ID 2", metadata={}),
        ]

        ids = store.upsert(documents)

        assert len(ids) == 2
        # UUIDs are 36 characters
        assert all(len(id_) == 36 for id_ in ids)


def test_upsert_with_explicit_ids() -> None:
    """Test upsert with explicitly provided ids."""
    from unittest.mock import patch

    from langchain_core.documents import Document

    from langchain_alibabacloud_mysql.vectorstores.alibabacloud_mysql import (
        AlibabaCloudMySQL,
    )

    with patch.object(AlibabaCloudMySQL, "__init__", lambda self, **kwargs: None):
        store = AlibabaCloudMySQL()
        store._table_name = "test_table"
        store._embedding = MagicMock()
        store._embedding.embed_documents = MagicMock(
            return_value=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        )
        store._create_table_if_not_exists = MagicMock()
        store._vector_to_string = lambda v: "[" + ",".join(map(str, v)) + "]"

        mock_cursor = MagicMock()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)
        store._get_cursor = MagicMock(return_value=mock_cursor)

        documents = [
            Document(id="ignored-id", page_content="Text 1", metadata={}),
            Document(id="also-ignored", page_content="Text 2", metadata={}),
        ]

        # Explicit IDs should override document.id
        ids = store.upsert(documents, ids=["explicit-1", "explicit-2"])

        assert ids == ["explicit-1", "explicit-2"]


def test_upsert_empty_list() -> None:
    """Test upsert with empty document list."""
    from unittest.mock import patch

    from langchain_alibabacloud_mysql.vectorstores.alibabacloud_mysql import (
        AlibabaCloudMySQL,
    )

    with patch.object(AlibabaCloudMySQL, "__init__", lambda self, **kwargs: None):
        store = AlibabaCloudMySQL()

        ids = store.upsert([])

        assert ids == []


def test_upsert_validates_ids_length() -> None:
    """Test that upsert validates ids length matches documents length."""
    from unittest.mock import patch

    from langchain_core.documents import Document

    from langchain_alibabacloud_mysql.vectorstores.alibabacloud_mysql import (
        AlibabaCloudMySQL,
    )

    with patch.object(AlibabaCloudMySQL, "__init__", lambda self, **kwargs: None):
        store = AlibabaCloudMySQL()

        documents = [
            Document(page_content="Doc 1", metadata={}),
            Document(page_content="Doc 2", metadata={}),
        ]

        with pytest.raises(ValueError, match="must match"):
            store.upsert(documents, ids=["only-one-id"])


def test_upsert_batch_processing() -> None:
    """Test that upsert processes in batches correctly."""
    from unittest.mock import patch

    from langchain_core.documents import Document

    from langchain_alibabacloud_mysql.vectorstores.alibabacloud_mysql import (
        AlibabaCloudMySQL,
    )

    with patch.object(AlibabaCloudMySQL, "__init__", lambda self, **kwargs: None):
        store = AlibabaCloudMySQL()
        store._table_name = "test_table"
        store._embedding = MagicMock()
        store._embedding.embed_documents = MagicMock(
            return_value=[[float(i), float(i), float(i)] for i in range(5)]
        )
        store._create_table_if_not_exists = MagicMock()
        store._vector_to_string = lambda v: "[" + ",".join(map(str, v)) + "]"

        mock_cursor = MagicMock()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)
        store._get_cursor = MagicMock(return_value=mock_cursor)

        documents = [
            Document(id=f"doc-{i}", page_content=f"Text {i}", metadata={})
            for i in range(5)
        ]

        # Use batch_size=2, should result in 3 batches (2, 2, 1)
        store.upsert(documents, batch_size=2)

        # executemany should be called 3 times
        assert mock_cursor.executemany.call_count == 3


# =============================================================================
# Async Method Tests
# =============================================================================


@pytest.mark.asyncio
async def test_aadd_texts_calls_aembed_documents() -> None:
    """Test that aadd_texts uses async embedding when available."""
    from unittest.mock import AsyncMock, MagicMock, patch

    from langchain_alibabacloud_mysql.vectorstores.alibabacloud_mysql import (
        AlibabaCloudMySQL,
    )

    with patch.object(AlibabaCloudMySQL, "__init__", lambda self, **kwargs: None):
        store = AlibabaCloudMySQL()
        store._table_name = "test_table"
        store._embedding = MagicMock()
        store._embedding.aembed_documents = AsyncMock(
            return_value=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        )
        store._acreate_table_if_not_exists = AsyncMock()
        store._vector_to_string = lambda v: "[" + ",".join(map(str, v)) + "]"

        mock_cursor = AsyncMock()
        mock_cursor.__aenter__ = AsyncMock(return_value=mock_cursor)
        mock_cursor.__aexit__ = AsyncMock(return_value=False)
        store._aget_cursor = MagicMock(return_value=mock_cursor)

        ids = await store.aadd_texts(["Text 1", "Text 2"])

        assert len(ids) == 2
        store._embedding.aembed_documents.assert_called_once()


@pytest.mark.asyncio
async def test_aadd_texts_empty_list() -> None:
    """Test that aadd_texts returns empty list for empty input."""
    from unittest.mock import patch

    from langchain_alibabacloud_mysql.vectorstores.alibabacloud_mysql import (
        AlibabaCloudMySQL,
    )

    with patch.object(AlibabaCloudMySQL, "__init__", lambda self, **kwargs: None):
        store = AlibabaCloudMySQL()

        ids = await store.aadd_texts([])

        assert ids == []


@pytest.mark.asyncio
async def test_aadd_documents_extracts_content() -> None:
    """Test that aadd_documents extracts content from documents."""
    from unittest.mock import AsyncMock, patch

    from langchain_core.documents import Document

    from langchain_alibabacloud_mysql.vectorstores.alibabacloud_mysql import (
        AlibabaCloudMySQL,
    )

    with patch.object(AlibabaCloudMySQL, "__init__", lambda self, **kwargs: None):
        store = AlibabaCloudMySQL()
        store.aadd_texts = AsyncMock(return_value=["id-1", "id-2"])

        documents = [
            Document(page_content="Text 1", metadata={"key": "value1"}),
            Document(page_content="Text 2", metadata={"key": "value2"}),
        ]

        ids = await store.aadd_documents(documents)

        assert ids == ["id-1", "id-2"]
        store.aadd_texts.assert_called_once()
        call_args = store.aadd_texts.call_args
        assert call_args.kwargs["texts"] == ["Text 1", "Text 2"]
        assert call_args.kwargs["metadatas"] == [{"key": "value1"}, {"key": "value2"}]


@pytest.mark.asyncio
async def test_aadd_documents_uses_document_ids() -> None:
    """Test that aadd_documents uses document.id when available."""
    from unittest.mock import AsyncMock, patch

    from langchain_core.documents import Document

    from langchain_alibabacloud_mysql.vectorstores.alibabacloud_mysql import (
        AlibabaCloudMySQL,
    )

    with patch.object(AlibabaCloudMySQL, "__init__", lambda self, **kwargs: None):
        store = AlibabaCloudMySQL()
        store.aadd_texts = AsyncMock(return_value=["doc-id-1", "doc-id-2"])

        documents = [
            Document(id="doc-id-1", page_content="Text 1", metadata={}),
            Document(id="doc-id-2", page_content="Text 2", metadata={}),
        ]

        await store.aadd_documents(documents)

        call_args = store.aadd_texts.call_args
        assert call_args.kwargs["ids"] == ["doc-id-1", "doc-id-2"]


@pytest.mark.asyncio
async def test_aadd_documents_empty_list() -> None:
    """Test that aadd_documents returns empty list for empty input."""
    from unittest.mock import patch

    from langchain_alibabacloud_mysql.vectorstores.alibabacloud_mysql import (
        AlibabaCloudMySQL,
    )

    with patch.object(AlibabaCloudMySQL, "__init__", lambda self, **kwargs: None):
        store = AlibabaCloudMySQL()

        ids = await store.aadd_documents([])

        assert ids == []


@pytest.mark.asyncio
async def test_aadd_embeddings_inserts_correctly() -> None:
    """Test that aadd_embeddings inserts pre-computed embeddings."""
    from unittest.mock import AsyncMock, MagicMock, patch

    from langchain_alibabacloud_mysql.vectorstores.alibabacloud_mysql import (
        AlibabaCloudMySQL,
    )

    with patch.object(AlibabaCloudMySQL, "__init__", lambda self, **kwargs: None):
        store = AlibabaCloudMySQL()
        store._table_name = "test_table"
        store._acreate_table_if_not_exists = AsyncMock()
        store._vector_to_string = lambda v: "[" + ",".join(map(str, v)) + "]"

        mock_cursor = AsyncMock()
        mock_cursor.__aenter__ = AsyncMock(return_value=mock_cursor)
        mock_cursor.__aexit__ = AsyncMock(return_value=False)
        store._aget_cursor = MagicMock(return_value=mock_cursor)

        text_embeddings = [
            ("Text 1", [1.0, 2.0, 3.0]),
            ("Text 2", [4.0, 5.0, 6.0]),
        ]

        ids = await store.aadd_embeddings(
            text_embeddings=text_embeddings,
            metadatas=[{"key": "v1"}, {"key": "v2"}],
        )

        assert len(ids) == 2
        mock_cursor.executemany.assert_called_once()


@pytest.mark.asyncio
async def test_aadd_embeddings_empty_list() -> None:
    """Test that aadd_embeddings returns empty list for empty input."""
    from unittest.mock import patch

    from langchain_alibabacloud_mysql.vectorstores.alibabacloud_mysql import (
        AlibabaCloudMySQL,
    )

    with patch.object(AlibabaCloudMySQL, "__init__", lambda self, **kwargs: None):
        store = AlibabaCloudMySQL()

        ids = await store.aadd_embeddings([])

        assert ids == []


@pytest.mark.asyncio
async def test_aupsert_with_documents() -> None:
    """Test that aupsert works with documents."""
    from unittest.mock import AsyncMock, MagicMock, patch

    from langchain_core.documents import Document

    from langchain_alibabacloud_mysql.vectorstores.alibabacloud_mysql import (
        AlibabaCloudMySQL,
    )

    with patch.object(AlibabaCloudMySQL, "__init__", lambda self, **kwargs: None):
        store = AlibabaCloudMySQL()
        store._table_name = "test_table"
        store._embedding = MagicMock()
        store._embedding.aembed_documents = AsyncMock(
            return_value=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        )
        store._acreate_table_if_not_exists = AsyncMock()
        store._vector_to_string = lambda v: "[" + ",".join(map(str, v)) + "]"

        mock_cursor = AsyncMock()
        mock_cursor.__aenter__ = AsyncMock(return_value=mock_cursor)
        mock_cursor.__aexit__ = AsyncMock(return_value=False)
        store._aget_cursor = MagicMock(return_value=mock_cursor)

        documents = [
            Document(id="doc-1", page_content="Text 1", metadata={"key": "v1"}),
            Document(id="doc-2", page_content="Text 2", metadata={"key": "v2"}),
        ]

        ids = await store.aupsert(documents)

        assert ids == ["doc-1", "doc-2"]
        mock_cursor.executemany.assert_called_once()


@pytest.mark.asyncio
async def test_aupsert_empty_list() -> None:
    """Test that aupsert returns empty list for empty input."""
    from unittest.mock import patch

    from langchain_alibabacloud_mysql.vectorstores.alibabacloud_mysql import (
        AlibabaCloudMySQL,
    )

    with patch.object(AlibabaCloudMySQL, "__init__", lambda self, **kwargs: None):
        store = AlibabaCloudMySQL()

        ids = await store.aupsert([])

        assert ids == []


@pytest.mark.asyncio
async def test_aupsert_validates_ids_length() -> None:
    """Test that aupsert validates ids length."""
    from unittest.mock import patch

    from langchain_core.documents import Document

    from langchain_alibabacloud_mysql.vectorstores.alibabacloud_mysql import (
        AlibabaCloudMySQL,
    )

    with patch.object(AlibabaCloudMySQL, "__init__", lambda self, **kwargs: None):
        store = AlibabaCloudMySQL()

        documents = [
            Document(page_content="Doc 1", metadata={}),
            Document(page_content="Doc 2", metadata={}),
        ]

        with pytest.raises(ValueError, match="must match"):
            await store.aupsert(documents, ids=["only-one"])


@pytest.mark.asyncio
async def test_asimilarity_search_calls_async_methods() -> None:
    """Test that asimilarity_search uses async search method."""
    from unittest.mock import AsyncMock, patch

    from langchain_core.documents import Document

    from langchain_alibabacloud_mysql.vectorstores.alibabacloud_mysql import (
        AlibabaCloudMySQL,
    )

    with patch.object(AlibabaCloudMySQL, "__init__", lambda self, **kwargs: None):
        store = AlibabaCloudMySQL()
        store.asimilarity_search_with_score = AsyncMock(
            return_value=[
                (Document(page_content="Result 1", metadata={}), 0.9),
                (Document(page_content="Result 2", metadata={}), 0.8),
            ]
        )

        results = await store.asimilarity_search("query", k=2)

        assert len(results) == 2
        assert results[0].page_content == "Result 1"
        store.asimilarity_search_with_score.assert_called_once()


@pytest.mark.asyncio
async def test_asimilarity_search_with_score_uses_aembed_query() -> None:
    """Test that asimilarity_search_with_score uses async embedding."""
    from unittest.mock import AsyncMock, MagicMock, patch

    from langchain_core.documents import Document

    from langchain_alibabacloud_mysql.vectorstores.alibabacloud_mysql import (
        AlibabaCloudMySQL,
    )

    with patch.object(AlibabaCloudMySQL, "__init__", lambda self, **kwargs: None):
        store = AlibabaCloudMySQL()
        store._embedding = MagicMock()
        store._embedding.aembed_query = AsyncMock(return_value=[1.0, 2.0, 3.0])
        store.asimilarity_search_with_score_by_vector = AsyncMock(
            return_value=[
                (Document(page_content="Result", metadata={}), 0.9),
            ]
        )

        await store.asimilarity_search_with_score("query", k=1)

        store._embedding.aembed_query.assert_called_once_with("query")


@pytest.mark.asyncio
async def test_adelete_executes_delete_query() -> None:
    """Test that adelete executes delete SQL."""
    from unittest.mock import AsyncMock, MagicMock, patch

    from langchain_alibabacloud_mysql.vectorstores.alibabacloud_mysql import (
        AlibabaCloudMySQL,
    )

    with patch.object(AlibabaCloudMySQL, "__init__", lambda self, **kwargs: None):
        store = AlibabaCloudMySQL()
        store._table_name = "test_table"

        mock_cursor = AsyncMock()
        mock_cursor.__aenter__ = AsyncMock(return_value=mock_cursor)
        mock_cursor.__aexit__ = AsyncMock(return_value=False)
        store._aget_cursor = MagicMock(return_value=mock_cursor)

        result = await store.adelete(ids=["id-1", "id-2"])

        assert result is True
        mock_cursor.execute.assert_called_once()


@pytest.mark.asyncio
async def test_adelete_returns_none_for_empty_ids() -> None:
    """Test that adelete returns None for empty ids."""
    from unittest.mock import patch

    from langchain_alibabacloud_mysql.vectorstores.alibabacloud_mysql import (
        AlibabaCloudMySQL,
    )

    with patch.object(AlibabaCloudMySQL, "__init__", lambda self, **kwargs: None):
        store = AlibabaCloudMySQL()

        result = await store.adelete(ids=None)

        assert result is None


@pytest.mark.asyncio
async def test_aget_by_ids_returns_documents() -> None:
    """Test that aget_by_ids returns documents."""
    from unittest.mock import AsyncMock, MagicMock, patch

    from langchain_alibabacloud_mysql.vectorstores.alibabacloud_mysql import (
        AlibabaCloudMySQL,
    )

    with patch.object(AlibabaCloudMySQL, "__init__", lambda self, **kwargs: None):
        store = AlibabaCloudMySQL()
        store._table_name = "test_table"

        mock_records = [
            {"id": "id-1", "text": "Text 1", "metadata": '{"key": "v1"}'},
            {"id": "id-2", "text": "Text 2", "metadata": '{"key": "v2"}'},
        ]

        # Create a proper async iterator
        async def async_iter():
            for record in mock_records:
                yield record

        mock_cursor = AsyncMock()
        mock_cursor.__aenter__ = AsyncMock(return_value=mock_cursor)
        mock_cursor.__aexit__ = AsyncMock(return_value=False)
        mock_cursor.__aiter__ = lambda self: async_iter()
        store._aget_cursor = MagicMock(return_value=mock_cursor)

        docs = await store.aget_by_ids(["id-1", "id-2"])

        assert len(docs) == 2
        assert docs[0].page_content == "Text 1"
        assert docs[1].page_content == "Text 2"


@pytest.mark.asyncio
async def test_aget_by_ids_empty_list() -> None:
    """Test that aget_by_ids returns empty list for empty input."""
    from unittest.mock import patch

    from langchain_alibabacloud_mysql.vectorstores.alibabacloud_mysql import (
        AlibabaCloudMySQL,
    )

    with patch.object(AlibabaCloudMySQL, "__init__", lambda self, **kwargs: None):
        store = AlibabaCloudMySQL()

        docs = await store.aget_by_ids([])

        assert docs == []


@pytest.mark.asyncio
async def test_amax_marginal_relevance_search_uses_async() -> None:
    """Test that amax_marginal_relevance_search uses async methods."""
    from unittest.mock import AsyncMock, MagicMock, patch

    from langchain_core.documents import Document

    from langchain_alibabacloud_mysql.vectorstores.alibabacloud_mysql import (
        AlibabaCloudMySQL,
    )

    with patch.object(AlibabaCloudMySQL, "__init__", lambda self, **kwargs: None):
        store = AlibabaCloudMySQL()
        store._embedding = MagicMock()
        store._embedding.aembed_query = AsyncMock(return_value=[1.0, 2.0, 3.0])
        store.amax_marginal_relevance_search_by_vector = AsyncMock(
            return_value=[Document(page_content="Result", metadata={})]
        )

        await store.amax_marginal_relevance_search("query", k=4)

        store._embedding.aembed_query.assert_called_once_with("query")
        store.amax_marginal_relevance_search_by_vector.assert_called_once()
