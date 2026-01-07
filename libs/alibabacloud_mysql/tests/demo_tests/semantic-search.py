"""Semantic search demo with automatic .env file loading.

This script demonstrates semantic search capabilities using:
- AlibabaCloud MySQL Vector Store for storing and retrieving document embeddings
- DashScope embeddings for generating vector representations
- PDF document loading and text splitting
"""

import asyncio
import os
import time
from pathlib import Path
from typing import List, Tuple

# =============================================================================
# Environment Variables Loading
# =============================================================================

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    # Try to find .env file in multiple locations
    current_file = Path(__file__).resolve()
    env_paths = [
        current_file.parent / ".env",  # Same directory as this file
        current_file.parent.parent / ".env",  # Parent directory
        Path.cwd() / ".env",  # Current working directory
    ]

    for env_path in env_paths:
        if env_path.exists():
            load_dotenv(env_path)
            print(f"✓ Loaded environment variables from: {env_path}")
            break
    else:
        # If no .env file found, try loading from default location
        # This will look for .env in current directory and parent directories
        load_dotenv()
except ImportError:
    # python-dotenv not installed, use system environment variables only
    print(
        "⚠ Warning: python-dotenv not installed. "
        "Using system environment variables only."
    )
    print("To enable .env file support, install it with: pip install python-dotenv")

# =============================================================================
# Imports
# =============================================================================

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Import AlibabaCloudMySQL
from langchain_alibabacloud_mysql import AlibabaCloudMySQL

# =============================================================================
# Configuration
# =============================================================================

# PDF file path
PDF_FILE_PATH = "./data/nke-10k-2023.pdf"

# Text splitter configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Vector store configuration
TABLE_NAME = "langchain_vectors_semantic_search"
DISTANCE_STRATEGY = "cosine"  # or "euclidean"
HNSW_M = 6  # HNSW index M parameter (3-200)

# =============================================================================
# Helper Functions
# =============================================================================


def get_dashscope_embeddings() -> Embeddings:
    """Get DashScope embeddings model.

    Uses text-embedding-v4 model from Alibaba Cloud DashScope.
    Dimension: 1024 (default) or 3072 (with dimensions parameter)

    Returns:
        DashScopeEmbeddings instance, or None if API key is not set.
    """
    try:
        from langchain_community.embeddings import DashScopeEmbeddings
    except ImportError:
        print("❌ Error: langchain-community not installed.")
        print("   Install with: pip install langchain-community dashscope")
        return None

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("❌ Error: DASHSCOPE_API_KEY environment variable not set")
        return None

    return DashScopeEmbeddings(
        model="text-embedding-v4",
        dashscope_api_key=api_key,
    )


def create_vector_store(embeddings: Embeddings) -> AlibabaCloudMySQL:
    """Create and initialize AlibabaCloudMySQL vector store.

    Args:
        embeddings: The embedding model to use.

    Returns:
        AlibabaCloudMySQL vector store instance.

    Raises:
        ValueError: If required MySQL connection settings are missing.
    """
    # Get MySQL connection settings from environment variables
    host = os.getenv("ALIBABACLOUD_MYSQL_HOST")
    port = int(os.getenv("ALIBABACLOUD_MYSQL_PORT", "3306"))
    user = os.getenv("ALIBABACLOUD_MYSQL_USER")
    password = os.getenv("ALIBABACLOUD_MYSQL_PASSWORD")
    database = os.getenv("ALIBABACLOUD_MYSQL_DATABASE")

    if not all([host, user, password, database]):
        raise ValueError(
            "Missing required MySQL connection settings. "
            "Please set ALIBABACLOUD_MYSQL_HOST, ALIBABACLOUD_MYSQL_USER, "
            "ALIBABACLOUD_MYSQL_PASSWORD, and ALIBABACLOUD_MYSQL_DATABASE "
            "in your .env file."
        )

    # Create vector store
    vectorstore = AlibabaCloudMySQL(
        host=host,
        port=port,
        user=user,
        password=password,
        database=database,
        embedding=embeddings,
        table_name=TABLE_NAME,
        distance_strategy=DISTANCE_STRATEGY,
        hnsw_m=HNSW_M,
    )

    return vectorstore


def load_and_split_documents(file_path: str) -> List[Document]:
    """Load PDF document and split it into chunks.

    Args:
        file_path: Path to the PDF file.

    Returns:
        List of Document objects representing the text chunks.
    """
    # Load PDF document
    print(f"Loading PDF from: {file_path}")
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    print(f"✓ Loaded {len(docs)} pages from PDF")

    # Display first page preview
    if docs:
        print("\nFirst page preview (first 200 chars):")
        print(f"  {docs[0].page_content[:200]}...")
        print(f"\nFirst page metadata: {docs[0].metadata}")

    # Split documents into chunks
    print("\nSplitting documents into chunks...")
    print(f"  Chunk size: {CHUNK_SIZE}")
    print(f"  Chunk overlap: {CHUNK_OVERLAP}")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs)
    print(f"✓ Created {len(all_splits)} text chunks")

    return all_splits


def test_embedding_generation(
    embeddings: Embeddings, documents: List[Document]
) -> None:
    """Test embedding generation for sample documents.

    Args:
        embeddings: The embedding model to use.
        documents: List of documents to test with.
    """
    print(f"\n{'=' * 70}")
    print("🧪 Test 1: Embedding Generation")
    print(f"{'=' * 70}")

    if not documents:
        print("⚠ No documents to test with")
        return

    # Generate embeddings for first two chunks
    print("Generating embeddings for sample documents...")
    vector_1 = embeddings.embed_query(documents[0].page_content)
    vector_2 = embeddings.embed_query(documents[1].page_content)

    # Verify vector dimensions
    assert len(vector_1) == len(vector_2), "Vector dimensions must match"
    print(f"✓ Generated vectors of length {len(vector_1)}")
    print(f"  First 10 dimensions of vector 1: {vector_1[:10]}")


def test_add_documents(
    vector_store: AlibabaCloudMySQL, documents: List[Document]
) -> List[str]:
    """Test adding documents to the vector store.

    Args:
        vector_store: The vector store instance.
        documents: List of documents to add.

    Returns:
        List of document IDs that were added.
    """
    print(f"\n{'=' * 70}")
    print("🧪 Test 2: Adding Documents to Vector Store")
    print(f"{'=' * 70}")

    print(f"Adding {len(documents)} documents to vector store...")
    print(f"  Table name: {TABLE_NAME}")
    print(f"  Distance strategy: {DISTANCE_STRATEGY}")

    ids = vector_store.add_documents(documents=documents)
    print(f"✓ Successfully added {len(ids)} documents")
    print(f"  First 3 document IDs: {ids[:3]}")

    return ids


def test_similarity_search(
    vector_store: AlibabaCloudMySQL, query: str, k: int = 5
) -> List[Document]:
    """Test similarity search functionality.

    Args:
        vector_store: The vector store instance.
        query: The search query.
        k: Number of results to return.

    Returns:
        List of Document objects matching the query.
    """
    print(f"\n{'=' * 70}")
    print("🧪 Test 3: Similarity Search")
    print(f"{'=' * 70}")

    print(f"Query: '{query}'")
    print(f"Returning top {k} results...")

    results = vector_store.similarity_search(query, k=k)
    print(f"✓ Found {len(results)} results\n")

    for i, doc in enumerate(results, 1):
        print(f"Result {i}:")
        print(f"  Content: {doc.page_content[:400]}...")
        if doc.metadata:
            print(f"  Metadata: {doc.metadata}")
        print()

    return results


def test_similarity_search_with_score(
    vector_store: AlibabaCloudMySQL, query: str, k: int = 5
) -> List[Tuple[Document, float]]:
    """Test similarity search with distance scores.

    Note: The score is a distance metric that varies inversely with similarity.
    Lower scores indicate higher similarity.

    Args:
        vector_store: The vector store instance.
        query: The search query.
        k: Number of results to return.

    Returns:
        List of tuples containing (Document, score) pairs.
    """
    print(f"\n{'=' * 70}")
    print("🧪 Test 4: Similarity Search with Scores")
    print(f"{'=' * 70}")

    print(f"Query: '{query}'")
    print(f"Returning top {k} results with distance scores...")
    print("Note: Lower scores indicate higher similarity\n")

    results = vector_store.similarity_search_with_score(query, k=k)
    print(f"✓ Found {len(results)} results\n")

    for i, (doc, score) in enumerate(results, 1):
        print(f"Result {i} (Score: {score:.4f}):")
        print(f"  Content: {doc.page_content[:400]}...")
        if doc.metadata:
            print(f"  Metadata: {doc.metadata}")
        print()

    return results


def test_search_by_vector(
    vector_store: AlibabaCloudMySQL,
    embeddings: Embeddings,
    query: str,
) -> List[Document]:
    """Test searching by pre-computed embedding vector.

    Args:
        vector_store: The vector store instance.
        embeddings: The embedding model to generate query vector.
        query: The search query to convert to embedding.

    Returns:
        List of Document objects matching the query vector.
    """
    print(f"\n{'=' * 70}")
    print("🧪 Test 5: Search by Vector")
    print(f"{'=' * 70}")

    print(f"Query: '{query}'")
    print("Converting query to embedding vector...")

    embedding = embeddings.embed_query(query)
    print(f"✓ Generated embedding vector of length {len(embedding)}")

    print("Searching vector store with pre-computed embedding...")
    results = vector_store.similarity_search_by_vector(embedding)
    print(f"✓ Found {len(results)} results\n")

    for i, doc in enumerate(results, 1):
        print(f"Result {i}:")
        print(f"  Content: {doc.page_content[:400]}...")
        if doc.metadata:
            print(f"  Metadata: {doc.metadata}")
        print()

    return results


def test_retriever_batch(vector_store: AlibabaCloudMySQL, queries: List[str]) -> None:
    """Test batch retrieval using retriever interface.

    Args:
        vector_store: The vector store instance.
        queries: List of search queries.
    """
    print(f"\n{'=' * 70}")
    print("🧪 Test 6: Batch Retrieval with Retriever")
    print(f"{'=' * 70}")

    print("Creating retriever with search_type='similarity', k=1...")
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 1},
    )

    print(f"Performing batch retrieval for {len(queries)} queries...")
    print(f"  Queries: {queries}\n")

    results = retriever.batch(queries)
    print(f"✓ Retrieved results for {len(results)} queries\n")

    for i, (query, docs) in enumerate(zip(queries, results), 1):
        print(f"Query {i}: '{query}'")
        if docs:
            print(f"  Result: {docs[0].page_content[:200]}...")
            if docs[0].metadata:
                print(f"  Metadata: {docs[0].metadata}")
        else:
            print("  No results found")
        print()


async def test_async_similarity_search(
    vector_store: AlibabaCloudMySQL, query: str, k: int = 3
) -> List[Document]:
    """Test async similarity search to verify it's truly asynchronous and non-blocking.

    This test verifies that async calls are truly asynchronous by:
    1. Executing multiple async queries concurrently
    2. Measuring total time vs sequential execution time
    3. Confirming that concurrent execution is faster than sequential

    Args:
        vector_store: The vector store instance.
        query: The search query.
        k: Number of results to return.

    Returns:
        List of Document objects matching the query.
    """
    print(f"\n{'=' * 70}")
    print("🧪 Test 7: Async Similarity Search (Non-blocking Verification)")
    print(f"{'=' * 70}")

    print(f"Query: '{query}'")
    print(f"Returning top {k} results...\n")

    # Test 1: Single async search
    print("1. Testing single async search...")
    start_time = time.time()
    async_results = await vector_store.asimilarity_search(query, k=k)
    single_time = time.time() - start_time
    print(f"✓ Single async search completed in {single_time:.4f} seconds")
    print(f"  Found {len(async_results)} results\n")

    # Test 2: Multiple concurrent async searches to verify non-blocking
    print("2. Testing concurrent async searches (verifying non-blocking)...")
    queries = [
        "When was Nike incorporated?",
        "What was Nike's revenue in 2023?",
        "How many distribution centers does Nike have?",
    ]

    # Sequential execution (blocking)
    print("   Sequential execution (blocking):")
    start_time = time.time()
    sequential_results = []
    for q in queries:
        result = await vector_store.asimilarity_search(q, k=k)
        sequential_results.append(result)
    sequential_time = time.time() - start_time
    print(f"   ✓ Sequential: {sequential_time:.4f} seconds\n")

    # Concurrent execution (non-blocking)
    print("   Concurrent execution (non-blocking):")
    start_time = time.time()
    await asyncio.gather(*[vector_store.asimilarity_search(q, k=k) for q in queries])
    concurrent_time = time.time() - start_time
    print(f"   ✓ Concurrent: {concurrent_time:.4f} seconds\n")

    # Verify non-blocking behavior
    print("3. Verification Results:")
    if concurrent_time < sequential_time:
        speedup = sequential_time / concurrent_time
        print("   ✓ Async is truly non-blocking!")
        print(f"   ✓ Concurrent execution is {speedup:.2f}x faster than sequential")
        print("   ✓ This confirms async calls don't block each other\n")
    else:
        print(
            f"   ⚠ Note: Concurrent time ({concurrent_time:.4f}s) >= "
            f"Sequential time ({sequential_time:.4f}s)"
        )
        print("   This may indicate the operations are CPU-bound or very fast\n")

    # Display results from the original query
    print("4. Results from original query:")
    for i, doc in enumerate(async_results, 1):
        print(f"Result {i}:")
        print(f"  Content: {doc.page_content[:400]}...")
        if doc.metadata:
            print(f"  Metadata: {doc.metadata}")
        print()

    return async_results


def cleanup_vector_store(vector_store: AlibabaCloudMySQL) -> None:
    """Clean up all data from the vector store table.

    Args:
        vector_store: The vector store instance to clean up.
    """
    print(f"\n{'=' * 70}")
    print("🧹 Cleaning Up Vector Store")
    print(f"{'=' * 70}")

    try:
        # Check if connection pool is still available
        if not hasattr(vector_store, "_pool") or vector_store._pool is None:
            print("⚠ Connection pool already closed, skipping data cleanup")
            print("  (This is normal if async connections were used)")
            return

        # Get count before cleanup
        count_before = vector_store.count()
        print(f"Records in table before cleanup: {count_before}")

        if count_before > 0:
            # Clear all data from the table
            print("Clearing all data from table...")
            vector_store.clear()

            # Verify cleanup
            count_after = vector_store.count()
            print(f"✓ Successfully cleared {count_before} records")
            print(f"Records in table after cleanup: {count_after}")
        else:
            print("Table is already empty, no cleanup needed")

        # Close connection pool
        print("Closing connection pool...")
        vector_store.close()
        print("✓ Connection pool closed")

    except Exception as e:
        print(f"⚠ Warning: Failed to cleanup vector store: {e}")
        # Try to close connection pool even if cleanup failed
        try:
            if hasattr(vector_store, "_pool") and vector_store._pool is not None:
                vector_store.close()
        except Exception:
            pass


# =============================================================================
# Main Execution
# =============================================================================


def main() -> None:
    """Main function to run all semantic search tests."""
    print("\n" + "=" * 70)
    print("🚀 Semantic Search Demo - AlibabaCloud MySQL Vector Store")
    print("=" * 70)

    vector_store = None

    try:
        # Step 1: Initialize embeddings
        print("\n📦 Step 1: Initializing Embeddings")
        embeddings = get_dashscope_embeddings()
        if not embeddings:
            print("❌ Failed to initialize embeddings. Exiting.")
            return
        print("✓ Embeddings initialized successfully")

        # Step 2: Load and split documents
        print("\n📦 Step 2: Loading and Splitting Documents")
        try:
            all_splits = load_and_split_documents(PDF_FILE_PATH)
        except Exception as e:
            print(f"❌ Failed to load documents: {e}")
            return

        # Step 3: Test embedding generation (Test 1)
        print("\n📦 Step 3: Testing Embedding Generation")
        test_embedding_generation(embeddings, all_splits)

        # Step 4: Create vector store
        print("\n📦 Step 4: Creating Vector Store")
        try:
            vector_store = create_vector_store(embeddings)
            print("✓ Vector store created successfully")
        except Exception as e:
            print(f"❌ Failed to create vector store: {e}")
            return

        # Step 5: Add documents to vector store (Test 2)
        print("\n📦 Step 5: Adding Documents to Vector Store")
        try:
            test_add_documents(vector_store, all_splits)
        except Exception as e:
            print(f"❌ Failed to add documents: {e}")
            return

        # Step 6: Run search tests (Test 3-6)
        print("\n📦 Step 6: Running Search Tests")

        # Test 3: Basic similarity search
        test_similarity_search(
            vector_store, "How many distribution centers does Nike have in the US?", k=3
        )

        # Test 4: Similarity search with scores
        test_similarity_search_with_score(
            vector_store, "What was Nike's revenue in 2023?", k=3
        )

        # Test 5: Search by vector
        test_search_by_vector(
            vector_store, embeddings, "How were Nike's margins impacted in 2023?"
        )

        # Test 6: Batch retrieval
        test_retriever_batch(
            vector_store,
            [
                "How many distribution centers does Nike have in the US?",
                "When was Nike incorporated?",
            ],
        )

        # Test 7: Async similarity search (performance test)
        print("\n📦 Step 7: Testing Async Search Performance")
        try:
            # Create a wrapper function to ensure async connections are closed
            async def run_async_test():
                try:
                    return await test_async_similarity_search(
                        vector_store, "When was Nike incorporated?", k=3
                    )
                finally:
                    # Close async connection pool, but keep sync pool for cleanup
                    try:
                        if (
                            hasattr(vector_store, "_async_pool")
                            and vector_store._async_pool
                        ):
                            vector_store._async_pool.close()
                            await vector_store._async_pool.wait_closed()
                            vector_store._async_pool = None
                            # Brief delay to allow connections to be released
                            await asyncio.sleep(0.1)
                    except Exception as e:
                        print(f"⚠ Warning: Error closing async connections: {e}")

            async_results = asyncio.run(run_async_test())
            # Display first result as requested
            if async_results:
                print("\nFirst result from async search:")
                print(f"  {async_results[0]}")
        except Exception as e:
            print(f"❌ Failed to run async search test: {e}")

        print("\n" + "=" * 70)
        print("✅ All tests completed successfully!")
        print("=" * 70)

    finally:
        # Cleanup - always execute cleanup regardless of test results
        if vector_store is not None:
            cleanup_vector_store(vector_store)
        print()


if __name__ == "__main__":
    main()
