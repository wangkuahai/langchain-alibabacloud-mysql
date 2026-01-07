"""Filter query demo with automatic .env file loading.

This script demonstrates metadata filtering capabilities using:
- AlibabaCloud MySQL Vector Store for storing and retrieving document embeddings
- DashScope embeddings for generating vector representations
- Metadata filtering for hybrid search (vector similarity + metadata conditions)
"""

import os
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from langchain_alibabacloud_mysql import AlibabaCloudMySQL

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
# Configuration from Environment Variables
# =============================================================================

# RAG Configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# Vector store configuration
TABLE_NAME = os.getenv("VECTOR_STORE_TABLE_NAME", "langchain_vectors_filter_query")
DISTANCE_STRATEGY = os.getenv("DISTANCE_STRATEGY", "cosine")  # or "euclidean"
HNSW_M = int(os.getenv("HNSW_M", "6"))  # HNSW index M parameter (3-200)

# Document path
DOCUMENT_PATH = os.getenv("DOCUMENT_PATH", "./data/paul_graham_essay.txt")

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


def load_and_split_document(file_path: str) -> list[Document]:
    """Load text document and split it into chunks.

    Args:
        file_path: Path to the text file.

    Returns:
        List of Document objects representing the text chunks.
    """
    print(f"Loading document from: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"✓ Loaded document ({len(text)} characters)")
    print(f"  Preview: {text[:200]}...")

    # Split the text into paragraphs
    paragraphs = text.split("\n\n")
    print(f"✓ Split into {len(paragraphs)} paragraphs")

    # Create documents from paragraphs
    documents = []
    for i, para in enumerate(paragraphs):
        if para.strip():  # Skip empty paragraphs
            # Check if this is a footnote (starts with [number])
            is_footnote = bool(re.search(r"^\s*\[\d+\]\s*", para))

            doc = Document(
                page_content=para.strip(),
                metadata={
                    "paragraph_id": i,
                    "is_footnote": is_footnote,
                    "source": file_path,
                },
            )
            documents.append(doc)

    print(f"✓ Created {len(documents)} documents")
    print(f"  Footnotes: {sum(1 for d in documents if d.metadata.get('is_footnote'))}")
    regular_count = sum(1 for d in documents if not d.metadata.get("is_footnote"))
    print(f"  Regular paragraphs: {regular_count}")

    return documents


def test_simple_similarity_search(
    vector_store: AlibabaCloudMySQL, query: str, k: int = 5
) -> None:
    """Test simple similarity search without filters.

    Args:
        vector_store: The vector store instance.
        query: The search query.
        k: Number of results to return.
    """
    print(f"\n{'=' * 70}")
    print("🧪 Test 1: Simple Similarity Search (No Filter)")
    print(f"{'=' * 70}")

    print(f"Query: '{query}'")
    print(f"Returning top {k} results...\n")

    results = vector_store.similarity_search(query, k=k)
    print(f"✓ Found {len(results)} results\n")

    for i, doc in enumerate(results, 1):
        print(f"Result {i}:")
        print(f"  Content: {doc.page_content[:200]}...")
        print(f"  Metadata: {doc.metadata}")
        print()


def test_metadata_filter_equality(
    vector_store: AlibabaCloudMySQL, query: str, k: int = 5
) -> None:
    """Test similarity search with metadata equality filter.

    Args:
        vector_store: The vector store instance.
        query: The search query.
        k: Number of results to return.
    """
    print(f"\n{'=' * 70}")
    print("🧪 Test 2: Similarity Search with Metadata Filter (Equality)")
    print(f"{'=' * 70}")

    print(f"Query: '{query}'")
    print("Filter: {'is_footnote': True}")
    print(f"Returning top {k} results...\n")

    results = vector_store.similarity_search(query, k=k, filter={"is_footnote": True})
    print(f"✓ Found {len(results)} results\n")

    for i, doc in enumerate(results, 1):
        print(f"Result {i}:")
        print(f"  Content: {doc.page_content[:200]}...")
        print(f"  Metadata: {doc.metadata}")
        # Verify filter worked
        assert doc.metadata.get("is_footnote"), "Filter failed!"
        print()


def test_metadata_filter_operators(vector_store: AlibabaCloudMySQL, query: str) -> None:
    """Test similarity search with metadata filter operators.

    Args:
        vector_store: The vector store instance.
        query: The search query.
    """
    print(f"\n{'=' * 70}")
    print("🧪 Test 3: Similarity Search with Metadata Filter Operators")
    print(f"{'=' * 70}")

    # Test $gt (greater than)
    print("\n3.1. Testing $gt operator (paragraph_id > 10):")
    print(f"Query: '{query}'")
    results = vector_store.similarity_search(
        query, k=5, filter={"paragraph_id": {"$gt": 10}}
    )
    print(f"✓ Found {len(results)} results")
    for doc in results:
        assert doc.metadata.get("paragraph_id", 0) > 10, "Filter failed!"
        print(f"  - Paragraph ID: {doc.metadata.get('paragraph_id')}")

    # Test $lt (less than)
    print("\n3.2. Testing $lt operator (paragraph_id < 5):")
    print(f"Query: '{query}'")
    results = vector_store.similarity_search(
        query, k=5, filter={"paragraph_id": {"$lt": 5}}
    )
    print(f"✓ Found {len(results)} results")
    for doc in results:
        assert doc.metadata.get("paragraph_id", 999) < 5, "Filter failed!"
        print(f"  - Paragraph ID: {doc.metadata.get('paragraph_id')}")

    # Test $gte and $lte (range)
    print("\n3.3. Testing range filter ($gte and $lte):")
    print(f"Query: '{query}'")
    results = vector_store.similarity_search(
        query, k=5, filter={"paragraph_id": {"$gte": 5, "$lte": 10}}
    )
    print(f"✓ Found {len(results)} results")
    for doc in results:
        para_id = doc.metadata.get("paragraph_id", 0)
        assert 5 <= para_id <= 10, "Filter failed!"
        print(f"  - Paragraph ID: {para_id}")


def test_metadata_only_search(vector_store: AlibabaCloudMySQL) -> None:
    """Test metadata-only search (no vector similarity).

    Args:
        vector_store: The vector store instance.
    """
    print(f"\n{'=' * 70}")
    print("🧪 Test 4: Metadata-Only Search (No Vector Similarity)")
    print(f"{'=' * 70}")

    print("Searching for all footnotes (is_footnote = True)...\n")

    results = vector_store.search_by_metadata(filter={"is_footnote": True}, limit=10)
    print(f"✓ Found {len(results)} footnotes\n")

    for i, doc in enumerate(results, 1):
        print(f"Footnote {i}:")
        print(f"  Content: {doc.page_content[:200]}...")
        print(f"  Metadata: {doc.metadata}")
        print()


def test_combined_filters(vector_store: AlibabaCloudMySQL, query: str) -> None:
    """Test similarity search with combined metadata filters.

    Args:
        vector_store: The vector store instance.
        query: The search query.
    """
    print(f"\n{'=' * 70}")
    print("🧪 Test 5: Combined Metadata Filters")
    print(f"{'=' * 70}")

    print(f"Query: '{query}'")
    print("Filter: {'is_footnote': False, 'paragraph_id': {'$gt': 5}}")
    print("(Regular paragraphs with paragraph_id > 5)\n")

    results = vector_store.similarity_search(
        query, k=5, filter={"is_footnote": False, "paragraph_id": {"$gt": 5}}
    )
    print(f"✓ Found {len(results)} results\n")

    for i, doc in enumerate(results, 1):
        print(f"Result {i}:")
        print(f"  Content: {doc.page_content[:200]}...")
        print(f"  Metadata: {doc.metadata}")
        # Verify filters
        assert not doc.metadata.get("is_footnote"), "Filter failed!"
        assert doc.metadata.get("paragraph_id", 0) > 5, "Filter failed!"
        print()


# =============================================================================
# Main Function
# =============================================================================


def main() -> None:
    """Main function for filter query demo."""
    print("\n" + "=" * 70)
    print("🚀 Filter Query Demo - AlibabaCloud MySQL Vector Store")
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

        # Step 2: Create vector store
        print("\n📦 Step 2: Creating Vector Store")
        try:
            vector_store = create_vector_store(embeddings)
            print("✓ Vector store created successfully")
            print(f"  Table name: {TABLE_NAME}")
            print(f"  Distance strategy: {DISTANCE_STRATEGY}")
        except Exception as e:
            print(f"❌ Failed to create vector store: {e}")
            return

        # Step 3: Load and prepare documents
        print("\n📦 Step 3: Loading and Preparing Documents")
        try:
            documents = load_and_split_document(DOCUMENT_PATH)
        except Exception as e:
            print(f"❌ Failed to load documents: {e}")
            return

        # Step 4: Add documents to vector store
        print("\n📦 Step 4: Adding Documents to Vector Store")
        try:
            document_ids = vector_store.add_documents(documents=documents)
            print(f"✓ Successfully added {len(document_ids)} documents")
            print(f"  First 3 document IDs: {document_ids[:3]}")
        except Exception as e:
            print(f"❌ Failed to add documents: {e}")
            return

        # Step 5: Run filter query tests
        print("\n📦 Step 5: Running Filter Query Tests")

        # Test 1: Simple similarity search
        test_simple_similarity_search(vector_store, "programming and computers", k=5)

        # Test 2: Metadata filter (equality)
        test_metadata_filter_equality(vector_store, "programming and computers", k=5)

        # Test 3: Metadata filter operators
        test_metadata_filter_operators(vector_store, "programming and computers")

        # Test 4: Metadata-only search
        test_metadata_only_search(vector_store)

        # Test 5: Combined filters
        test_combined_filters(vector_store, "programming and computers")

        # Step 6: Create agent with filter support
        print("\n📦 Step 6: Creating Agent with Filter Support")
        try:
            from langchain.agents import create_agent
            from langchain.tools import tool
            from langchain_community.chat_models.tongyi import ChatTongyi

            chatLLM = ChatTongyi(
                streaming=True,
            )
            print("✓ Chat LLM initialized successfully")

            @tool(response_format="content_and_artifact")
            def retrieve_with_filter(
                query: str,
                is_footnote: bool = None,
                paragraph_id_min: int = None,
                paragraph_id_max: int = None,
            ) -> str:
                """Retrieve information with optional metadata filters.

                Args:
                    query: The search query.
                    is_footnote: Filter by footnote status
                        (True for footnotes, False for regular paragraphs,
                        None for all).
                    paragraph_id_min: Minimum paragraph ID (inclusive).
                    paragraph_id_max: Maximum paragraph ID (inclusive).

                Returns:
                    Serialized retrieved documents with metadata.
                """
                # Build filter dictionary
                filter_dict = {}

                if is_footnote is not None:
                    filter_dict["is_footnote"] = is_footnote

                if paragraph_id_min is not None or paragraph_id_max is not None:
                    para_filter = {}
                    if paragraph_id_min is not None:
                        para_filter["$gte"] = paragraph_id_min
                    if paragraph_id_max is not None:
                        para_filter["$lte"] = paragraph_id_max
                    if para_filter:
                        filter_dict["paragraph_id"] = para_filter

                # Perform filtered search
                if filter_dict:
                    retrieved_docs = vector_store.similarity_search(
                        query, k=3, filter=filter_dict
                    )
                else:
                    retrieved_docs = vector_store.similarity_search(query, k=3)

                serialized = "\n\n".join(
                    (f"Source: {doc.metadata}\nContent: {doc.page_content}")
                    for doc in retrieved_docs
                )
                return serialized, retrieved_docs

            tools = [retrieve_with_filter]
            prompt = (
                "You have access to a tool that retrieves context "
                "from a document with metadata filtering. "
                "The document contains paragraphs and footnotes. "
                "You can filter by footnote status (is_footnote) "
                "and paragraph ID range. "
                "Use the tool to help answer user queries, "
                "and use filters when the user asks for specific types of content."
            )
            agent = create_agent(chatLLM, tools, system_prompt=prompt)
            print("✓ Agent with filter support created successfully")
        except Exception as e:
            print(f"❌ Failed to create agent: {e}")
            return

        # Step 7: Test agent with filter queries
        print("\n📦 Step 7: Testing Agent with Filter Queries")
        print("=" * 70)

        # Test 1: Query without filter
        print("\n7.1. Testing agent query without filter:")
        query1 = "What did the author say about programming?"
        print(f"Query: {query1}\n")
        try:
            for event in agent.stream(
                {"messages": [{"role": "user", "content": query1}]},
                stream_mode="values",
            ):
                event["messages"][-1].pretty_print()
        except Exception as e:
            print(f"❌ Failed to run query: {e}")

        # Test 2: Query with footnote filter
        print("\n7.2. Testing agent query with footnote filter:")
        query2 = "What did the author say about programming? Only from footnotes."
        print(f"Query: {query2}\n")
        try:
            for event in agent.stream(
                {"messages": [{"role": "user", "content": query2}]},
                stream_mode="values",
            ):
                event["messages"][-1].pretty_print()
        except Exception as e:
            print(f"❌ Failed to run query: {e}")

        # Test 3: Query with paragraph ID range filter
        print("\n7.3. Testing agent query with paragraph ID range filter:")
        query3 = "What did the author say about programming? (first 10 paragraphs)?"
        print(f"Query: {query3}\n")
        try:
            for event in agent.stream(
                {"messages": [{"role": "user", "content": query3}]},
                stream_mode="values",
            ):
                event["messages"][-1].pretty_print()
        except Exception as e:
            print(f"❌ Failed to run query: {e}")

        # Test 4: Query with combined filters
        print("\n7.4. Testing agent query with combined filters:")
        query4 = (
            "Find information about Lisp, but exclude footnotes "
            "and only search in paragraphs 5-20."
        )
        print(f"Query: {query4}\n")
        try:
            for event in agent.stream(
                {"messages": [{"role": "user", "content": query4}]},
                stream_mode="values",
            ):
                event["messages"][-1].pretty_print()
        except Exception as e:
            print(f"❌ Failed to run query: {e}")

        print("\n" + "=" * 70)
        print("✅ All filter query tests completed successfully!")
        print("=" * 70)

    finally:
        # Cleanup - close connection pool
        if vector_store is not None:
            try:
                print("\n🧹 Cleaning up...")
                vector_store.clear()
                vector_store.close()
                print("✓ Connection pool closed")
            except Exception as e:
                print(f"⚠ Warning: Failed to close vector store: {e}")


if __name__ == "__main__":
    main()
