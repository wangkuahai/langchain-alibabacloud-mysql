"""RAG Agent demo with automatic .env file loading.

This script demonstrates RAG (Retrieval-Augmented Generation) capabilities.
"""

import os
from pathlib import Path

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
TABLE_NAME = os.getenv("VECTOR_STORE_TABLE_NAME", "langchain_vectors_rag_agent")
DISTANCE_STRATEGY = os.getenv("DISTANCE_STRATEGY", "cosine")  # or "euclidean"
HNSW_M = int(os.getenv("HNSW_M", "6"))  # HNSW index M parameter (3-200)

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


# =============================================================================
# Main Function
# =============================================================================


def main() -> None:
    """Main function for RAG agent."""
    print("\n" + "=" * 70)
    print("🚀 RAG Agent Demo")
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

        # Step 3: Load documents
        print("\n📦 Step 3: Loading Documents")
        try:
            import bs4
            from langchain_community.document_loaders import WebBaseLoader

            # Only keep post title, headers, and content from the full HTML.
            bs4_strainer = bs4.SoupStrainer(
                class_=("post-title", "post-header", "post-content")
            )
            loader = WebBaseLoader(
                web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
                bs_kwargs={"parse_only": bs4_strainer},
            )
            docs = loader.load()

            assert len(docs) == 1
            print("✓ Loaded document")
            print(f"  Total characters: {len(docs[0].page_content)}")
            print(f"  Preview: {docs[0].page_content[:500]}...")
        except Exception as e:
            print(f"❌ Failed to load documents: {e}")
            return

        # Step 4: Split documents
        print("\n📦 Step 4: Splitting Documents")
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                add_start_index=True,  # track index in original document
            )
            all_splits = text_splitter.split_documents(docs)

            print(f"✓ Split document into {len(all_splits)} sub-documents")
            print(f"  Chunk size: {CHUNK_SIZE}")
            print(f"  Chunk overlap: {CHUNK_OVERLAP}")
        except Exception as e:
            print(f"❌ Failed to split documents: {e}")
            return

        # Step 5: Add documents to vector store
        print("\n📦 Step 5: Adding Documents to Vector Store")
        try:
            document_ids = vector_store.add_documents(documents=all_splits)
            print(f"✓ Successfully added {len(document_ids)} documents")
            print(f"  First 3 document IDs: {document_ids[:3]}")
        except Exception as e:
            print(f"❌ Failed to add documents: {e}")
            return

        # Step 6: Initialize chat LLM
        print("\n📦 Step 6: Initializing Chat LLM")
        try:
            from langchain_community.chat_models.tongyi import ChatTongyi

            chatLLM = ChatTongyi(
                streaming=True,
            )
            print("✓ Chat LLM initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize chat LLM: {e}")
            return

        # Step 7: Create retrieval tool and agent
        print("\n📦 Step 7: Creating Retrieval Tool and Agent")
        try:
            from langchain.agents import create_agent
            from langchain.tools import tool

            @tool(response_format="content_and_artifact")
            def retrieve_context(query: str):
                """Retrieve information to help answer a query."""
                retrieved_docs = vector_store.similarity_search(query, k=2)
                serialized = "\n\n".join(
                    (f"Source: {doc.metadata}\nContent: {doc.page_content}")
                    for doc in retrieved_docs
                )
                return serialized, retrieved_docs

            tools = [retrieve_context]
            # If desired, specify custom instructions
            prompt = (
                "You have access to a tool that retrieves context from a blog post. "
                "Use the tool to help answer user queries."
            )
            agent = create_agent(chatLLM, tools, system_prompt=prompt)
            print("✓ Retrieval tool and agent created successfully")
        except Exception as e:
            print(f"❌ Failed to create tool and agent: {e}")
            return

        # Step 8: Run RAG query
        print("\n📦 Step 8: Running RAG Query")
        print("=" * 70)
        query = (
            "What is the standard method for Task Decomposition?\n\n"
            "Once you get the answer, look up common extensions of that method."
        )
        print(f"Query: {query}\n")

        try:
            for event in agent.stream(
                {"messages": [{"role": "user", "content": query}]},
                stream_mode="values",
            ):
                event["messages"][-1].pretty_print()
        except Exception as e:
            print(f"❌ Failed to run query: {e}")
            return

        # Step 9: Create agent with dynamic_prompt middleware
        print("\n📦 Step 9: Creating Agent with Dynamic Prompt Middleware")
        try:
            from langchain.agents import create_agent
            from langchain.agents.middleware import ModelRequest, dynamic_prompt

            @dynamic_prompt
            def prompt_with_context(request: ModelRequest) -> str:
                """Inject context into state messages."""
                last_query = request.state["messages"][-1].text
                retrieved_docs = vector_store.similarity_search(last_query)

                docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

                system_message = (
                    "You are a helpful assistant. "
                    "Use the following context in your response:"
                    f"\n\n{docs_content}"
                )

                return system_message  # noqa: E501

            agent_with_middleware = create_agent(
                chatLLM, tools=[], middleware=[prompt_with_context]
            )
            print("✓ Agent with dynamic prompt middleware created successfully")
        except Exception as e:
            print(f"❌ Failed to create agent with middleware: {e}")
            return

        # Step 10: Run RAG query with dynamic prompt
        print("\n📦 Step 10: Running RAG Query with Dynamic Prompt")
        print("=" * 70)
        query = "What is task decomposition?"
        print(f"Query: {query}\n")

        try:
            for step in agent_with_middleware.stream(
                {"messages": [{"role": "user", "content": query}]},
                stream_mode="values",
            ):
                step["messages"][-1].pretty_print()
        except Exception as e:
            print(f"❌ Failed to run query: {e}")
            return

        print("\n" + "=" * 70)
        print("✅ RAG Agent setup and query completed successfully!")
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
