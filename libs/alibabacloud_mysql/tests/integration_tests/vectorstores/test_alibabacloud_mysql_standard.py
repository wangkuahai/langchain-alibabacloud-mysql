"""Standard integration tests for AlibabaCloudMySQL using VectorStoreIntegrationTests.

These tests use the langchain-tests standard test suite to ensure compatibility
with the LangChain ecosystem.

To run these tests, create a .env file in the project root with:
    ALIBABACLOUD_MYSQL_HOST=your-host
    ALIBABACLOUD_MYSQL_PORT=3306
    ALIBABACLOUD_MYSQL_USER=your-user
    ALIBABACLOUD_MYSQL_PASSWORD=your-password
    ALIBABACLOUD_MYSQL_DATABASE=your-database

Or set the environment variables directly.
"""

import asyncio
import os
import time
import uuid
from pathlib import Path
from typing import Generator

import pytest
from langchain_core.vectorstores import VectorStore
from langchain_tests.integration_tests import VectorStoreIntegrationTests

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    # Try to find .env file in project root or libs directory
    env_paths = [
        # project root
        Path(__file__).parent.parent.parent.parent.parent.parent / ".env",
        Path(__file__).parent.parent.parent.parent / ".env",  # libs/alibabacloud_mysql
        Path(__file__).parent.parent.parent / ".env",  # tests
        Path(__file__).parent.parent / ".env",  # integration_tests
        Path.cwd() / ".env",  # current working directory
    ]
    for env_path in env_paths:
        if env_path.exists():
            load_dotenv(env_path)
            break
except ImportError:
    pass  # python-dotenv not installed, use system environment variables


def get_env_var(name: str, default: str = "") -> str:
    """Get environment variable or return default."""
    return os.environ.get(name, default)


# Skip all tests if MySQL connection info not provided
pytestmark = pytest.mark.skipif(
    not get_env_var("ALIBABACLOUD_MYSQL_HOST"),
    reason="AlibabaCloud MySQL connection info not provided",
)


class TestAlibabaCloudMySQLStandard(VectorStoreIntegrationTests):
    """Standard integration tests for AlibabaCloudMySQL.

    Uses VectorStoreIntegrationTests.

    This test class inherits from VectorStoreIntegrationTests, which provides
    a comprehensive set of tests for vector store implementations including:

    - Basic CRUD operations (add, search, delete)
    - Document management (add_documents, get_by_ids)
    - Idempotency tests
    - Async operations
    - Edge cases (empty stores, missing documents, etc.)

    The fixture ensures each test starts with an empty vector store and
    cleans up after each test.
    """

    @pytest.fixture()
    def vectorstore(self) -> Generator[VectorStore, None, None]:  # type: ignore[override]
        """Get an empty vectorstore for unit tests.

        This fixture:
        1. Creates a new AlibabaCloudMySQL instance with a unique table name
        2. Ensures the table is empty (drops it if it exists)
        3. Yields the vectorstore for testing
        4. Cleans up by dropping the table after the test

        Returns:
            An empty AlibabaCloudMySQL vectorstore instance.
        """
        from langchain_alibabacloud_mysql import AlibabaCloudMySQL

        # Generate a unique table name for this test
        table_name = f"test_standard_{uuid.uuid4().hex[:8]}"

        store = AlibabaCloudMySQL(
            host=get_env_var("ALIBABACLOUD_MYSQL_HOST", "localhost"),
            port=int(get_env_var("ALIBABACLOUD_MYSQL_PORT", "3306")),
            user=get_env_var("ALIBABACLOUD_MYSQL_USER", "root"),
            password=get_env_var("ALIBABACLOUD_MYSQL_PASSWORD", ""),
            database=get_env_var("ALIBABACLOUD_MYSQL_DATABASE", "test"),
            embedding=self.get_embeddings(),
            table_name=table_name,
            pre_delete_table=True,  # Ensure table is dropped before creation
            pool_size=2,  # Smaller pool to reduce connection overhead
        )

        try:
            yield store
        finally:
            # Cleanup: drop the table and close connections
            try:
                store.drop_table()
            except Exception:
                pass  # Table might not exist or already dropped

            # Close async connections if they exist
            # Reference test_alibabacloud_mysql.py for cleanup pattern
            if hasattr(store, "_async_pool") and store._async_pool:
                try:
                    # Try to close async pool if event loop is available
                    try:
                        loop = asyncio.get_running_loop()

                        # Event loop is running, use aclose() method
                        # Create a task to close the pool
                        async def close_async():
                            try:
                                await store.aclose()
                            except Exception:
                                pass

                        asyncio.create_task(close_async())
                    except RuntimeError:
                        # No running event loop, try to get existing loop
                        try:
                            loop = asyncio.get_event_loop()
                            if not loop.is_closed():
                                # Use aclose() method to properly close async pool
                                async def close_async():
                                    try:
                                        await store.aclose()
                                    except Exception:
                                        pass

                                loop.run_until_complete(close_async())
                            else:
                                # Loop is closed, set to None to avoid __del__ errors
                                store._async_pool = None
                        except RuntimeError:
                            # No event loop available, set to None
                            # Prevents __del__ from trying to close after loop closes
                            store._async_pool = None
                except Exception:
                    # If anything goes wrong, set to None to prevent __del__ errors
                    try:
                        store._async_pool = None
                    except Exception:
                        pass

            # Close sync connections
            store.close()
            # Brief delay to allow connections to be released
            time.sleep(0.1)

    @property
    def has_get_by_ids(self) -> bool:
        """Whether the VectorStore supports get_by_ids.

        AlibabaCloudMySQL supports get_by_ids, so we return True.
        """
        return True

    @property
    def has_sync(self) -> bool:
        """Whether sync tests are supported.

        AlibabaCloudMySQL supports sync operations.
        """
        return True

    @property
    def has_async(self) -> bool:
        """Whether async tests are supported.

        AlibabaCloudMySQL supports async operations.
        """
        return True
