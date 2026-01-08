"""Integration tests for AlibabaCloudMySQL vector store.

These tests require a running AlibabaCloud RDS MySQL instance with vector support.

To run these tests, create a .env file in the project root with:
    ALIBABACLOUD_MYSQL_HOST=your-host
    ALIBABACLOUD_MYSQL_PORT=3306
    ALIBABACLOUD_MYSQL_USER=your-user
    ALIBABACLOUD_MYSQL_PASSWORD=your-password
    ALIBABACLOUD_MYSQL_DATABASE=your-database
    DASHSCOPE_API_KEY=your-dashscope-api-key

Or set the environment variables directly.
"""

import os
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, AsyncGenerator, Generator

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

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

if TYPE_CHECKING:
    from langchain_alibabacloud_mysql import AlibabaCloudMySQL


def get_env_var(name: str, default: str = "") -> str:
    """Get environment variable or return default."""
    return os.environ.get(name, default)


# =============================================================================
# Test Output Helper
# =============================================================================

# Output prefix for alignment with pytest -v -s
_PREFIX = "    "


def _log(msg: str = "", prefix: bool = True, newline_before: bool = False) -> None:
    """Print formatted log message for test output.

    Args:
        msg: Message to print.
        prefix: Whether to add prefix for alignment.
        newline_before: Whether to add a newline before the message.
    """
    if newline_before:
        print()
    if prefix and msg:
        print(f"{_PREFIX}{msg}")
    else:
        print(msg)


def _log_header(title: str) -> None:
    """Print a header line."""
    print()
    print(f"{_PREFIX}{'─' * 56}")
    print(f"{_PREFIX}{title}")
    print(f"{_PREFIX}{'─' * 56}")


def _log_test_name(name: str) -> None:
    """Print the current test name."""
    print()
    print(f"{_PREFIX}▶ Running: {name}")


@pytest.fixture(autouse=True)
def log_test_name(request: pytest.FixtureRequest) -> None:
    """Automatically log the test name before each test runs."""
    _log_test_name(request.node.name)


# Skip all tests if MySQL connection info not provided
pytestmark = pytest.mark.skipif(
    not get_env_var("ALIBABACLOUD_MYSQL_HOST"),
    reason="AlibabaCloud MySQL connection info not provided",
)


# =============================================================================
# Connection and Vector Support Tests (run first to verify environment)
# =============================================================================


@pytest.mark.integration
def test_mysql_connection() -> None:
    """Test MySQL connection - run this first to verify connectivity.

    This test verifies:
    1. Network connectivity to the MySQL server
    2. Authentication credentials are correct
    3. Database exists and is accessible
    """
    try:
        import mysql.connector
    except ImportError:
        pytest.skip("mysql-connector-python not installed")

    host = get_env_var("ALIBABACLOUD_MYSQL_HOST")
    port = int(get_env_var("ALIBABACLOUD_MYSQL_PORT", "3306"))
    user = get_env_var("ALIBABACLOUD_MYSQL_USER")
    password = get_env_var("ALIBABACLOUD_MYSQL_PASSWORD")
    database = get_env_var("ALIBABACLOUD_MYSQL_DATABASE")

    _log_header("Testing MySQL Connection")
    _log(f"Host: {host}")
    _log(f"Port: {port}")
    _log(f"User: {user}")
    _log(f"Database: {database}")

    conn = None
    try:
        conn = mysql.connector.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
            charset="utf8mb4",
            connection_timeout=10,
        )
        assert conn.is_connected(), "Connection should be established"

        cursor = conn.cursor(dictionary=True)

        # Get MySQL version
        cursor.execute("SELECT VERSION() as version")
        result = cursor.fetchone()
        version = result["version"] if result else "Unknown"
        _log("Connected successfully!")
        _log(f"   MySQL Version: {version}")

        # Get database info
        cursor.execute("SELECT DATABASE() as db")
        result = cursor.fetchone()
        current_db = result["db"] if result else "Unknown"
        _log(f"   Current Database: {current_db}")

        cursor.close()

    except mysql.connector.Error as e:
        _log()
        _log("─" * 56)
        _log("❌ MySQL Connection Failed!")
        _log("─" * 56)
        _log(f"Error Code: {e.errno}")
        _log(f"Error Message: {e.msg}")
        _log()
        _log("Troubleshooting:")
        _log("1. Check if RDS instance has public network access enabled")
        _log("2. Verify IP whitelist includes your current IP")
        _log("3. Confirm credentials are correct")
        _log(f"4. Test with: nc -zv {host} {port}")
        _log("─" * 56)
        pytest.fail(f"MySQL connection failed: {e}")
    finally:
        if conn and conn.is_connected():
            conn.close()


@pytest.mark.integration
def test_vector_support() -> None:
    """Test if MySQL server supports vector operations.

    This test verifies:
    1. VEC_FromText() function is available
    2. VEC_DISTANCE_COSINE() function is available
    3. VEC_DISTANCE_EUCLIDEAN() function is available
    4. VECTOR data type is supported

    Requirements:
    - AlibabaCloud RDS MySQL 8.0.36+
    - rds_release_date >= 20251031
    """
    try:
        import mysql.connector
    except ImportError:
        pytest.skip("mysql-connector-python not installed")

    host = get_env_var("ALIBABACLOUD_MYSQL_HOST")
    port = int(get_env_var("ALIBABACLOUD_MYSQL_PORT", "3306"))
    user = get_env_var("ALIBABACLOUD_MYSQL_USER")
    password = get_env_var("ALIBABACLOUD_MYSQL_PASSWORD")
    database = get_env_var("ALIBABACLOUD_MYSQL_DATABASE")

    _log_header("Testing Vector Support")

    conn = None
    try:
        conn = mysql.connector.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
            charset="utf8mb4",
        )
        cursor = conn.cursor(dictionary=True)

        # Test 1: VEC_FromText function
        _log("1. Testing VEC_FromText() function...", newline_before=True)
        try:
            cursor.execute(
                "SELECT VEC_FromText('[1.0, 2.0, 3.0]') IS NOT NULL as supported"
            )
            result = cursor.fetchone()
            if result and result.get("supported"):
                _log("   ✅ VEC_FromText() is available")
            else:
                pytest.fail("VEC_FromText() returned NULL")
        except mysql.connector.Error as e:
            pytest.fail(f"VEC_FromText() not available: {e}")

        # Test 2: VEC_DISTANCE_COSINE function
        _log("2. Testing VEC_DISTANCE_COSINE() function...")
        try:
            cursor.execute("""
                SELECT VEC_DISTANCE_COSINE(
                    VEC_FromText('[1.0, 0.0, 0.0]'),
                    VEC_FromText('[0.0, 1.0, 0.0]')
                ) as distance
            """)
            result = cursor.fetchone()
            distance = result["distance"] if result else None
            _log(f"   ✅ VEC_DISTANCE_COSINE() is available (distance: {distance})")
        except mysql.connector.Error as e:
            pytest.fail(f"VEC_DISTANCE_COSINE() not available: {e}")

        # Test 3: VEC_DISTANCE_EUCLIDEAN function
        _log("3. Testing VEC_DISTANCE_EUCLIDEAN() function...")
        try:
            cursor.execute("""
                SELECT VEC_DISTANCE_EUCLIDEAN(
                    VEC_FromText('[1.0, 0.0, 0.0]'),
                    VEC_FromText('[0.0, 1.0, 0.0]')
                ) as distance
            """)
            result = cursor.fetchone()
            distance = result["distance"] if result else None
            _log(f"   ✅ VEC_DISTANCE_EUCLIDEAN() is available (distance: {distance})")
        except mysql.connector.Error as e:
            pytest.fail(f"VEC_DISTANCE_EUCLIDEAN() not available: {e}")

        # Test 4: VECTOR data type (create temp table)
        _log("4. Testing VECTOR data type...")
        test_table = f"_test_vector_support_{uuid.uuid4().hex[:8]}"
        try:
            cursor.execute(f"""
                CREATE TABLE {test_table} (
                    id INT PRIMARY KEY,
                    embedding VECTOR(3) NOT NULL
                )
            """)
            _log("   ✅ VECTOR data type is supported")

            # Test insert
            cursor.execute(f"""
                INSERT INTO {test_table} (id, embedding)
                VALUES (1, VEC_FromText('[1.0, 2.0, 3.0]'))
            """)
            _log("   ✅ Vector insert works")

            # Cleanup
            cursor.execute(f"DROP TABLE {test_table}")
            conn.commit()

        except mysql.connector.Error as e:
            pytest.fail(f"VECTOR data type not supported: {e}")

        # Test 5: VECTOR INDEX
        _log("5. Testing VECTOR INDEX...")
        test_table = f"_test_vector_idx_{uuid.uuid4().hex[:8]}"
        try:
            cursor.execute(f"""
                CREATE TABLE {test_table} (
                    id INT PRIMARY KEY,
                    embedding VECTOR(3) NOT NULL,
                    VECTOR INDEX (embedding) M=6 DISTANCE=COSINE
                )
            """)
            _log("   ✅ VECTOR INDEX is supported")

            # Cleanup
            cursor.execute(f"DROP TABLE {test_table}")
            conn.commit()

        except mysql.connector.Error as e:
            pytest.fail(f"VECTOR INDEX not supported: {e}")

        _log()
        _log("─" * 56)
        _log("✅ All vector features are supported!")
        _log("─" * 56)

        cursor.close()

    except mysql.connector.Error as e:
        pytest.fail(f"Vector support test failed: {e}")
    finally:
        if conn and conn.is_connected():
            conn.close()


# =============================================================================
# Embedding and VectorStore Tests
# =============================================================================


def get_dashscope_embeddings() -> Embeddings:
    """Get DashScope embeddings model.

    Uses text-embedding-v4 model from Alibaba Cloud DashScope.
    Dimension: 1024 (default) or 3072 (with dimensions parameter)
    """
    try:
        from langchain_community.embeddings import DashScopeEmbeddings
    except ImportError:
        pytest.skip(
            "langchain-community not installed. "
            "Install with: pip install langchain-community dashscope"
        )

    api_key = get_env_var("DASHSCOPE_API_KEY")
    if not api_key:
        pytest.skip("DASHSCOPE_API_KEY environment variable not set")

    return DashScopeEmbeddings(
        model="text-embedding-v4",
        dashscope_api_key=api_key,
    )


@pytest.fixture
def embeddings() -> Embeddings:
    """Create DashScope embeddings."""
    return get_dashscope_embeddings()


@pytest.fixture
def table_name() -> str:
    """Generate a unique table name for testing."""
    return f"test_langchain_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def vectorstore(
    embeddings: Embeddings, table_name: str
) -> Generator["AlibabaCloudMySQL", None, None]:
    """Create a vector store for testing."""
    import time

    from langchain_alibabacloud_mysql import AlibabaCloudMySQL

    store = AlibabaCloudMySQL(
        host=get_env_var("ALIBABACLOUD_MYSQL_HOST", "localhost"),
        port=int(get_env_var("ALIBABACLOUD_MYSQL_PORT", "3306")),
        user=get_env_var("ALIBABACLOUD_MYSQL_USER", "root"),
        password=get_env_var("ALIBABACLOUD_MYSQL_PASSWORD", ""),
        database=get_env_var("ALIBABACLOUD_MYSQL_DATABASE", "test"),
        embedding=embeddings,
        table_name=table_name,
        pre_delete_table=True,
        pool_size=2,  # Smaller pool to reduce connection overhead
    )

    yield store

    # Cleanup
    try:
        store.drop_table()
    except Exception:
        pass
    store.close()
    # Brief delay to allow connections to be released
    time.sleep(0.1)


@pytest.mark.integration
def test_add_texts(vectorstore: "AlibabaCloudMySQL") -> None:
    """Test adding texts to the vector store."""
    texts = ["你好世界", "LangChain是一个很棒的框架", "阿里云数据库"]
    ids = vectorstore.add_texts(texts)

    assert len(ids) == 3
    assert vectorstore.count() == 3


@pytest.mark.integration
def test_add_texts_with_metadata(vectorstore: "AlibabaCloudMySQL") -> None:
    """Test adding texts with metadata."""
    texts = ["机器学习入门", "深度学习进阶"]
    metadatas = [{"source": "book1", "chapter": 1}, {"source": "book2", "chapter": 2}]

    ids = vectorstore.add_texts(texts, metadatas=metadatas)

    assert len(ids) == 2

    # Retrieve and verify
    docs = vectorstore.get_by_ids(ids)
    assert len(docs) == 2
    assert docs[0].metadata.get("source") in ["book1", "book2"]


@pytest.mark.integration
def test_similarity_search(vectorstore: "AlibabaCloudMySQL") -> None:
    """Test similarity search."""
    texts = [
        "苹果是一种水果",
        "香蕉富含钾元素",
        "橙子含有丰富的维生素C",
        "汽车是一种交通工具",
    ]
    vectorstore.add_texts(texts)

    # Search for fruit-related content
    results = vectorstore.similarity_search("水果有哪些", k=3)

    assert len(results) == 3
    assert all(isinstance(doc, Document) for doc in results)
    # Note: With real embeddings, semantic search should work better
    _log(
        f"Search results: {[doc.page_content for doc in results]}",
        newline_before=True,
    )


@pytest.mark.integration
def test_similarity_search_with_score(vectorstore: "AlibabaCloudMySQL") -> None:
    """Test similarity search with scores."""
    texts = [
        "人工智能正在改变世界",
        "机器学习是AI的重要分支",
        "今天天气很好",
    ]
    vectorstore.add_texts(texts)

    results = vectorstore.similarity_search_with_score("AI技术发展", k=3)

    assert len(results) == 3
    _log("Results:", newline_before=True)
    for doc, score in results:
        assert isinstance(doc, Document)
        assert isinstance(score, float)
        _log(f"  Score: {score:.4f}, Content: {doc.page_content}")


@pytest.mark.integration
def test_delete(vectorstore: "AlibabaCloudMySQL") -> None:
    """Test deleting vectors."""
    texts = ["测试文本1", "测试文本2"]
    ids = vectorstore.add_texts(texts)

    assert vectorstore.count() == 2

    vectorstore.delete(ids=[ids[0]])

    assert vectorstore.count() == 1


@pytest.mark.integration
def test_get_by_ids(vectorstore: "AlibabaCloudMySQL") -> None:
    """Test getting documents by IDs."""
    texts = ["文档一", "文档二"]
    ids = vectorstore.add_texts(texts)

    docs = vectorstore.get_by_ids(ids)

    assert len(docs) == 2
    assert set(doc.page_content for doc in docs) == {"文档一", "文档二"}


@pytest.mark.integration
def test_from_texts(embeddings: Embeddings, table_name: str) -> None:
    """Test creating vector store from texts."""
    from langchain_alibabacloud_mysql import AlibabaCloudMySQL

    texts = ["从文本创建1", "从文本创建2"]

    store = AlibabaCloudMySQL.from_texts(
        texts=texts,
        embedding=embeddings,
        host=get_env_var("ALIBABACLOUD_MYSQL_HOST", "localhost"),
        port=int(get_env_var("ALIBABACLOUD_MYSQL_PORT", "3306")),
        user=get_env_var("ALIBABACLOUD_MYSQL_USER", "root"),
        password=get_env_var("ALIBABACLOUD_MYSQL_PASSWORD", ""),
        database=get_env_var("ALIBABACLOUD_MYSQL_DATABASE", "test"),
        table_name=table_name,
        pool_size=2,
    )

    try:
        assert store.count() == 2
    finally:
        store.drop_table()
        store.close()


@pytest.mark.integration
def test_clear(vectorstore: "AlibabaCloudMySQL") -> None:
    """Test clearing the vector store."""
    texts = ["清空测试1", "清空测试2"]
    vectorstore.add_texts(texts)

    assert vectorstore.count() == 2

    vectorstore.clear()

    assert vectorstore.count() == 0


@pytest.mark.integration
def test_metadata_filter(vectorstore: "AlibabaCloudMySQL") -> None:
    """Test similarity search with metadata filter."""
    texts = [
        "Python编程入门",
        "Java开发指南",
        "Python高级技巧",
        "JavaScript前端开发",
    ]
    metadatas = [
        {"language": "python", "level": "beginner"},
        {"language": "java", "level": "intermediate"},
        {"language": "python", "level": "advanced"},
        {"language": "javascript", "level": "intermediate"},
    ]
    vectorstore.add_texts(texts, metadatas=metadatas)

    # Search with filter
    results = vectorstore.similarity_search(
        "编程教程",
        k=4,
        filter={"language": "python"},
    )

    assert len(results) <= 2  # Should only return Python-related docs
    _log("Filtered results:", newline_before=True)
    for doc in results:
        assert doc.metadata.get("language") == "python"
        _log(f"  {doc.page_content}")


# =============================================================================
# Advanced Metadata Filter Tests (New Features)
# =============================================================================


@pytest.mark.integration
def test_filter_with_gt_operator(vectorstore: "AlibabaCloudMySQL") -> None:
    """Test metadata filter with $gt (greater than) operator."""
    texts = ["商品A", "商品B", "商品C", "商品D"]
    metadatas = [
        {"name": "A", "price": 100},
        {"name": "B", "price": 500},
        {"name": "C", "price": 1000},
        {"name": "D", "price": 2000},
    ]
    vectorstore.add_texts(texts, metadatas=metadatas)

    # Search with $gt filter
    results = vectorstore.similarity_search(
        "商品",
        k=4,
        filter={"price": {"$gt": 500}},
    )

    _log("$gt filter results (price > 500):", newline_before=True)
    for doc in results:
        _log(f"  - {doc.page_content}: price={doc.metadata.get('price')}")

    # Should only return items with price > 500
    assert len(results) == 2
    for doc in results:
        assert int(doc.metadata.get("price", 0)) > 500


@pytest.mark.integration
def test_filter_with_lt_operator(vectorstore: "AlibabaCloudMySQL") -> None:
    """Test metadata filter with $lt (less than) operator."""
    texts = ["产品1", "产品2", "产品3"]
    metadatas = [
        {"name": "P1", "score": 30},
        {"name": "P2", "score": 60},
        {"name": "P3", "score": 90},
    ]
    vectorstore.add_texts(texts, metadatas=metadatas)

    results = vectorstore.similarity_search(
        "产品",
        k=3,
        filter={"score": {"$lt": 60}},
    )

    _log("$lt filter results (score < 60):", newline_before=True)
    for doc in results:
        _log(f"  - {doc.page_content}: score={doc.metadata.get('score')}")

    assert len(results) == 1
    assert int(results[0].metadata.get("score", 0)) < 60


@pytest.mark.integration
def test_filter_with_range(vectorstore: "AlibabaCloudMySQL") -> None:
    """Test metadata filter with range ($gte and $lte)."""
    texts = ["低价商品", "中价商品", "高价商品", "超高价商品"]
    metadatas = [
        {"name": "low", "price": 50},
        {"name": "mid", "price": 200},
        {"name": "high", "price": 500},
        {"name": "premium", "price": 1000},
    ]
    vectorstore.add_texts(texts, metadatas=metadatas)

    # Search with range filter: 100 <= price <= 600
    results = vectorstore.similarity_search(
        "商品",
        k=4,
        filter={"price": {"$gte": 100, "$lte": 600}},
    )

    _log("Range filter results (100 <= price <= 600):", newline_before=True)
    for doc in results:
        _log(f"  - {doc.page_content}: price={doc.metadata.get('price')}")

    assert len(results) == 2
    for doc in results:
        price = int(doc.metadata.get("price", 0))
        assert 100 <= price <= 600


@pytest.mark.integration
def test_filter_with_eq_operator(vectorstore: "AlibabaCloudMySQL") -> None:
    """Test metadata filter with explicit $eq operator."""
    texts = ["文档A", "文档B", "文档C"]
    metadatas = [
        {"type": "report"},
        {"type": "memo"},
        {"type": "report"},
    ]
    vectorstore.add_texts(texts, metadatas=metadatas)

    results = vectorstore.similarity_search(
        "文档",
        k=3,
        filter={"type": {"$eq": "report"}},
    )

    _log("$eq filter results (type == 'report'):", newline_before=True)
    for doc in results:
        _log(f"  - {doc.page_content}: type={doc.metadata.get('type')}")

    assert len(results) == 2
    for doc in results:
        assert doc.metadata.get("type") == "report"


@pytest.mark.integration
def test_filter_with_like_operator(vectorstore: "AlibabaCloudMySQL") -> None:
    """Test metadata filter with $like operator for pattern matching."""
    texts = ["iPhone 15 Pro", "iPhone 14", "Samsung Galaxy", "Pixel Pro"]
    metadatas = [
        {"name": "iPhone 15 Pro"},
        {"name": "iPhone 14"},
        {"name": "Samsung Galaxy S24"},
        {"name": "Google Pixel 8 Pro"},
    ]
    vectorstore.add_texts(texts, metadatas=metadatas)

    # Search for names containing "Pro"
    results = vectorstore.similarity_search(
        "手机",
        k=4,
        filter={"name": {"$like": "%Pro%"}},
    )

    _log("$like filter results (name LIKE '%Pro%'):", newline_before=True)
    for doc in results:
        _log(f"  - {doc.page_content}: name={doc.metadata.get('name')}")

    assert len(results) == 2
    for doc in results:
        assert "Pro" in doc.metadata.get("name", "")


@pytest.mark.integration
def test_filter_with_in_operator(vectorstore: "AlibabaCloudMySQL") -> None:
    """Test metadata filter with $in operator."""
    texts = [
        "苹果手机",
        "华为手机",
        "小米手机",
        "联想电脑",
    ]
    metadatas = [
        {"brand": "apple", "category": "phone"},
        {"brand": "huawei", "category": "phone"},
        {"brand": "xiaomi", "category": "phone"},
        {"brand": "lenovo", "category": "computer"},
    ]
    vectorstore.add_texts(texts, metadatas=metadatas)

    # Search with $in filter
    results = vectorstore.similarity_search(
        "电子产品",
        k=4,
        filter={"brand": {"$in": ["apple", "huawei"]}},
    )

    _log("$in filter results (brand in ['apple', 'huawei']):", newline_before=True)
    for doc in results:
        _log(f"  - {doc.page_content}: brand={doc.metadata.get('brand')}")

    assert len(results) == 2
    for doc in results:
        assert doc.metadata.get("brand") in ["apple", "huawei"]


@pytest.mark.integration
def test_filter_with_nin_operator(vectorstore: "AlibabaCloudMySQL") -> None:
    """Test metadata filter with $nin (not in) operator."""
    texts = ["文档A", "文档B", "文档C"]
    metadatas = [
        {"status": "active", "type": "report"},
        {"status": "archived", "type": "report"},
        {"status": "deleted", "type": "memo"},
    ]
    vectorstore.add_texts(texts, metadatas=metadatas)

    # Search with $nin filter - exclude archived and deleted
    results = vectorstore.similarity_search(
        "文档",
        k=3,
        filter={"status": {"$nin": ["archived", "deleted"]}},
    )

    _log(
        "$nin filter results (status not in ['archived', 'deleted']):",
        newline_before=True,
    )
    for doc in results:
        _log(f"  - {doc.page_content}: status={doc.metadata.get('status')}")

    assert len(results) == 1
    assert results[0].metadata.get("status") == "active"


@pytest.mark.integration
def test_filter_with_ne_operator(vectorstore: "AlibabaCloudMySQL") -> None:
    """Test metadata filter with $ne (not equal) operator."""
    texts = ["任务1", "任务2", "任务3"]
    metadatas = [
        {"priority": "high"},
        {"priority": "medium"},
        {"priority": "low"},
    ]
    vectorstore.add_texts(texts, metadatas=metadatas)

    results = vectorstore.similarity_search(
        "任务",
        k=3,
        filter={"priority": {"$ne": "low"}},
    )

    _log("$ne filter results (priority != 'low'):", newline_before=True)
    for doc in results:
        _log(f"  - {doc.page_content}: priority={doc.metadata.get('priority')}")

    assert len(results) == 2
    for doc in results:
        assert doc.metadata.get("priority") != "low"


@pytest.mark.integration
def test_filter_combined_with_multiple_fields(vectorstore: "AlibabaCloudMySQL") -> None:
    """Test metadata filter with multiple fields."""
    texts = [
        "iPhone 15 Pro",
        "iPhone 14",
        "Samsung Galaxy S24",
        "Pixel 8 Pro",
    ]
    metadatas = [
        {"brand": "apple", "price": 9999, "year": 2024},
        {"brand": "apple", "price": 6999, "year": 2023},
        {"brand": "samsung", "price": 7999, "year": 2024},
        {"brand": "google", "price": 5999, "year": 2024},
    ]
    vectorstore.add_texts(texts, metadatas=metadatas)

    # Filter by brand and price range
    results = vectorstore.similarity_search(
        "手机",
        k=4,
        filter={
            "brand": "apple",
            "price": {"$gt": 7000},
        },
    )

    _log(
        "Combined filter results (brand='apple' AND price > 7000):",
        newline_before=True,
    )
    for doc in results:
        _log(
            f"  - {doc.page_content}: "
            f"brand={doc.metadata.get('brand')}, "
            f"price={doc.metadata.get('price')}"
        )

    assert len(results) == 1
    assert results[0].metadata.get("brand") == "apple"
    assert int(results[0].metadata.get("price", 0)) > 7000


# =============================================================================
# search_by_metadata Tests
# =============================================================================


@pytest.mark.integration
def test_search_by_metadata(vectorstore: "AlibabaCloudMySQL") -> None:
    """Test search by metadata without vector similarity."""
    texts = ["文档1", "文档2", "文档3", "文档4"]
    metadatas = [
        {"author": "Alice", "year": 2020},
        {"author": "Bob", "year": 2021},
        {"author": "Alice", "year": 2022},
        {"author": "Charlie", "year": 2023},
    ]
    vectorstore.add_texts(texts, metadatas=metadatas)

    # Search by author
    results = vectorstore.search_by_metadata(
        filter={"author": "Alice"},
        limit=10,
    )

    _log("search_by_metadata results (author='Alice'):", newline_before=True)
    for doc in results:
        _log(f"  - {doc.page_content}: author={doc.metadata.get('author')}")

    assert len(results) == 2
    for doc in results:
        assert doc.metadata.get("author") == "Alice"


@pytest.mark.integration
def test_search_by_metadata_with_operators(vectorstore: "AlibabaCloudMySQL") -> None:
    """Test search by metadata using operators."""
    texts = ["记录A", "记录B", "记录C"]
    metadatas = [
        {"score": 85},
        {"score": 92},
        {"score": 78},
    ]
    vectorstore.add_texts(texts, metadatas=metadatas)

    results = vectorstore.search_by_metadata(
        filter={"score": {"$gte": 80}},
        limit=10,
    )

    _log("search_by_metadata results (score >= 80):", newline_before=True)
    for doc in results:
        _log(f"  - {doc.page_content}: score={doc.metadata.get('score')}")

    assert len(results) == 2
    for doc in results:
        assert int(doc.metadata.get("score", 0)) >= 80


@pytest.mark.integration
def test_search_by_metadata_no_filter(vectorstore: "AlibabaCloudMySQL") -> None:
    """Test search by metadata without filter returns all documents."""
    texts = ["文档1", "文档2", "文档3"]
    metadatas = [
        {"type": "A"},
        {"type": "B"},
        {"type": "C"},
    ]
    vectorstore.add_texts(texts, metadatas=metadatas)

    # Search without filter should return all documents up to limit
    results = vectorstore.search_by_metadata(limit=10)

    _log("search_by_metadata results (no filter):", newline_before=True)
    for doc in results:
        _log(f"  - {doc.page_content}: type={doc.metadata.get('type')}")

    assert len(results) == 3


@pytest.mark.integration
def test_search_by_metadata_with_in_operator(vectorstore: "AlibabaCloudMySQL") -> None:
    """Test search by metadata with $in operator."""
    texts = ["项目A", "项目B", "项目C", "项目D"]
    metadatas = [
        {"status": "active"},
        {"status": "pending"},
        {"status": "completed"},
        {"status": "cancelled"},
    ]
    vectorstore.add_texts(texts, metadatas=metadatas)

    results = vectorstore.search_by_metadata(
        filter={"status": {"$in": ["active", "pending"]}},
        limit=10,
    )

    _log(
        "search_by_metadata results (status in ['active', 'pending']):",
        newline_before=True,
    )
    for doc in results:
        _log(f"  - {doc.page_content}: status={doc.metadata.get('status')}")

    assert len(results) == 2
    for doc in results:
        assert doc.metadata.get("status") in ["active", "pending"]


# =============================================================================
# MMR with Filter Tests
# =============================================================================


@pytest.mark.integration
def test_mmr_with_filter(vectorstore: "AlibabaCloudMySQL") -> None:
    """Test MMR search with metadata filter."""
    texts = [
        "苹果公司是一家科技公司",
        "苹果手机非常流行",
        "苹果电脑性能强劲",
        "三星手机也很受欢迎",
        "华为手机国产品牌",
    ]
    metadatas = [
        {"brand": "apple", "type": "company"},
        {"brand": "apple", "type": "phone"},
        {"brand": "apple", "type": "computer"},
        {"brand": "samsung", "type": "phone"},
        {"brand": "huawei", "type": "phone"},
    ]
    vectorstore.add_texts(texts, metadatas=metadatas)

    # MMR search with filter for apple brand only
    results = vectorstore.max_marginal_relevance_search(
        "苹果产品",
        k=2,
        fetch_k=5,
        filter={"brand": "apple"},
    )

    _log("MMR with filter results (brand='apple'):", newline_before=True)
    for doc in results:
        _log(f"  - {doc.page_content}: brand={doc.metadata.get('brand')}")

    assert len(results) == 2
    for doc in results:
        assert doc.metadata.get("brand") == "apple"


@pytest.mark.integration
def test_mmr_with_filter_2(vectorstore: "AlibabaCloudMySQL") -> None:
    """Test MMR search with metadata filter."""
    texts = [
        "苹果公司是一家科技公司",
        "苹果公司是一家科技公司",
        "苹果公司是一家科技公司",
        "苹果公司是一家科技公司",
        "苹果公司是一家科技公司",
        "苹果手机非常流行",
        "苹果电脑性能强劲",
        "三星手机也很受欢迎",
        "华为手机国产品牌",
    ]
    metadatas = [
        {"brand": "apple", "type": "company"},
        {"brand": "apple", "type": "company"},
        {"brand": "apple", "type": "company"},
        {"brand": "apple", "type": "company"},
        {"brand": "apple", "type": "company"},
        {"brand": "apple", "type": "phone"},
        {"brand": "apple", "type": "computer"},
        {"brand": "samsung", "type": "phone"},
        {"brand": "huawei", "type": "phone"},
    ]
    vectorstore.add_texts(texts, metadatas=metadatas)

    # MMR search with filter for apple brand only
    results = vectorstore.max_marginal_relevance_search(
        "苹果公司",
        k=2,
        fetch_k=9,
        filter={"brand": "apple"},
    )

    _log("MMR with filter results (brand='apple'):", newline_before=True)
    for doc in results:
        _log(f"  - {doc.page_content}: brand={doc.metadata.get('brand')}")

    assert len(results) == 2
    for doc in results:
        assert doc.metadata.get("brand") == "apple"


# =============================================================================
# delete_by_metadata Tests
# =============================================================================


@pytest.mark.integration
def test_delete_by_metadata(vectorstore: "AlibabaCloudMySQL") -> None:
    """Test deleting documents by metadata."""
    texts = ["保留文档", "删除文档1", "删除文档2"]
    metadatas = [
        {"status": "active"},
        {"status": "deleted"},
        {"status": "deleted"},
    ]
    vectorstore.add_texts(texts, metadatas=metadatas)

    assert vectorstore.count() == 3

    # Delete documents with status='deleted'
    deleted_count = vectorstore.delete_by_metadata(filter={"status": "deleted"})

    _log(
        f"Deleted {deleted_count} documents with status='deleted'",
        newline_before=True,
    )

    assert deleted_count == 2
    assert vectorstore.count() == 1

    # Verify remaining document
    remaining = vectorstore.search_by_metadata(limit=10)
    assert len(remaining) == 1
    assert remaining[0].metadata.get("status") == "active"


@pytest.mark.integration
def test_delete_by_metadata_with_operators(vectorstore: "AlibabaCloudMySQL") -> None:
    """Test deleting documents by metadata using operators."""
    texts = ["高价值", "中价值", "低价值1", "低价值2"]
    metadatas = [
        {"value": 1000},
        {"value": 500},
        {"value": 100},
        {"value": 50},
    ]
    vectorstore.add_texts(texts, metadatas=metadatas)

    assert vectorstore.count() == 4

    # Delete low value items (value < 200)
    deleted_count = vectorstore.delete_by_metadata(filter={"value": {"$lt": 200}})

    _log(f"Deleted {deleted_count} documents with value < 200", newline_before=True)

    assert deleted_count == 2
    assert vectorstore.count() == 2


# =============================================================================
# exists() Test
# =============================================================================


@pytest.mark.integration
def test_exists(vectorstore: "AlibabaCloudMySQL") -> None:
    """Test checking if a document exists."""
    texts = ["测试文档"]
    ids = vectorstore.add_texts(texts)

    doc_id = ids[0]

    # Document should exist
    assert vectorstore.exists(doc_id) is True

    # Non-existent ID should return False
    assert vectorstore.exists("non-existent-id-12345") is False

    # After deletion, should return False
    vectorstore.delete(ids=[doc_id])
    assert vectorstore.exists(doc_id) is False


# =============================================================================
# add_documents() Tests
# =============================================================================


@pytest.mark.integration
def test_add_documents(vectorstore: "AlibabaCloudMySQL") -> None:
    """Test adding documents to the vector store."""
    documents = [
        Document(page_content="这是第一个文档", metadata={"source": "test1"}),
        Document(page_content="这是第二个文档", metadata={"source": "test2"}),
        Document(page_content="这是第三个文档", metadata={"source": "test3"}),
    ]

    ids = vectorstore.add_documents(documents)

    assert len(ids) == 3
    assert vectorstore.count() == 3

    # Verify documents can be retrieved
    retrieved = vectorstore.get_by_ids(ids)
    assert len(retrieved) == 3
    contents = {doc.page_content for doc in retrieved}
    assert "这是第一个文档" in contents
    assert "这是第二个文档" in contents
    assert "这是第三个文档" in contents


@pytest.mark.integration
def test_add_documents_with_ids(vectorstore: "AlibabaCloudMySQL") -> None:
    """Test adding documents with custom IDs."""
    documents = [
        Document(page_content="文档A", metadata={"type": "A"}),
        Document(page_content="文档B", metadata={"type": "B"}),
    ]
    custom_ids = ["custom-id-001", "custom-id-002"]

    ids = vectorstore.add_documents(documents, ids=custom_ids)

    assert ids == custom_ids
    assert vectorstore.count() == 2

    # Verify we can retrieve by custom IDs
    retrieved = vectorstore.get_by_ids(custom_ids)
    assert len(retrieved) == 2


@pytest.mark.integration
def test_add_documents_uses_document_id(vectorstore: "AlibabaCloudMySQL") -> None:
    """Test that add_documents uses document.id if set."""
    documents = [
        Document(id="doc-uuid-001", page_content="带ID的文档1", metadata={}),
        Document(id="doc-uuid-002", page_content="带ID的文档2", metadata={}),
    ]

    ids = vectorstore.add_documents(documents)

    assert ids == ["doc-uuid-001", "doc-uuid-002"]

    # Verify retrieval by document IDs
    retrieved = vectorstore.get_by_ids(ids)
    assert len(retrieved) == 2


@pytest.mark.integration
def test_add_documents_searchable(vectorstore: "AlibabaCloudMySQL") -> None:
    """Test that added documents are searchable."""
    documents = [
        Document(page_content="苹果是一种健康的水果", metadata={"category": "fruit"}),
        Document(page_content="汽车需要定期保养", metadata={"category": "vehicle"}),
        Document(page_content="香蕉富含钾元素", metadata={"category": "fruit"}),
    ]

    vectorstore.add_documents(documents)

    # Search for fruit-related content
    results = vectorstore.similarity_search("水果有哪些种类", k=2)

    assert len(results) == 2
    # Both results should be fruit-related
    _log("Results:", newline_before=True)
    for doc in results:
        _log(f"  {doc.page_content}, metadata: {doc.metadata}")


@pytest.mark.integration
def test_add_documents_empty_list(vectorstore: "AlibabaCloudMySQL") -> None:
    """Test adding empty document list."""
    ids = vectorstore.add_documents([])

    assert ids == []
    # Note: count() not called here because empty list doesn't create the table


# =============================================================================
# add_embeddings() Tests
# =============================================================================


@pytest.mark.integration
def test_add_embeddings(
    vectorstore: "AlibabaCloudMySQL", embeddings: Embeddings
) -> None:
    """Test adding pre-computed embeddings to the vector store."""
    texts = ["预计算嵌入测试1", "预计算嵌入测试2", "预计算嵌入测试3"]

    # Compute embeddings manually
    computed_embeddings = embeddings.embed_documents(texts)

    # Create text_embeddings tuples
    text_embeddings = list(zip(texts, computed_embeddings))

    # Add using add_embeddings
    ids = vectorstore.add_embeddings(
        text_embeddings=text_embeddings,
        metadatas=[{"source": "test1"}, {"source": "test2"}, {"source": "test3"}],
    )

    assert len(ids) == 3
    assert vectorstore.count() == 3

    # Verify documents can be retrieved
    retrieved = vectorstore.get_by_ids(ids)
    assert len(retrieved) == 3


@pytest.mark.integration
def test_add_embeddings_with_custom_ids(
    vectorstore: "AlibabaCloudMySQL", embeddings: Embeddings
) -> None:
    """Test adding pre-computed embeddings with custom IDs."""
    texts = ["自定义ID嵌入1", "自定义ID嵌入2"]
    computed_embeddings = embeddings.embed_documents(texts)
    text_embeddings = list(zip(texts, computed_embeddings))

    custom_ids = ["embed-custom-001", "embed-custom-002"]

    ids = vectorstore.add_embeddings(
        text_embeddings=text_embeddings,
        ids=custom_ids,
    )

    assert ids == custom_ids

    # Verify retrieval
    retrieved = vectorstore.get_by_ids(custom_ids)
    assert len(retrieved) == 2
    assert {doc.page_content for doc in retrieved} == {"自定义ID嵌入1", "自定义ID嵌入2"}


@pytest.mark.integration
def test_add_embeddings_searchable(
    vectorstore: "AlibabaCloudMySQL", embeddings: Embeddings
) -> None:
    """Test that embeddings added via add_embeddings are searchable."""
    texts = [
        "机器学习是人工智能的一个分支",
        "深度学习使用神经网络",
        "自然语言处理研究文本理解",
        "今天天气晴朗适合出门",
    ]
    computed_embeddings = embeddings.embed_documents(texts)
    text_embeddings = list(zip(texts, computed_embeddings))

    metadatas = [
        {"topic": "AI"},
        {"topic": "AI"},
        {"topic": "AI"},
        {"topic": "weather"},
    ]

    vectorstore.add_embeddings(text_embeddings=text_embeddings, metadatas=metadatas)

    # Search for AI-related content
    results = vectorstore.similarity_search("人工智能技术", k=3)

    assert len(results) == 3
    # Most results should be AI-related
    ai_results = [doc for doc in results if doc.metadata.get("topic") == "AI"]
    assert len(ai_results) >= 2, (
        f"Expected at least 2 AI results, got {len(ai_results)}"
    )


@pytest.mark.integration
def test_add_embeddings_empty_list(vectorstore: "AlibabaCloudMySQL") -> None:
    """Test adding empty embeddings list."""
    ids = vectorstore.add_embeddings(text_embeddings=[])

    assert ids == []
    # Note: count() not called here because empty list doesn't create the table


@pytest.mark.integration
def test_add_embeddings_no_metadata(
    vectorstore: "AlibabaCloudMySQL", embeddings: Embeddings
) -> None:
    """Test adding embeddings without metadata."""
    texts = ["无元数据的文本1", "无元数据的文本2"]
    computed_embeddings = embeddings.embed_documents(texts)
    text_embeddings = list(zip(texts, computed_embeddings))

    ids = vectorstore.add_embeddings(text_embeddings=text_embeddings)

    assert len(ids) == 2

    # Verify retrieval - metadata should be empty dict
    retrieved = vectorstore.get_by_ids(ids)
    assert len(retrieved) == 2
    for doc in retrieved:
        assert (
            doc.metadata == {} or doc.metadata is None or doc.metadata == {"id": doc.id}
        )


@pytest.mark.integration
def test_add_embeddings_vs_add_texts_equivalence(
    vectorstore: "AlibabaCloudMySQL", embeddings: Embeddings, table_name: str
) -> None:
    """Test that add_embeddings produces equivalent results to add_texts."""
    from langchain_alibabacloud_mysql import AlibabaCloudMySQL

    # Create a second vectorstore for comparison
    vectorstore2 = AlibabaCloudMySQL(
        host=get_env_var("ALIBABACLOUD_MYSQL_HOST", "localhost"),
        port=int(get_env_var("ALIBABACLOUD_MYSQL_PORT", "3306")),
        user=get_env_var("ALIBABACLOUD_MYSQL_USER", "root"),
        password=get_env_var("ALIBABACLOUD_MYSQL_PASSWORD", ""),
        database=get_env_var("ALIBABACLOUD_MYSQL_DATABASE", "test"),
        embedding=embeddings,
        table_name=f"{table_name}_compare",
        pre_delete_table=True,
        pool_size=2,  # Reduce pool size to avoid connection exhaustion
    )

    try:
        texts = ["比较测试文本1", "比较测试文本2"]
        metadatas = [{"key": "value1"}, {"key": "value2"}]

        # Add via add_texts to vectorstore1
        ids1 = vectorstore.add_texts(texts, metadatas=metadatas)

        # Add via add_embeddings to vectorstore2
        computed_embeddings = embeddings.embed_documents(texts)
        text_embeddings = list(zip(texts, computed_embeddings))
        ids2 = vectorstore2.add_embeddings(
            text_embeddings=text_embeddings, metadatas=metadatas
        )

        assert len(ids1) == len(ids2) == 2

        # Both should produce the same search results
        query = "测试文本"
        results1 = vectorstore.similarity_search_with_score(query, k=2)
        results2 = vectorstore2.similarity_search_with_score(query, k=2)

        # Scores should be very similar (allow small floating point differences)
        for (doc1, score1), (doc2, score2) in zip(results1, results2):
            assert doc1.page_content == doc2.page_content
            assert abs(score1 - score2) < 0.001, f"Scores differ: {score1} vs {score2}"

    finally:
        vectorstore2.drop_table()
        vectorstore2.close()


# =============================================================================
# upsert() Tests
# =============================================================================


@pytest.mark.integration
def test_upsert_insert_new_documents(vectorstore: "AlibabaCloudMySQL") -> None:
    """Test upsert inserts new documents."""
    documents = [
        Document(id="upsert-new-1", page_content="新文档1", metadata={"version": 1}),
        Document(id="upsert-new-2", page_content="新文档2", metadata={"version": 1}),
    ]

    ids = vectorstore.upsert(documents)

    assert ids == ["upsert-new-1", "upsert-new-2"]
    assert vectorstore.count() == 2

    # Verify documents can be retrieved
    retrieved = vectorstore.get_by_ids(ids)
    assert len(retrieved) == 2


@pytest.mark.integration
def test_upsert_update_existing_documents(vectorstore: "AlibabaCloudMySQL") -> None:
    """Test upsert updates existing documents."""
    # First, insert documents
    initial_docs = [
        Document(
            id="upsert-update-1",
            page_content="原始内容1",
            metadata={"version": 1},
        ),
        Document(
            id="upsert-update-2",
            page_content="原始内容2",
            metadata={"version": 1},
        ),
    ]
    vectorstore.upsert(initial_docs)

    assert vectorstore.count() == 2

    # Now upsert with updated content
    updated_docs = [
        Document(
            id="upsert-update-1",
            page_content="更新后的内容1",
            metadata={"version": 2},
        ),
        Document(
            id="upsert-update-2",
            page_content="更新后的内容2",
            metadata={"version": 2},
        ),
    ]
    ids = vectorstore.upsert(updated_docs)

    # Count should still be 2 (updated, not duplicated)
    assert vectorstore.count() == 2

    # Verify content was updated
    retrieved = vectorstore.get_by_ids(ids)
    assert len(retrieved) == 2

    contents = {doc.page_content for doc in retrieved}
    assert "更新后的内容1" in contents
    assert "更新后的内容2" in contents
    assert "原始内容1" not in contents

    # Verify metadata was updated
    for doc in retrieved:
        assert doc.metadata.get("version") == 2


@pytest.mark.integration
def test_upsert_mixed_insert_and_update(vectorstore: "AlibabaCloudMySQL") -> None:
    """Test upsert with mix of new and existing documents."""
    # First, insert one document
    initial_docs = [
        Document(
            id="mixed-existing",
            page_content="已存在的文档",
            metadata={"type": "old"},
        ),
    ]
    vectorstore.upsert(initial_docs)
    assert vectorstore.count() == 1

    # Upsert with one existing and one new document
    mixed_docs = [
        Document(
            id="mixed-existing",
            page_content="更新已存在的文档",
            metadata={"type": "updated"},
        ),
        Document(id="mixed-new", page_content="新增的文档", metadata={"type": "new"}),
    ]
    ids = vectorstore.upsert(mixed_docs)

    assert set(ids) == {"mixed-existing", "mixed-new"}
    assert vectorstore.count() == 2

    # Verify both documents
    retrieved = vectorstore.get_by_ids(ids)
    assert len(retrieved) == 2

    for doc in retrieved:
        if doc.id == "mixed-existing":
            assert doc.page_content == "更新已存在的文档"
            assert doc.metadata.get("type") == "updated"
        else:
            assert doc.page_content == "新增的文档"
            assert doc.metadata.get("type") == "new"


@pytest.mark.integration
def test_upsert_generates_ids(vectorstore: "AlibabaCloudMySQL") -> None:
    """Test upsert generates UUIDs when document.id is None."""
    documents = [
        Document(page_content="无ID文档1", metadata={}),
        Document(page_content="无ID文档2", metadata={}),
    ]

    ids = vectorstore.upsert(documents)

    assert len(ids) == 2
    # UUIDs are 36 characters
    assert all(len(id_) == 36 for id_ in ids)
    assert vectorstore.count() == 2


@pytest.mark.integration
def test_upsert_with_explicit_ids(vectorstore: "AlibabaCloudMySQL") -> None:
    """Test upsert with explicitly provided ids."""
    documents = [
        Document(id="ignored-1", page_content="文档1", metadata={}),
        Document(id="ignored-2", page_content="文档2", metadata={}),
    ]

    # Explicit IDs should override document.id
    ids = vectorstore.upsert(documents, ids=["explicit-id-1", "explicit-id-2"])

    assert ids == ["explicit-id-1", "explicit-id-2"]

    # Verify documents can be retrieved by explicit IDs
    retrieved = vectorstore.get_by_ids(["explicit-id-1", "explicit-id-2"])
    assert len(retrieved) == 2


@pytest.mark.integration
def test_upsert_searchable(vectorstore: "AlibabaCloudMySQL") -> None:
    """Test that upserted documents are searchable."""
    documents = [
        Document(
            id="search-1",
            page_content="机器学习是人工智能的分支",
            metadata={"topic": "AI"},
        ),
        Document(
            id="search-2",
            page_content="今天天气很好",
            metadata={"topic": "weather"},
        ),
    ]

    vectorstore.upsert(documents)

    # Search for AI-related content
    results = vectorstore.similarity_search("人工智能技术", k=2)

    assert len(results) == 2
    # AI document should be more relevant
    _log("Upsert search results:", newline_before=True)
    for doc in results:
        _log(f"  - {doc.page_content}: topic={doc.metadata.get('topic')}")


@pytest.mark.integration
def test_upsert_empty_list(vectorstore: "AlibabaCloudMySQL") -> None:
    """Test upsert with empty document list."""
    ids = vectorstore.upsert([])

    assert ids == []


# =============================================================================
# Async Method Integration Tests
# =============================================================================


@pytest.fixture
async def async_vectorstore(
    embeddings: Embeddings,
) -> AsyncGenerator["AlibabaCloudMySQL", None]:
    """Create an async AlibabaCloudMySQL instance for testing."""
    import asyncio

    from langchain_alibabacloud_mysql import AlibabaCloudMySQL

    table_name = f"test_langchain_async_{uuid.uuid4().hex[:8]}"

    store = AlibabaCloudMySQL(
        host=get_env_var("ALIBABACLOUD_MYSQL_HOST"),
        port=int(get_env_var("ALIBABACLOUD_MYSQL_PORT", "3306")),
        user=get_env_var("ALIBABACLOUD_MYSQL_USER"),
        password=get_env_var("ALIBABACLOUD_MYSQL_PASSWORD"),
        database=get_env_var("ALIBABACLOUD_MYSQL_DATABASE"),
        embedding=embeddings,
        table_name=table_name,
        pre_delete_table=True,
        pool_size=2,  # Reduce pool size to avoid connection exhaustion
    )

    yield store

    # Cleanup
    try:
        store.drop_table()
    except Exception:
        pass
    await store.aclose()
    # Longer delay to allow connections to be properly released on remote RDS
    await asyncio.sleep(0.5)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_aadd_texts_basic(async_vectorstore: "AlibabaCloudMySQL") -> None:
    """Test async add_texts basic functionality."""
    texts = ["异步测试文本1", "异步测试文本2"]
    metadatas = [{"source": "async_test"}, {"source": "async_test"}]

    ids = await async_vectorstore.aadd_texts(texts=texts, metadatas=metadatas)

    assert len(ids) == 2
    _log(f"aadd_texts: Added {len(ids)} texts asynchronously", newline_before=True)

    # Verify using sync method
    count = async_vectorstore.count()
    assert count == 2


@pytest.mark.integration
@pytest.mark.asyncio
async def test_aadd_documents_basic(async_vectorstore: "AlibabaCloudMySQL") -> None:
    """Test async add_documents functionality."""
    documents = [
        Document(
            id="async-doc-1",
            page_content="异步文档1",
            metadata={"type": "async"},
        ),
        Document(
            id="async-doc-2",
            page_content="异步文档2",
            metadata={"type": "async"},
        ),
    ]

    ids = await async_vectorstore.aadd_documents(documents)

    assert ids == ["async-doc-1", "async-doc-2"]
    _log(
        f"aadd_documents: Added {len(ids)} documents asynchronously",
        newline_before=True,
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_aadd_embeddings_basic(async_vectorstore: "AlibabaCloudMySQL") -> None:
    """Test async add_embeddings with pre-computed embeddings."""
    # Get embeddings first using sync method
    texts = ["预计算嵌入1", "预计算嵌入2"]
    embeddings_data = async_vectorstore.embeddings.embed_documents(texts)
    text_embeddings = list(zip(texts, embeddings_data))

    ids = await async_vectorstore.aadd_embeddings(
        text_embeddings=text_embeddings,
        metadatas=[{"precomputed": True}, {"precomputed": True}],
    )

    assert len(ids) == 2
    _log(
        f"aadd_embeddings: Added {len(ids)} embeddings asynchronously",
        newline_before=True,
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_aupsert_basic(async_vectorstore: "AlibabaCloudMySQL") -> None:
    """Test async upsert functionality."""
    # Insert
    documents = [
        Document(id="aupsert-1", page_content="原始内容", metadata={"version": 1}),
    ]
    ids = await async_vectorstore.aupsert(documents)
    assert ids == ["aupsert-1"]

    # Update
    updated_docs = [
        Document(id="aupsert-1", page_content="更新后内容", metadata={"version": 2}),
    ]
    ids = await async_vectorstore.aupsert(updated_docs)
    assert ids == ["aupsert-1"]

    # Verify update
    retrieved = async_vectorstore.get_by_ids(["aupsert-1"])
    assert len(retrieved) == 1
    assert retrieved[0].page_content == "更新后内容"
    assert retrieved[0].metadata["version"] == 2
    _log("aupsert: Insert and update successful", newline_before=True)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_asimilarity_search_basic(async_vectorstore: "AlibabaCloudMySQL") -> None:
    """Test async similarity search."""
    # Add documents first
    documents = [
        Document(
            id="search-1",
            page_content="人工智能是未来的发展方向",
            metadata={"topic": "AI"},
        ),
        Document(
            id="search-2",
            page_content="今天的天气非常好",
            metadata={"topic": "weather"},
        ),
    ]
    await async_vectorstore.aadd_documents(documents)

    # Async search
    results = await async_vectorstore.asimilarity_search("AI技术", k=2)

    assert len(results) == 2
    _log("asimilarity_search results:", newline_before=True)
    for doc in results:
        _log(f"  - {doc.page_content[:30]}...")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_asimilarity_search_with_score(
    async_vectorstore: "AlibabaCloudMySQL",
) -> None:
    """Test async similarity search with score."""
    documents = [
        Document(id="score-1", page_content="深度学习神经网络", metadata={}),
        Document(id="score-2", page_content="烹饪美食教程", metadata={}),
    ]
    await async_vectorstore.aadd_documents(documents)

    results = await async_vectorstore.asimilarity_search_with_score("机器学习", k=2)

    assert len(results) == 2
    _log("asimilarity_search_with_score results:", newline_before=True)
    for doc, score in results:
        _log(f"  - score={score:.4f}: {doc.page_content}")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_adelete_basic(async_vectorstore: "AlibabaCloudMySQL") -> None:
    """Test async delete functionality."""
    # Add documents
    documents = [
        Document(id="del-1", page_content="待删除1", metadata={}),
        Document(id="del-2", page_content="待删除2", metadata={}),
    ]
    await async_vectorstore.aadd_documents(documents)
    assert async_vectorstore.count() == 2

    # Delete one
    result = await async_vectorstore.adelete(ids=["del-1"])
    assert result is True
    assert async_vectorstore.count() == 1

    # Delete remaining
    result = await async_vectorstore.adelete(ids=["del-2"])
    assert result is True
    assert async_vectorstore.count() == 0
    _log("adelete: Deleted documents asynchronously", newline_before=True)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_aget_by_ids_basic(async_vectorstore: "AlibabaCloudMySQL") -> None:
    """Test async get_by_ids functionality."""
    documents = [
        Document(id="get-1", page_content="获取测试1", metadata={"key": "v1"}),
        Document(id="get-2", page_content="获取测试2", metadata={"key": "v2"}),
    ]
    await async_vectorstore.aadd_documents(documents)

    # Async get
    retrieved = await async_vectorstore.aget_by_ids(["get-1", "get-2"])

    assert len(retrieved) == 2
    _log("aget_by_ids: Retrieved documents asynchronously", newline_before=True)
    for doc in retrieved:
        _log(f"  - {doc.id}: {doc.page_content}")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_amax_marginal_relevance_search(
    async_vectorstore: "AlibabaCloudMySQL",
) -> None:
    """Test async MMR search."""
    documents = [
        Document(id="mmr-1", page_content="机器学习模型训练", metadata={}),
        Document(id="mmr-2", page_content="深度学习神经网络架构", metadata={}),
        Document(id="mmr-3", page_content="自然语言处理技术", metadata={}),
        Document(id="mmr-4", page_content="烹饪美食教程", metadata={}),
    ]
    await async_vectorstore.aadd_documents(documents)

    results = await async_vectorstore.amax_marginal_relevance_search(
        "AI人工智能",
        k=3,
        fetch_k=4,
    )

    assert len(results) <= 3
    _log("amax_marginal_relevance_search results:", newline_before=True)
    for doc in results:
        _log(f"  - {doc.page_content}")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_afrom_texts_class_method() -> None:
    """Test async from_texts class method."""
    from langchain_alibabacloud_mysql import AlibabaCloudMySQL

    embeddings = _create_embeddings()
    table_name = f"test_langchain_afrom_{uuid.uuid4().hex[:8]}"

    try:
        store = await AlibabaCloudMySQL.afrom_texts(
            texts=["异步创建文本1", "异步创建文本2"],
            embedding=embeddings,
            host=get_env_var("ALIBABACLOUD_MYSQL_HOST"),
            port=int(get_env_var("ALIBABACLOUD_MYSQL_PORT", "3306")),
            user=get_env_var("ALIBABACLOUD_MYSQL_USER"),
            password=get_env_var("ALIBABACLOUD_MYSQL_PASSWORD"),
            database=get_env_var("ALIBABACLOUD_MYSQL_DATABASE"),
            table_name=table_name,
            pool_size=2,
        )

        assert store.count() == 2
        _log("afrom_texts: Created store asynchronously", newline_before=True)
    finally:
        try:
            store.drop_table()
            await store.aclose()
        except Exception:
            pass


@pytest.mark.integration
@pytest.mark.asyncio
async def test_afrom_documents_class_method() -> None:
    """Test async from_documents class method."""
    from langchain_alibabacloud_mysql import AlibabaCloudMySQL

    embeddings = _create_embeddings()
    table_name = f"test_langchain_afromdoc_{uuid.uuid4().hex[:8]}"

    documents = [
        Document(id="afd-1", page_content="异步文档1", metadata={"async": True}),
        Document(id="afd-2", page_content="异步文档2", metadata={"async": True}),
    ]

    try:
        store = await AlibabaCloudMySQL.afrom_documents(
            documents=documents,
            embedding=embeddings,
            host=get_env_var("ALIBABACLOUD_MYSQL_HOST"),
            port=int(get_env_var("ALIBABACLOUD_MYSQL_PORT", "3306")),
            user=get_env_var("ALIBABACLOUD_MYSQL_USER"),
            password=get_env_var("ALIBABACLOUD_MYSQL_PASSWORD"),
            database=get_env_var("ALIBABACLOUD_MYSQL_DATABASE"),
            table_name=table_name,
            pool_size=2,
        )

        assert store.count() == 2
        _log("afrom_documents: Created store asynchronously", newline_before=True)
    finally:
        try:
            store.drop_table()
            await store.aclose()
        except Exception:
            pass


@pytest.mark.integration
@pytest.mark.asyncio
async def test_aclear_and_acount(async_vectorstore: "AlibabaCloudMySQL") -> None:
    """Test async clear and count."""
    # Add documents
    documents = [
        Document(id="clear-1", page_content="清空测试1", metadata={}),
        Document(id="clear-2", page_content="清空测试2", metadata={}),
    ]
    await async_vectorstore.aadd_documents(documents)

    # Async count
    count = await async_vectorstore.acount()
    assert count == 2

    # Async clear
    await async_vectorstore.aclear()

    count = await async_vectorstore.acount()
    assert count == 0
    _log("aclear/acount: Cleared and counted asynchronously", newline_before=True)


def _create_embeddings() -> Embeddings:
    """Create embeddings for testing."""
    try:
        from langchain_community.embeddings import DashScopeEmbeddings

        return DashScopeEmbeddings(
            model="text-embedding-v3",
            dashscope_api_key=get_env_var("DASHSCOPE_API_KEY"),
        )
    except Exception:
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings()
