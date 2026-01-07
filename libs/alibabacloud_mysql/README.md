# 🦜️🔗 LangChain AlibabaCloud MySQL

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/LangChainAI.svg?style=social&label=Follow%20%40LangChain)](https://x.com/LangChainAI)

This package provides LangChain integration with AlibabaCloud RDS MySQL's native vector search capabilities.

## Requirements

- **Python 3.10+**
- **AlibabaCloud RDS MySQL 8.0.36+** with Vector support
- **rds_release_date >= 20251031**

## Installation

```bash
pip install -U langchain-alibabacloud-mysql
```

### Optional Dependencies

For DashScope embeddings (recommended for Alibaba Cloud):
```bash
pip install langchain-community dashscope
```

## Quick Start

### Using DashScope Embeddings (Recommended)

```python
from langchain_alibabacloud_mysql import AlibabaCloudMySQL
from langchain_community.embeddings import DashScopeEmbeddings

embeddings = DashScopeEmbeddings(
    model="text-embedding-v4",
    dashscope_api_key="your-dashscope-api-key",
)

vectorstore = AlibabaCloudMySQL(
    host="your-rds-host.mysql.rds.aliyuncs.com",
    port=3306,
    user="your-user",
    password="your-password",
    database="your-database",
    embedding=embeddings,
    table_name="langchain_vectors",
    distance_strategy="cosine",
    hnsw_m=6,
)

# Add texts
vectorstore.add_texts(["Hello world", "LangChain is great"])

# Search
results = vectorstore.similarity_search("Hello", k=2)
```

### Using Other Embedding Models

Any LangChain-compatible embedding model can be used:

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
# ... rest remains the same
```

## Features

- **Native VECTOR data type** for efficient storage
- **HNSW vector index** for fast approximate nearest neighbor search
- **Cosine and Euclidean** distance metrics
- **MMR search** for diverse results
- **Metadata filtering** support
- **Batch operations** for efficient data loading

## API Reference

### AlibabaCloudMySQL

Main vector store class.

**Constructor Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `host` | str | - | MySQL host |
| `port` | int | - | MySQL port |
| `user` | str | - | Username |
| `password` | str | - | Password |
| `database` | str | - | Database name |
| `embedding` | Embeddings | - | Embedding model |
| `table_name` | str | "langchain_vectors" | Table name |
| `distance_strategy` | str | "cosine" | "cosine" or "euclidean" |
| `hnsw_m` | int | 6 | HNSW M parameter |

**Methods:**

- `add_texts(texts, metadatas, ids)` - Add texts to the store
- `similarity_search(query, k, filter)` - Search for similar documents
- `similarity_search_with_score(query, k, filter)` - Search with similarity scores
- `max_marginal_relevance_search(query, k, fetch_k, lambda_mult, filter)` - MMR search
- `delete(ids)` - Delete vectors by IDs
- `get_by_ids(ids)` - Get documents by IDs
- `count()` - Count total vectors
- `clear()` - Clear all vectors
- `drop_table()` - Drop the vector table
- `from_texts(texts, embedding, ...)` - Create from texts
- `from_documents(documents, embedding, ...)` - Create from documents

## Demo Tests

See `tests/demo_tests/` for comprehensive examples:
- `RAG-agent.py` - RAG agent with retrieval tools
- `filter-query.py` - Metadata filtering examples
- `semantic-search.py` - Basic similarity search

## License

MIT License
