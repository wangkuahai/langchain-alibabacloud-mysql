# 🦜️🔗 LangChain AlibabaCloud MySQL

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/LangChainAI.svg?style=social&label=Follow%20%40LangChain)](https://x.com/LangChainAI)

This package provides LangChain integration with AlibabaCloud RDS MySQL's native vector search capabilities.

## Requirements

- **AlibabaCloud RDS MySQL 8.0.36+** with Vector support
- **rds_release_date >= 20251031**

## Installation

```bash
pip install -U langchain-alibabacloud-mysql
```

## Quick Start

```python
from langchain_alibabacloud_mysql import AlibabaCloudMySQL
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

vectorstore = AlibabaCloudMySQL(
    host="your-rds-host.mysql.rds.aliyuncs.com",
    port=3306,
    user="your-user",
    password="your-password",
    database="your-database",
    embedding=embeddings,
    table_name="langchain_vectors",
    distance_strategy="cosine",
)

# Add texts
vectorstore.add_texts(["Hello world", "LangChain is great"])

# Search
results = vectorstore.similarity_search("Hello", k=2)
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
- `similarity_search_with_score(query, k)` - Search with similarity scores
- `max_marginal_relevance_search(query, k)` - MMR search
- `delete(ids)` - Delete vectors by IDs
- `get_by_ids(ids)` - Get documents by IDs
- `from_texts(texts, embedding, ...)` - Create from texts
- `from_documents(documents, embedding, ...)` - Create from documents

## License

MIT License
