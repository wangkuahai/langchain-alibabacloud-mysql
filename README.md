# 🦜️🔗 LangChain AlibabaCloud MySQL

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/LangChainAI.svg?style=social&label=Follow%20%40LangChain)](https://x.com/LangChainAI)

This repository contains the AlibabaCloud RDS MySQL integration package for LangChain:

- [langchain-alibabacloud-mysql](https://pypi.org/project/langchain-alibabacloud-mysql/) - A powerful integration between LangChain and AlibabaCloud RDS MySQL, enabling native vector search capabilities for AI applications.

## Overview

LangChain AlibabaCloud MySQL provides seamless integration between LangChain, a framework for building applications with large language models (LLMs), and AlibabaCloud RDS MySQL with native vector search support. This integration enables efficient vector storage and retrieval for AI applications like semantic search, recommendation systems, and RAG (Retrieval Augmented Generation).

## Requirements

- **AlibabaCloud RDS MySQL 8.0.36+** with Vector support enabled
- **rds_release_date >= 20251031**

## Features

- **Native Vector Storage**: Store embeddings using MySQL's native `VECTOR` data type
- **HNSW Index**: Efficient approximate nearest neighbor search with configurable M parameter
- **Multiple Distance Metrics**: Support for both Cosine and Euclidean distance
- **Similarity Search**: Perform efficient similarity searches on vector data
- **MMR Search**: Maximal Marginal Relevance search for diverse results
- **Metadata Filtering**: Filter search results by metadata fields
- **Batch Operations**: Efficient batch insert with configurable batch size

## Installation

```bash
pip install -U langchain-alibabacloud-mysql
```

## Quick Start

```python
from langchain_alibabacloud_mysql import AlibabaCloudMySQL
from langchain_openai import OpenAIEmbeddings

# Initialize embeddings
embeddings = OpenAIEmbeddings()

# Create vector store
vectorstore = AlibabaCloudMySQL(
    host="your-rds-host.mysql.rds.aliyuncs.com",
    port=3306,
    user="your-user",
    password="your-password",
    database="your-database",
    embedding=embeddings,
    table_name="langchain_vectors",
    distance_strategy="cosine",  # or "euclidean"
    hnsw_m=6,  # HNSW index M parameter (3-200)
)

# Add documents
texts = ["LangChain is a framework for LLM applications", 
         "AlibabaCloud provides cloud services",
         "Vector search enables semantic similarity"]
ids = vectorstore.add_texts(texts)

# Similarity search
results = vectorstore.similarity_search("What is LangChain?", k=2)
for doc in results:
    print(f"- {doc.page_content}")

# Search with scores
results_with_scores = vectorstore.similarity_search_with_score("cloud services", k=2)
for doc, score in results_with_scores:
    print(f"[Score: {score:.4f}] {doc.page_content}")
```

## Usage Examples

### Creating from Documents

```python
from langchain_core.documents import Document

documents = [
    Document(page_content="Hello world", metadata={"source": "greeting"}),
    Document(page_content="LangChain is great", metadata={"source": "review"}),
]

vectorstore = AlibabaCloudMySQL.from_documents(
    documents=documents,
    embedding=embeddings,
    host="your-host",
    port=3306,
    user="your-user",
    password="your-password",
    database="your-database",
)
```

### Search with Metadata Filter

```python
# Add texts with metadata
texts = ["Apple is a fruit", "Banana is yellow", "Car is a vehicle"]
metadatas = [
    {"category": "fruit"},
    {"category": "fruit"},
    {"category": "vehicle"},
]
vectorstore.add_texts(texts, metadatas=metadatas)

# Search with filter
results = vectorstore.similarity_search(
    "yellow things",
    k=2,
    filter={"category": "fruit"}
)
```

### Maximal Marginal Relevance (MMR) Search

```python
# MMR search for diverse results
results = vectorstore.max_marginal_relevance_search(
    "technology",
    k=4,
    fetch_k=20,
    lambda_mult=0.5,  # 0 = max diversity, 1 = max relevance
)
```

### Delete and Manage Vectors

```python
# Delete by IDs
vectorstore.delete(ids=["id1", "id2"])

# Get documents by IDs
docs = vectorstore.get_by_ids(["id1", "id2"])

# Count vectors
count = vectorstore.count()

# Clear all data
vectorstore.clear()

# Drop table
vectorstore.drop_table()
```

## Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `host` | str | - | MySQL host address |
| `port` | int | - | MySQL port number |
| `user` | str | - | MySQL username |
| `password` | str | - | MySQL password |
| `database` | str | - | MySQL database name |
| `embedding` | Embeddings | - | LangChain embedding model |
| `table_name` | str | "langchain_vectors" | Table name for vectors |
| `distance_strategy` | str | "cosine" | Distance function ("cosine" or "euclidean") |
| `hnsw_m` | int | 6 | HNSW index M parameter (3-200) |
| `pool_size` | int | 5 | Connection pool size |
| `pre_delete_table` | bool | False | Delete table before creating |

## MySQL Vector Functions Used

This integration uses AlibabaCloud RDS MySQL's native vector functions:

- `VEC_FromText('[1,2,3]')` - Convert JSON array to VECTOR type
- `VEC_DISTANCE_COSINE(v1, v2)` - Cosine distance between vectors
- `VEC_DISTANCE_EUCLIDEAN(v1, v2)` - Euclidean distance between vectors
- `VECTOR(dimension)` - Vector column data type
- `VECTOR INDEX ... M=m DISTANCE=distance` - HNSW vector index

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](https://github.com/langchain-ai/langchain-alibabacloud-mysql/blob/main/CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/langchain-ai/langchain-alibabacloud-mysql/blob/main/LICENSE) file for details.
