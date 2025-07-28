# EmbeddingTest

Embedding model selection for RAG database - **RSPEED-1798**

## Project Goal

Select an embedding model for use in our RAG (Retrieval-Augmented Generation) database that will be shipped with the system for document embedding and query processing.

## Requirements

### Acceptance Criteria
- ✅ **Legal Compliance**: Must be approved by Legal for redistribution
- ✅ **Performance**: Should rank highly on HuggingFace MTEB benchmark
- ✅ **Context Window**: Must support ≥500 tokens
- ✅ **Resource Constraints**: Must run locally in container within on-premises machine resources

## Evaluation Methodology

### Task-Specific Performance Focus
Rather than relying solely on overall MTEB rankings, we prioritized tasks most relevant to RAG:

**Primary RAG Tasks:**
- **Retrieval**: Finding relevant documents for queries (core RAG functionality)
- **Semantic Textual Similarity (STS)**: Measuring text passage similarity

**Secondary Tasks:**
- Clustering, Instruction Retrieval (useful for document organization)

**Excluded Tasks:**
- Classification, Multilabel Classification, Pair Classification, Bitext Mining (not relevant to RAG use case)

### Analysis Dataset
- Source: HuggingFace MTEB leaderboard
- Filter: Models with <100M parameters
- Focus: RAG-optimized performance metrics (Retrieval + STS scores)

## Recommended Models

Based on our criteria analysis, we recommend these three models for further evaluation:

### 1. intfloat/e5-small-v2
- **RAG Score**: 49.63 (Retrieval: 39.38, STS: 59.87)
- **Memory**: 127MB RAM
- **Context Window**: 512 tokens
- **Parameters**: 33M
- **License**: MIT ✅

### 2. intfloat/e5-small  
- **RAG Score**: 48.81 (Retrieval: 36.25, STS: 61.37)
- **Memory**: 127MB RAM
- **Context Window**: 512 tokens
- **Parameters**: 33M
- **License**: MIT ✅

### 3. BAAI/bge-small-en-v1.5
- **RAG Score**: 48.00 (Retrieval: 36.26, STS: 59.73)
- **Memory**: 127MB RAM
- **Context Window**: 512 tokens
- **Parameters**: 33M
- **License**: MIT ✅

## Selection Rationale

All three models:
- Use MIT license (approved for redistribution)
- Excel at RAG-specific tasks rather than general embedding tasks
- Meet context window requirements (512 > 500 tokens)
- Have reasonable memory footprint (~127MB) for container deployment
- Represent the top performers when optimized for retrieval use cases


### Manual Usage
```bash
# Install dependencies
pip install -r requirements.txt

# Fetch latest MTEB data (saves timestamped CSV)
python3 analyze_mteb_csv.py hugging_face_stats_2025_07_25.csv
```

## Next Steps

1. Performance testing with representative documents from target domain
2. Latency benchmarking for query embedding generation
3. Final model selection based on domain-specific evaluation results
4. **Periodic model updates**: Download a new `hugging_face_stats` csv and run `analyze_mteb_csv.py` on it
