# EmbeddingTest

Embedding model selection for RAG database - **RSPEED-1798**

## Project Goal

Select an embedding model for use in our RAG (Retrieval-Augmented Generation) database that will be shipped with the system for document embedding and query processing.

#### Steps:
1. **Install Prerequisites**: `pip install -r requirements.txt`
1. **Download CSV**: Visit [https://huggingface.co/spaces/mteb/leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
1. **Apply Filters**: Set "Number of Parameters" to <100M 
1. **Export Data**: Click "Download CSV" button
1. **Save File**: Place CSV in `/stats` directory with date format: `hugging_face_stats_YYYY_MM_DD.csv` (script will auto-select most recent)
1. **Run Analysis**: `./run_full_analysis.sh`

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

### Note On Initial Approach: MTEB Library (Not Recommended)

We initially attempted to use the official MTEB Python library to fetch live benchmark data:

```bash
python3 fetch_with_mteb_lib.py  # Available for reference
```

**Problems Discovered:**
1. **Ranking Discrepancies**: Models ranked differently in raw MTEB data vs. official HuggingFace leaderboard
2. **Evaluation Bias**: Newer models evaluated on fewer tasks appeared to score higher due to selection bias
3. **Data Quality Issues**: Raw MTEB data includes experimental/unvalidated evaluations not shown on official leaderboard
4. **Performance**: Loading complete MTEB database is slow and resource-intensive

**Example Issue**: `prdev/mini-gte` showed high scores in raw MTEB data but appears with empty scores on the [official leaderboard](https://huggingface.co/spaces/mteb/leaderboard), indicating the raw data contains unvalidated results.

### Recommended Approach: Manual CSV Download

The **HuggingFace MTEB leaderboard** uses curated, validated evaluation results. For reliable model selection:
