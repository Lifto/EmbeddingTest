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

## Data Source Strategy

### Initial Approach: MTEB Library (Not Recommended)

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

#### Steps:
1. **Download CSV**: Visit [https://huggingface.co/spaces/mteb/leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
2. **Apply Filters**: Set "Number of Parameters" to <100M 
3. **Export Data**: Click "Download CSV" button
4. **Save File**: Place CSV in `/stats` directory with date format: `hugging_face_stats_YYYY_MM_DD.csv` (script will auto-select most recent)

#### Analysis:
```bash
# Install dependencies
pip install -r requirements.txt

# Basic analysis (uses license info from CSV)
python3 analyze_mteb_csv.py ./stats/hugging_face_stats_YYYY_MM_DD.csv

# Enhanced analysis with license scraping (slower but more accurate)  
python3 analyze_mteb_csv.py ./stats/hugging_face_stats_YYYY_MM_DD.csv --scrape-licenses

# Note: run_full_analysis.sh automatically finds the most recent dated CSV file
```

#### License Verification

The script includes two modes for license analysis:

**Basic Mode** (default): Uses license information from the CSV file
- ✅ **Fast**: No web requests required
- ⚠️ **Limited**: Some CSV files may not contain license information

**Enhanced Mode** (`--scrape-licenses` flag): Smart scraping for top models until 3 redistributable licenses found
- ✅ **Fast & Smart**: Only scrapes top performers until finding 3 redistributable options
- ✅ **Comprehensive**: Gets current license AND language information directly from source
- ✅ **Optimized**: Stops early rather than scraping all models
- 📊 **Transparent**: Shows "License_Source" and "Languages" columns indicating data origin

### Why This Approach Works Better

- ✅ **Validated Data**: Only includes approved evaluation results
- ✅ **Consistent Rankings**: Matches official leaderboard
- ✅ **Quality Control**: HuggingFace team filters out problematic evaluations
- ✅ **Performance**: Fast analysis without downloading entire MTEB database
- ✅ **Reliability**: Trusted source for production model selection

## Local Model Testing

After identifying promising models, validate them on your actual hardware:

### Prerequisites

1. **Install Ollama**: Download from [https://ollama.ai/](https://ollama.ai/)
2. **Install Dependencies**: `pip install -r requirements.txt`

### Individual Model Testing

Test a single model's performance:

```bash
# Test specific model
python3 test_embedding_models.py intfloat/e5-small-v2

# Test other top candidates
python3 test_embedding_models.py avsolatorio/NoInstruct-small-Embedding-v0
python3 test_embedding_models.py avsolatorio/GIST-small-Embedding-v0
```

**What it tests:**
- ⏱️ **Download time**: How long to pull the model
- 🚀 **Embedding speed**: Time per query (critical for RAG latency)
- 💾 **Memory usage**: RAM increase during inference
- 📏 **Vector dimensions**: Embedding size for accuracy assessment
- 🧪 **Real queries**: Uses actual RHEL system administration questions

### Batch Testing (Recommended)

Test all top models automatically and get a comparison:

```bash
python3 test_all_models.py
```

**Features:**
- 🏆 **Automated comparison**: Tests all 3 top models sequentially
- 📊 **Performance ranking**: Sorts by speed and provides grades
- 💡 **System requirements**: Analyzes hardware needs
- 🎯 **Final recommendation**: Picks the best model for your system
- 💾 **Detailed logs**: Saves complete results for later analysis

**Expected Output:**
```
🏆 RECOMMENDED MODEL: intfloat/e5-small-v2
   ⚡ Speed: 0.45s average per query
   📏 Dimension: 384 (good for accuracy)
   💾 Memory: 127MB increase during inference
   🎯 Assessment: 🚀 EXCELLENT - Perfect for real-time RAG applications
```

### Complete Workflow (One-Click)

Run the entire analysis and testing pipeline automatically:

```bash
./run_full_analysis.sh
```

**What it does:**
1. 🔧 **Setup**: Creates virtual environment and installs dependencies
2. 🔍 **Analysis**: Runs CSV analysis with license scraping  
3. 🧪 **Testing**: Tests all top models locally with Ollama
4. 📊 **Report**: Generates comprehensive comparison and recommendation
5. 💾 **Documentation**: Saves all results for future reference

This is the **recommended approach** for a complete evaluation workflow.

## Next Steps

1. Performance testing with representative documents from target domain
2. Latency benchmarking for query embedding generation
3. Final model selection based on domain-specific evaluation results
4. **Periodic model updates**: Download a new `hugging_face_stats` csv and run `analyze_mteb_csv.py` on it
