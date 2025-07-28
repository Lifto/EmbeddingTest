#!/usr/bin/env python3
"""
Direct Embedding Model Testing Script

Tests HuggingFace embedding models directly using sentence_transformers
without requiring Ollama. More reliable for models from our MTEB analysis.

REQUIREMENTS:
pip install sentence-transformers torch

USAGE:
    python3 test_direct_embedding.py <model_name>
    
    Example:
    python3 test_direct_embedding.py intfloat/e5-small-v2

WHAT IT TESTS:
- Model download/loading time
- Embedding generation speed  
- Memory usage during inference
- Output vector dimensions
- Works with actual HuggingFace model names from our analysis
"""

import time
import json
import sys
import psutil
import os
import torch
from typing import List, Dict, Tuple, Optional
from datetime import datetime

# Test queries - common RHEL system administration questions
# Note: e5 models require "query: " prefix
TEST_QUERIES = [
    "query: How do I configure firewall rules in RHEL 9 using firewalld?",
    "query: What are the steps to troubleshoot high CPU usage on a Red Hat Enterprise Linux server?", 
    "query: How do I set up automatic security updates in Red Hat Enterprise Linux using dnf-automatic?"
]

class DirectEmbeddingTester:
    def __init__(self):
        self.model = None
        self.model_name = None
        
    def load_model(self, model_name: str) -> Tuple[bool, float, Dict]:
        """Load the embedding model"""
        print(f"📥 Loading model '{model_name}'...")
        start_time = time.time()
        
        try:
            # Import here to give better error messages
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                print("❌ sentence-transformers not installed!")
                print("Install with: pip install sentence-transformers")
                return False, 0.0, {"error": "sentence-transformers not installed"}
            
            # Load the model
            self.model = SentenceTransformer(model_name)
            self.model_name = model_name
            
            load_time = time.time() - start_time
            
            # Get model info
            model_info = {
                "embedding_dimension": self.model.get_sentence_embedding_dimension(),
                "max_seq_length": getattr(self.model, 'max_seq_length', 'Unknown'),
                "device": str(self.model.device),
            }
            
            print(f"✅ Model loaded successfully in {load_time:.1f}s")
            print(f"   📏 Embedding dimension: {model_info['embedding_dimension']}")
            print(f"   📝 Max sequence length: {model_info['max_seq_length']}")
            print(f"   💻 Device: {model_info['device']}")
            
            return True, load_time, model_info
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False, 0.0, {"error": str(e)}
    
    def get_embeddings(self, texts: List[str]) -> Tuple[Optional[List], float, Dict]:
        """Generate embeddings for given texts"""
        if self.model is None:
            return None, 0.0, {"error": "Model not loaded"}
        
        start_time = time.time()
        process = psutil.Process(os.getpid())
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        individual_times = []
        
        try:
            print(f"🔄 Generating embeddings for {len(texts)} texts...")
            
            # Time each embedding individually for detailed metrics
            embeddings = []
            for i, text in enumerate(texts, 1):
                query_start = time.time()
                
                # Generate single embedding
                embedding = self.model.encode([text], normalize_embeddings=True)
                embeddings.append(embedding[0])  # Extract from batch
                
                query_time = time.time() - query_start
                individual_times.append(query_time)
                
                print(f"   ✅ Embedding {i}/{len(texts)}: {len(embedding[0])}-dim in {query_time:.3f}s")
            
            total_time = time.time() - start_time
            end_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = end_memory - start_memory
            
            stats = {
                'total_time': total_time,
                'avg_time_per_query': sum(individual_times) / len(individual_times),
                'individual_times': individual_times,
                'memory_increase_mb': memory_increase,
                'embedding_dimension': len(embeddings[0]) if embeddings else 0,
                'total_embeddings': len(embeddings),
                'embeddings_per_second': len(embeddings) / total_time
            }
            
            return embeddings, total_time, stats
            
        except Exception as e:
            print(f"❌ Error generating embeddings: {e}")
            return None, 0.0, {"error": str(e)}
    
    def test_similarity(self, embeddings: List) -> Dict:
        """Test similarity computation between embeddings"""
        if len(embeddings) < 2:
            return {}
        
        try:
            import numpy as np
            
            # Convert to numpy for easier computation
            embed_array = np.array(embeddings)
            
            # Compute similarity matrix
            similarities = np.dot(embed_array, embed_array.T)
            
            # Extract some interesting similarities
            sim_stats = {
                'query1_vs_query2': float(similarities[0, 1]),
                'query1_vs_query3': float(similarities[0, 2]), 
                'query2_vs_query3': float(similarities[1, 2]),
                'avg_inter_query_similarity': float(np.mean([similarities[0, 1], similarities[0, 2], similarities[1, 2]]))
            }
            
            return sim_stats
            
        except Exception as e:
            print(f"⚠️  Could not compute similarities: {e}")
            return {}
    
    def test_model(self, model_name: str) -> Dict:
        """Complete test of a model"""
        print(f"\n{'='*80}")
        print(f"🧪 TESTING EMBEDDING MODEL: {model_name}")
        print(f"{'='*80}")
        
        # Step 1: Load model
        load_success, load_time, model_info = self.load_model(model_name)
        if not load_success:
            return {"error": f"Failed to load model: {model_info.get('error', 'Unknown error')}"}
        
        # Step 2: Test embeddings
        print(f"\n📝 Testing with {len(TEST_QUERIES)} RHEL queries...")
        embeddings, total_time, stats = self.get_embeddings(TEST_QUERIES)
        
        if embeddings is None:
            return {"error": f"Failed to generate embeddings: {stats.get('error', 'Unknown error')}"}
        
        # Step 3: Test similarities
        print(f"\n🔍 Computing query similarities...")
        similarity_stats = self.test_similarity(embeddings)
        
        # Step 4: Compile results
        results = {
            "model": model_name,
            "timestamp": datetime.now().isoformat(),
            "load_time_seconds": load_time,
            "model_info": model_info,
            "embedding_stats": stats,
            "similarity_stats": similarity_stats,
            "test_queries": TEST_QUERIES,
            "success": True
        }
        
        return results

def print_test_results(results: Dict):
    """Print formatted test results"""
    if "error" in results:
        print(f"\n❌ TEST FAILED: {results['error']}")
        return
    
    model = results['model']
    stats = results['embedding_stats']
    model_info = results['model_info']
    load_time = results['load_time_seconds']
    sim_stats = results.get('similarity_stats', {})
    
    print(f"\n🎉 TEST COMPLETED SUCCESSFULLY!")
    print(f"\n📊 PERFORMANCE SUMMARY:")
    print(f"   Model: {model}")
    print(f"   Load time: {load_time:.1f}s")
    print(f"   Embedding dimension: {model_info['embedding_dimension']}")
    print(f"   Max sequence length: {model_info['max_seq_length']}")
    print(f"   Device: {model_info['device']}")
    print(f"   Total queries processed: {stats['total_embeddings']}")
    print(f"   Total embedding time: {stats['total_time']:.2f}s")
    print(f"   Average time per query: {stats['avg_time_per_query']:.3f}s")
    print(f"   Embeddings per second: {stats['embeddings_per_second']:.1f}")
    print(f"   Memory increase: {stats['memory_increase_mb']:.1f}MB")
    
    print(f"\n⏱️  INDIVIDUAL QUERY TIMES:")
    for i, (query, time_taken) in enumerate(zip(TEST_QUERIES, stats['individual_times']), 1):
        # Remove "query: " prefix for display
        display_query = query.replace("query: ", "")
        print(f"   {i}. {time_taken:.3f}s - {display_query[:60]}...")
    
    # Similarity analysis
    if sim_stats:
        print(f"\n🔍 SIMILARITY ANALYSIS:")
        print(f"   Average inter-query similarity: {sim_stats.get('avg_inter_query_similarity', 0):.3f}")
        print(f"   Query 1 vs 2: {sim_stats.get('query1_vs_query2', 0):.3f}")
        print(f"   Query 1 vs 3: {sim_stats.get('query1_vs_query3', 0):.3f}")
        print(f"   Query 2 vs 3: {sim_stats.get('query2_vs_query3', 0):.3f}")
    
    # Performance assessment
    avg_time = stats['avg_time_per_query']
    if avg_time < 0.1:
        performance = "🚀 Excellent"
    elif avg_time < 0.5:
        performance = "✅ Good" 
    elif avg_time < 1.0:
        performance = "⚠️  Acceptable"
    else:
        performance = "🐌 Slow"
    
    print(f"\n🎯 PERFORMANCE ASSESSMENT: {performance}")
    print(f"   Average {avg_time:.3f}s per query is {performance.split()[1].lower()} for RAG applications")

def main():
    """Main execution"""
    if len(sys.argv) != 2:
        print("Usage: python3 test_direct_embedding.py <model_name>")
        print("\nExamples:")
        print("  python3 test_direct_embedding.py intfloat/e5-small-v2")
        print("  python3 test_direct_embedding.py avsolatorio/NoInstruct-small-Embedding-v0")
        print("  python3 test_direct_embedding.py avsolatorio/GIST-small-Embedding-v0")
        print("\nNote: These models will be downloaded from HuggingFace automatically")
        sys.exit(1)
    
    model_name = sys.argv[1]
    
    # Initialize tester
    tester = DirectEmbeddingTester()
    
    # Run test
    results = tester.test_model(model_name)
    
    # Print results
    print_test_results(results)
    
    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"direct_embedding_test_{model_name.replace('/', '_')}_{timestamp}.json"
    
    try:
        with open(filename, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            if 'embeddings' in results:
                del results['embeddings']  # Don't save actual embeddings, too large
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Results saved to: {filename}")
    except Exception as e:
        print(f"\n⚠️  Could not save results: {e}")

if __name__ == "__main__":
    main() 