#!/usr/bin/env python3
"""
Direct Transformers Model Testing Script

Tests HuggingFace embedding models using the transformers library directly
to bypass sentence-transformers compatibility issues.

REQUIREMENTS:
pip install transformers torch numpy

USAGE:
    python3 test_transformers_direct.py <model_name>
    
    Example:
    python3 test_transformers_direct.py avsolatorio/NoInstruct-small-Embedding-v0

WHAT IT TESTS:
- Model download/loading time using raw transformers
- Embedding generation speed with manual pooling
- Memory usage during inference
- Output vector dimensions
- Works around sentence-transformers compatibility issues
"""

import time
import json
import sys
import psutil
import os
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime

# Test queries - using both with and without prefixes to test compatibility
TEST_QUERIES = [
    "How do I configure firewall rules in RHEL 9 using firewalld?",
    "What are the steps to troubleshoot high CPU usage on a Red Hat Enterprise Linux server?", 
    "How do I set up automatic security updates in Red Hat Enterprise Linux using dnf-automatic?"
]

def average_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Average pooling function similar to what sentence-transformers uses"""
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

class TransformersDirectTester:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_name = None
        
    def load_model(self, model_name: str) -> Tuple[bool, float, Dict]:
        """Load the embedding model using transformers directly"""
        print(f"📥 Loading model '{model_name}' with transformers...")
        start_time = time.time()
        
        try:
            # Import here to give better error messages
            try:
                from transformers import AutoTokenizer, AutoModel
            except ImportError:
                print("❌ transformers not installed!")
                print("Install with: pip install transformers torch")
                return False, 0.0, {"error": "transformers not installed"}
            
            # Load tokenizer and model
            print("   🔤 Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            print("   🧠 Loading model...")
            self.model = AutoModel.from_pretrained(model_name)
            
            # Move to appropriate device
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
            self.model = self.model.to(device)
            print(f"   💻 Moved to device: {device}")
            
            self.model_name = model_name
            
            load_time = time.time() - start_time
            
            # Get model info
            model_info = {
                "embedding_dimension": self.model.config.hidden_size,
                "max_position_embeddings": getattr(self.model.config, 'max_position_embeddings', 'Unknown'),
                "device": str(device),
                "vocab_size": self.tokenizer.vocab_size,
                "model_type": self.model.config.model_type,
            }
            
            print(f"✅ Model loaded successfully in {load_time:.1f}s")
            print(f"   📏 Hidden size: {model_info['embedding_dimension']}")
            print(f"   📝 Max positions: {model_info['max_position_embeddings']}")
            print(f"   🔤 Vocab size: {model_info['vocab_size']}")
            print(f"   🏗️  Model type: {model_info['model_type']}")
            
            return True, load_time, model_info
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False, 0.0, {"error": str(e)}
    
    def get_embeddings(self, texts: List[str], try_prefixes: bool = True) -> Tuple[Optional[List], float, Dict]:
        """Generate embeddings for given texts"""
        if self.model is None or self.tokenizer is None:
            return None, 0.0, {"error": "Model not loaded"}
        
        start_time = time.time()
        process = psutil.Process(os.getpid())
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        individual_times = []
        
        try:
            print(f"🔄 Generating embeddings for {len(texts)} texts...")
            
            # Try with different prefixes to see what works best
            test_variants = []
            if try_prefixes:
                test_variants.extend([
                    ("query: " + text for text in texts),  # e5-style prefix
                    texts,  # no prefix
                ])
            else:
                test_variants = [texts]
            
            best_embeddings = None
            best_stats = None
            best_avg_time = float('inf')
            
            for variant_idx, text_variant in enumerate(test_variants):
                variant_texts = list(text_variant)
                variant_times = []
                variant_embeddings = []
                
                prefix_type = "with 'query:' prefix" if variant_idx == 0 and try_prefixes else "without prefix"
                print(f"   🧪 Testing {prefix_type}...")
                
                try:
                    for i, text in enumerate(variant_texts, 1):
                        query_start = time.time()
                        
                        # Tokenize
                        inputs = self.tokenizer(
                            text, 
                            max_length=512, 
                            padding=True, 
                            truncation=True, 
                            return_tensors='pt'
                        )
                        
                        # Move to same device as model
                        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                        
                        # Get model outputs
                        with torch.no_grad():
                            outputs = self.model(**inputs)
                        
                        # Pool the embeddings (average pooling)
                        embeddings = average_pool(outputs.last_hidden_state, inputs['attention_mask'])
                        
                        # Normalize
                        embeddings = F.normalize(embeddings, p=2, dim=1)
                        
                        # Convert to numpy
                        embedding = embeddings.cpu().numpy()[0]
                        variant_embeddings.append(embedding)
                        
                        query_time = time.time() - query_start
                        variant_times.append(query_time)
                        
                        print(f"      ✅ Embedding {i}/{len(texts)}: {len(embedding)}-dim in {query_time:.3f}s")
                    
                    # Check if this variant is better (faster on average)
                    avg_time = sum(variant_times) / len(variant_times)
                    if avg_time < best_avg_time:
                        best_avg_time = avg_time
                        best_embeddings = variant_embeddings
                        best_stats = {
                            'individual_times': variant_times,
                            'prefix_used': prefix_type,
                            'variant_index': variant_idx
                        }
                        print(f"      🏆 Best variant so far (avg: {avg_time:.3f}s)")
                    
                except Exception as e:
                    print(f"      ❌ Variant failed: {e}")
                    continue
            
            if best_embeddings is None:
                return None, 0.0, {"error": "All variants failed"}
            
            total_time = time.time() - start_time
            end_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = end_memory - start_memory
            
            stats = {
                'total_time': total_time,
                'avg_time_per_query': best_avg_time,
                'individual_times': best_stats['individual_times'],
                'memory_increase_mb': memory_increase,
                'embedding_dimension': len(best_embeddings[0]) if best_embeddings else 0,
                'total_embeddings': len(best_embeddings),
                'embeddings_per_second': len(best_embeddings) / total_time,
                'best_prefix': best_stats['prefix_used'],
                'variants_tested': len(test_variants)
            }
            
            print(f"   🎯 Best approach: {best_stats['prefix_used']}")
            
            return best_embeddings, total_time, stats
            
        except Exception as e:
            print(f"❌ Error generating embeddings: {e}")
            return None, 0.0, {"error": str(e)}
    
    def test_similarity(self, embeddings: List) -> Dict:
        """Test similarity computation between embeddings"""
        if len(embeddings) < 2:
            return {}
        
        try:
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
        print(f"🧪 TESTING EMBEDDING MODEL (TRANSFORMERS DIRECT): {model_name}")
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
            "method": "transformers_direct",
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
    print(f"   Method: Direct transformers (bypassed sentence-transformers)")
    print(f"   Load time: {load_time:.1f}s")
    print(f"   Embedding dimension: {model_info['embedding_dimension']}")
    print(f"   Max positions: {model_info['max_position_embeddings']}")
    print(f"   Device: {model_info['device']}")
    print(f"   Best prefix: {stats.get('best_prefix', 'Unknown')}")
    print(f"   Total queries processed: {stats['total_embeddings']}")
    print(f"   Total embedding time: {stats['total_time']:.2f}s")
    print(f"   Average time per query: {stats['avg_time_per_query']:.3f}s")
    print(f"   Embeddings per second: {stats['embeddings_per_second']:.1f}")
    print(f"   Memory increase: {stats['memory_increase_mb']:.1f}MB")
    
    print(f"\n⏱️  INDIVIDUAL QUERY TIMES:")
    for i, (query, time_taken) in enumerate(zip(TEST_QUERIES, stats['individual_times']), 1):
        print(f"   {i}. {time_taken:.3f}s - {query[:60]}...")
    
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
        print("Usage: python3 test_transformers_direct.py <model_name>")
        print("\nExamples:")
        print("  python3 test_transformers_direct.py avsolatorio/NoInstruct-small-Embedding-v0")
        print("  python3 test_transformers_direct.py intfloat/e5-small-v2")
        print("  python3 test_transformers_direct.py avsolatorio/GIST-small-Embedding-v0")
        print("\nNote: This uses transformers directly to bypass sentence-transformers issues")
        sys.exit(1)
    
    model_name = sys.argv[1]
    
    # Initialize tester
    tester = TransformersDirectTester()
    
    # Run test
    results = tester.test_model(model_name)
    
    # Print results
    print_test_results(results)
    
    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"transformers_direct_test_{model_name.replace('/', '_')}_{timestamp}.json"
    
    try:
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Results saved to: {filename}")
    except Exception as e:
        print(f"\n⚠️  Could not save results: {e}")

if __name__ == "__main__":
    main() 