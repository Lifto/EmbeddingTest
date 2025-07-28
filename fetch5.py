#!/usr/bin/env python3
"""
MTEB data fetch with metadata filtering
Stage 1: Filter and show acceptable models
"""

import mteb
import pprint

def meets_criteria(meta):
    """Check if model meets our RAG criteria"""
    try:
        # Check max tokens (500+)
        if not meta.max_tokens or meta.max_tokens < 500:
            return False, "max_tokens < 500"
        
        # Check parameters (<100M)
        if not meta.n_parameters or meta.n_parameters >= 100_000_000:
            return False, "parameters >= 100M"
        
        # Check license (not None/empty)
        if not meta.license or meta.license.lower() in ['none', 'unknown', '']:
            return False, "no license"
        
        # Check English language
        if not meta.languages or 'eng' not in str(meta.languages).lower():
            return False, "not English"
        
        # Check Sentence Transformers compatibility
        if not meta.framework or 'Sentence Transformers' not in str(meta.framework):
            return False, "not Sentence Transformers"
        
        return True, "meets criteria"
        
    except Exception as e:
        return False, f"error: {e}"

def model_summary(meta):
    """Create a summary dict for display"""
    return {
        'name': meta.name,
        'parameters': f"{meta.n_parameters/1_000_000:.1f}M" if meta.n_parameters else "Unknown",
        'max_tokens': meta.max_tokens,
        'license': meta.license,
        'languages': meta.languages,
        'framework': meta.framework,
        'memory_mb': meta.memory_usage_mb
    }

def main():
    print("🔍 Stage 1: Getting and filtering model metadata...")
    
    # Get all metadata
    print("Getting all model metadata...")
    metas = mteb.get_model_metas()
    print(f"Found {len(metas)} total models")
    
    # Filter based on criteria
    print("\nFiltering models...")
    acceptable_metas = []
    rejected_reasons = {}
    
    for meta in metas:
        meets, reason = meets_criteria(meta)
        if meets:
            acceptable_metas.append(meta)
        else:
            rejected_reasons[reason] = rejected_reasons.get(reason, 0) + 1
    
    # Show filtering results
    print(f"\n📊 FILTERING RESULTS:")
    print(f"✅ Acceptable models: {len(acceptable_metas)}")
    print(f"❌ Rejected models: {len(metas) - len(acceptable_metas)}")
    
    print(f"\n📋 Rejection reasons:")
    for reason, count in sorted(rejected_reasons.items(), key=lambda x: x[1], reverse=True):
        print(f"   {reason}: {count} models")
    
    # Show acceptable models
    if acceptable_metas:
        print(f"\n✅ ACCEPTABLE MODELS ({len(acceptable_metas)}):")
        print("=" * 80)
        
        for i, meta in enumerate(acceptable_metas, 1):
            print(f"\n{i}. {meta.name}")
            summary = model_summary(meta)
            for key, value in summary.items():
                if key != 'name':
                    print(f"   {key}: {value}")
    else:
        print("\n❌ No models found meeting all criteria!")
    
    print(f"\n🎯 Ready for Stage 2: Fetch scores for {len(acceptable_metas)} models")

if __name__ == "__main__":
    main() 