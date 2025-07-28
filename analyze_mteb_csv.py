#!/usr/bin/env python3
"""
RAG-Focused MTEB Analysis Script

Processes HuggingFace MTEB leaderboard CSV to create custom analysis
focused on RAG (Retrieval-Augmented Generation) use cases.

HOW TO GET THE CSV:
1. Go to https://huggingface.co/spaces/mteb/leaderboard
2. Apply filters if desired (e.g., sub-100M parameters)
3. Click "Download CSV" button
4. Save as 'mteb_data.csv' in this directory

USAGE:
    python3 analyze_mteb_csv.py [csv_file]
    
    If no csv_file specified, looks for 'mteb_data.csv'
"""

import pandas as pd
import logging
import sys
from datetime import datetime
from typing import Dict, List, Tuple
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def safe_numeric_convert(value, default=0):
    """Safely convert value to numeric, handling various formats"""
    if pd.isna(value) or value == '-' or value == '':
        return default
    
    try:
        # Handle string formats like '33M', '500K', etc.
        if isinstance(value, str):
            value = value.strip().upper()
            if value.endswith('M'):
                return float(value[:-1]) * 1_000_000
            elif value.endswith('K'):
                return float(value[:-1]) * 1_000
            elif value.endswith('B'):
                return float(value[:-1]) * 1_000_000_000
        
        return float(value)
    except (ValueError, TypeError):
        return default

def calculate_rag_score(row: pd.Series) -> float:
    """Calculate RAG-specific score focusing on retrieval and STS tasks"""
    
    # Get scores for RAG-relevant tasks
    retrieval = safe_numeric_convert(row.get('Retrieval', 0))
    sts = safe_numeric_convert(row.get('STS', 0))  # Fixed column name
    reranking = safe_numeric_convert(row.get('Reranking', 0))
    
    # If no scores available, return 0
    if retrieval == 0 and sts == 0 and reranking == 0:
        return 0.0
    
    # Weighted combination emphasizing retrieval
    # Retrieval: 50%, STS: 35%, Reranking: 15%
    rag_score = (retrieval * 0.5) + (sts * 0.35) + (reranking * 0.15)
    
    return round(rag_score, 2)

def get_license_status(license_str) -> str:
    """Determine license status for redistribution"""
    if pd.isna(license_str) or license_str == '-' or not license_str:
        return "❓ Unknown"
    
    license_lower = str(license_str).lower()
    
    # Licenses approved for redistribution
    approved = ['mit', 'apache-2.0', 'apache 2.0', 'bsd-3-clause', 'bsd', 'cc0-1.0']
    
    # Restrictive licenses
    restrictive = ['cc-by', 'gpl', 'lgpl', 'agpl', 'cc-by-sa']
    
    for approved_license in approved:
        if approved_license in license_lower:
            return "✅ Approved"
    
    for restrictive_license in restrictive:
        if restrictive_license in license_lower:
            return "⚠️ Restrictive"
    
    return "❓ Review Needed"

def analyze_mteb_data(csv_file: str) -> pd.DataFrame:
    """Analyze MTEB data and create RAG-focused leaderboard"""
    
    logger.info(f"Loading MTEB data from {csv_file}")
    
    try:
        df = pd.read_csv(csv_file)
        logger.info(f"Loaded {len(df)} models from CSV")
    except FileNotFoundError:
        logger.error(f"CSV file not found: {csv_file}")
        logger.error("Please download CSV from https://huggingface.co/spaces/mteb/leaderboard")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error reading CSV: {e}")
        return pd.DataFrame()
    
    # Show available columns
    logger.info(f"Available columns: {list(df.columns)}")
    
    # Check if license column exists
    has_license_info = 'License' in df.columns
    if not has_license_info:
        logger.warning("No license column found in CSV - license analysis will be limited")
    
    # Process each model
    processed_models = []
    skipped_count = 0
    
    for idx, row in df.iterrows():
        # Get basic model info
        model_name = row.get('Model')
        if pd.isna(model_name):
            skipped_count += 1
            continue
        
        # Parse parameters
        params_raw = row.get('Number of Parameters', 0)
        n_parameters = safe_numeric_convert(params_raw)
        
        # Parse other numeric fields
        max_tokens = safe_numeric_convert(row.get('Max Tokens', 0))
        memory_mb = safe_numeric_convert(row.get('Memory Usage (MB)', 0))
        embed_dim = safe_numeric_convert(row.get('Embedding Dimensions', 0))
        
        # Apply our filtering criteria
        meets_criteria = (
            n_parameters > 0 and                    # Has parameter info
            n_parameters <= 100_000_000 and        # ≤100M parameters  
            max_tokens >= 500 and                  # ≥500 token context
            memory_mb > 0                          # Has memory usage info
        )
        
        if not meets_criteria:
            skipped_count += 1
            continue
        
        # Calculate scores
        rag_score = calculate_rag_score(row)
        license_status = get_license_status(row.get('License'))
        
        processed_models.append({
            # Basic info
            'Model': model_name,
            'Parameters': f"{n_parameters/1_000_000:.0f}M",
            'Memory_MB': int(memory_mb),
            'Max_Tokens': int(max_tokens),
            'Embed_Dim': int(embed_dim) if embed_dim else 0,
            
            # License info
            'License': row.get('License', 'Unknown'),
            'License_Status': license_status,
            
            # Performance scores
            'RAG_Score': rag_score,
            'Retrieval': safe_numeric_convert(row.get('Retrieval', 0)),
            'STS': safe_numeric_convert(row.get('STS', 0)),  # Fixed column name
            'Reranking': safe_numeric_convert(row.get('Reranking', 0)),
            'Clustering': safe_numeric_convert(row.get('Clustering', 0)),
            'Classification': safe_numeric_convert(row.get('Classification', 0)),
            'Overall_MTEB': safe_numeric_convert(row.get('Average', 0)),
        })
    
    logger.info(f"Processed {len(processed_models)} models, skipped {skipped_count}")
    
    if not processed_models:
        logger.warning("No models found matching criteria!")
        return pd.DataFrame()
    
    # Create DataFrame and sort by RAG score
    result_df = pd.DataFrame(processed_models)
    result_df = result_df.sort_values('RAG_Score', ascending=False)
    
    return result_df

def generate_report(df: pd.DataFrame):
    """Generate comprehensive report"""
    
    if df.empty:
        print("❌ No data to analyze!")
        return
    
    print("\n" + "="*80)
    print("🎯 RAG-FOCUSED EMBEDDING MODEL ANALYSIS")
    print("="*80)
    
    # Summary statistics
    total_models = len(df)
    approved_models = len(df[df['License_Status'] == '✅ Approved'])
    high_performers = len(df[df['RAG_Score'] > 40])
    
    print(f"\n📊 DATASET SUMMARY:")
    print(f"   Total models analyzed: {total_models}")
    print(f"   Approved licenses: {approved_models}")
    print(f"   High RAG performers (>40): {high_performers}")
    print(f"   Average RAG score: {df['RAG_Score'].mean():.2f}")
    
    # Top performers
    print(f"\n🏆 TOP 10 MODELS BY RAG SCORE:")
    display_cols = ['Model', 'RAG_Score', 'Retrieval', 'STS', 'License_Status', 'Memory_MB']
    print(df[display_cols].head(10).to_string(index=False))
    
    # Check if we have license info
    approved_df = df[df['License_Status'] == '✅ Approved']
    if not approved_df.empty:
        print(f"\n✅ TOP 5 APPROVED LICENSE MODELS:")
        print(approved_df[display_cols].head(5).to_string(index=False))
        
        # Our recommendations
        print(f"\n🎯 FINAL RECOMMENDATIONS FOR RAG:")
        top3 = approved_df.head(3)
        for i, (_, model) in enumerate(top3.iterrows(), 1):
            print(f"\n{i}. {model['Model']}")
            print(f"   • RAG Score: {model['RAG_Score']} (Retrieval: {model['Retrieval']:.1f}, STS: {model['STS']:.1f})")
            print(f"   • Resources: {model['Memory_MB']}MB RAM, {model['Max_Tokens']} token context")
            print(f"   • License: {model['License']} ✅")
            
            # Comparison to our original recommendations
            if model['Model'] in ['intfloat/e5-small-v2', 'intfloat/e5-small', 'BAAI/bge-small-en-v1.5']:
                print(f"   • ✨ Matches our original analysis!")
    else:
        if len(df[df['License_Status'] == '❓ Unknown']) == len(df):
            print(f"\n⚠️  LICENSE INFO NOT AVAILABLE IN CSV")
            print(f"🎯 TOP 3 RAG RECOMMENDATIONS (License verification needed):")
            top3 = df.head(3)
            for i, (_, model) in enumerate(top3.iterrows(), 1):
                print(f"\n{i}. {model['Model']}")
                print(f"   • RAG Score: {model['RAG_Score']} (Retrieval: {model['Retrieval']:.1f}, STS: {model['STS']:.1f})")
                print(f"   • Resources: {model['Memory_MB']}MB RAM, {model['Max_Tokens']} token context")
                print(f"   • ⚠️  License: Verify manually at model's HuggingFace page")
                
                # Comparison to our original recommendations
                if any(orig in model['Model'] for orig in ['e5-small-v2', 'e5-small', 'bge-small-en-v1.5']):
                    print(f"   • ✨ Matches our original analysis (all MIT licensed)!")
            
            print(f"\n💡 Note: Based on previous analysis, e5-small-v2, e5-small, and bge-small-en-v1.5 are MIT licensed")
        else:
            print(f"\n❌ No approved license models found!")
    
    # License breakdown
    print(f"\n📋 LICENSE BREAKDOWN:")
    license_counts = df['License_Status'].value_counts()
    for status, count in license_counts.items():
        print(f"   {status}: {count} models")

def main():
    """Main execution"""
    
    # Determine CSV file
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        # Look for common filenames
        possible_files = ['mteb_data.csv', 'hugging_face_stats_2025_07_25.csv', 'mteb_leaderboard.csv']
        csv_file = None
        
        for filename in possible_files:
            if Path(filename).exists():
                csv_file = filename
                break
        
        if not csv_file:
            print("❌ No CSV file found!")
            print("\nTo get the CSV file:")
            print("1. Visit: https://huggingface.co/spaces/mteb/leaderboard")
            print("2. Apply filters (e.g., sub-100M parameters)")
            print("3. Click 'Download CSV'")
            print("4. Save as 'mteb_data.csv' in this directory")
            print("\nOr specify file: python3 analyze_mteb_csv.py your_file.csv")
            return
    
    # Analyze data
    df = analyze_mteb_data(csv_file)
    
    if df.empty:
        return
    
    # Save results
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M")
    output_file = f"rag_analysis_{timestamp}.csv"
    df.to_csv(output_file, index=False)
    logger.info(f"Results saved to {output_file}")
    
    # Generate report
    generate_report(df)
    
    print(f"\n💾 Full results saved to: {output_file}")

if __name__ == "__main__":
    main() 