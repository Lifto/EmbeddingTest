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
    python3 analyze_mteb_csv.py [csv_file] [--scrape-licenses]
    
    Arguments:
        csv_file: Path to MTEB CSV file (default: looks for 'mteb_data.csv')
        --scrape-licenses: Scrape licenses and language info for top models until 3 redistributable found (optimized for speed)
"""

import pandas as pd
import logging
import sys
import re
import requests
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from urllib.parse import urljoin

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

# License scraping functionality
LICENSE_CACHE = {}  # Cache to avoid duplicate requests

def extract_model_url(model_str: str) -> Optional[str]:
    """Extract HuggingFace URL from markdown-formatted model name"""
    # Pattern: [model-name](https://huggingface.co/path)
    match = re.search(r'\]\((https://huggingface\.co/[^)]+)\)', str(model_str))
    return match.group(1) if match else None

def scrape_model_info_from_hf(url: str) -> Dict[str, Optional[str]]:
    """Scrape license and language information from HuggingFace model page"""
    
    # Check cache first
    if url in LICENSE_CACHE:
        return LICENSE_CACHE[url]
    
    try:
        logger.info(f"Scraping model info from: {url}")
        
        # Add headers to appear as a regular browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Look for information in the HTML content
        html_content = response.text.lower()
        
        # Initialize result
        result = {'license': None, 'languages': None}
        
        # Common license patterns found on HuggingFace pages
        license_patterns = {
            'mit': ['mit license', 'mit'],
            'apache-2.0': ['apache-2.0', 'apache 2.0', 'apache license 2.0'],
            'bsd-3-clause': ['bsd-3-clause', 'bsd 3-clause'],
            'cc-by-4.0': ['cc-by-4.0', 'creative commons attribution 4.0'],
            'gpl-3.0': ['gpl-3.0', 'gnu general public license v3.0'],
            'other': ['other']
        }
        
        # Search for license mentions in the HTML
        for license_type, patterns in license_patterns.items():
            for pattern in patterns:
                if pattern in html_content:
                    result['license'] = license_type
                    logger.info(f"Found license: {license_type}")
                    break
            if result['license']:
                break
        
        # If no specific license found, try to find generic license mentions
        if not result['license'] and 'license' in html_content:
            result['license'] = 'unknown'
            logger.warning(f"License mentioned but not recognized for {url}")
        
        # Language detection patterns
        language_patterns = {
            'english': ['english', 'en', 'english texts', 'english only', 'works for english'],
            'multilingual': ['multilingual', 'multiple languages', 'multi-lingual', 'many languages'],
            'chinese': ['chinese', 'zh', 'mandarin', 'simplified chinese', 'traditional chinese'],
            'spanish': ['spanish', 'es', 'español'],
            'french': ['french', 'fr', 'français'],
            'german': ['german', 'de', 'deutsch'],
            'japanese': ['japanese', 'ja', '日本語'],
            'korean': ['korean', 'ko', '한국어'],
            'arabic': ['arabic', 'ar', 'العربية'],
            'russian': ['russian', 'ru', 'русский'],
            'portuguese': ['portuguese', 'pt', 'português'],
            'italian': ['italian', 'it', 'italiano'],
        }
        
        # Search for language mentions
        detected_languages = []
        for lang, patterns in language_patterns.items():
            for pattern in patterns:
                if pattern in html_content:
                    detected_languages.append(lang)
                    break
        
        # Special checks for common phrases
        if 'this model only works for english' in html_content:
            detected_languages = ['english-only']
        elif 'english texts' in html_content and 'only' in html_content:
            detected_languages = ['english-only']
        elif len(detected_languages) > 3:  # If too many languages detected, likely multilingual
            detected_languages = ['multilingual']
        
        if detected_languages:
            result['languages'] = ', '.join(detected_languages)
            logger.info(f"Found languages: {result['languages']}")
        else:
            # Try to infer from model tags or description
            if 'bert' in html_content and 'english' not in html_content:
                result['languages'] = 'likely-english'
            else:
                result['languages'] = 'unknown'
            logger.warning(f"No clear language information found for {url}")
        
        LICENSE_CACHE[url] = result
        time.sleep(1)  # Be respectful to HuggingFace servers
        return result
        
    except requests.RequestException as e:
        logger.error(f"Error scraping {url}: {e}")
        error_result = {'license': None, 'languages': None}
        LICENSE_CACHE[url] = error_result
        time.sleep(2)  # Wait longer after errors
        return error_result
    except Exception as e:
        logger.error(f"Unexpected error scraping {url}: {e}")
        error_result = {'license': None, 'languages': None}
        LICENSE_CACHE[url] = error_result
        time.sleep(2)
        return error_result

def scrape_license_from_hf(url: str) -> Optional[str]:
    """Backward compatibility wrapper for license scraping"""
    info = scrape_model_info_from_hf(url)
    return info.get('license')

def get_scraped_license_status(model_str: str, fallback_license: str = None) -> str:
    """Get license status by scraping HuggingFace page, with fallback"""
    
    # Extract URL from model string
    url = extract_model_url(model_str)
    if not url:
        return get_license_status(fallback_license)
    
    # Scrape license from HuggingFace page
    scraped_license = scrape_license_from_hf(url)
    
    # Use scraped license if available, otherwise fall back
    license_to_check = scraped_license if scraped_license else fallback_license
    return get_license_status(license_to_check)

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

def analyze_mteb_data(csv_file: str, scrape_licenses: bool = False) -> pd.DataFrame:
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
    if not has_license_info and not scrape_licenses:
        logger.warning("No license column found in CSV - license analysis will be limited")
        logger.info("Use --scrape-licenses flag to fetch license info from HuggingFace pages")
    elif scrape_licenses:
        logger.info("🔍 License & language scraping enabled - will scrape top models until 3 redistributable licenses found")
        logger.info("⚡ This optimized approach will be much faster!")
    
    # First pass: Process all models without license scraping
    logger.info("📊 First pass: Calculating RAG scores for all models...")
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
        
        processed_models.append({
            # Basic info
            'Model': model_name,
            'Parameters': f"{n_parameters/1_000_000:.0f}M",
            'Memory_MB': int(memory_mb),
            'Max_Tokens': int(max_tokens),
            'Embed_Dim': int(embed_dim) if embed_dim else 0,
            
            # License info (will be updated in second pass if scraping)
            'License': row.get('License', 'Unknown'),
            'License_Status': 'Pending' if scrape_licenses else get_license_status(row.get('License')),
            'License_Source': 'CSV Data',
            'Languages': 'Pending' if scrape_licenses else 'Unknown',  # Will be updated if scraping
            
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
    result_df = result_df.sort_values('RAG_Score', ascending=False).reset_index(drop=True)
    
    # Second pass: Scrape licenses for top models if requested
    if scrape_licenses:
        logger.info("🎯 Second pass: Scraping licenses for top models until 3 redistributable found...")
        
        redistributable_found = 0
        models_checked = 0
        
        for idx in range(len(result_df)):
            if redistributable_found >= 3:
                logger.info(f"✅ Found 3 redistributable licenses! Stopping scraping.")
                break
                
            model_name = result_df.loc[idx, 'Model']
            models_checked += 1
            
            print(f"🔍 [{models_checked}] Scraping license & language for {model_name}...")
            
            # Extract URL and scrape model info (license + language)
            url = extract_model_url(model_name)
            if url:
                model_info = scrape_model_info_from_hf(url)
                scraped_license = model_info.get('license')
                scraped_languages = model_info.get('languages')
                
                if scraped_license:
                    # Update license information
                    result_df.loc[idx, 'License'] = scraped_license
                    result_df.loc[idx, 'License_Status'] = get_license_status(scraped_license)
                    result_df.loc[idx, 'License_Source'] = 'Scraped from HF'
                else:
                    # Fallback to original license if scraping failed
                    original_license = result_df.loc[idx, 'License']
                    result_df.loc[idx, 'License_Status'] = get_license_status(original_license)
                    result_df.loc[idx, 'License_Source'] = 'CSV Data (scrape failed)'
                
                # Update language information
                if scraped_languages:
                    result_df.loc[idx, 'Languages'] = scraped_languages
                else:
                    result_df.loc[idx, 'Languages'] = 'Unknown'
            else:
                # No URL found, use original license and unknown languages
                original_license = result_df.loc[idx, 'License']
                result_df.loc[idx, 'License_Status'] = get_license_status(original_license)
                result_df.loc[idx, 'License_Source'] = 'CSV Data (no URL)'
                result_df.loc[idx, 'Languages'] = 'Unknown'
            
            # Check if this is redistributable
            current_license_status = result_df.loc[idx, 'License_Status']
            if current_license_status == "✅ Approved":
                redistributable_found += 1
                scraped_license_name = result_df.loc[idx, 'License']
                logger.info(f"🎉 Found redistributable model #{redistributable_found}: {model_name} ({scraped_license_name})")
        
        # Update remaining models to show they weren't scraped
        for idx in range(models_checked, len(result_df)):
            if result_df.loc[idx, 'License_Status'] == 'Pending':
                result_df.loc[idx, 'License_Status'] = get_license_status(result_df.loc[idx, 'License'])
            if result_df.loc[idx, 'Languages'] == 'Pending':
                result_df.loc[idx, 'Languages'] = 'Unknown'
        
        logger.info(f"📈 Scraping summary: Checked {models_checked} models, found {redistributable_found} redistributable")
    
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
    display_cols = ['Model', 'RAG_Score', 'Retrieval', 'STS', 'License_Status', 'Languages', 'Memory_MB']
    print(df[display_cols].head(10).to_string(index=False))
    
    # Check if we have license info
    approved_df = df[df['License_Status'] == '✅ Approved']
    if not approved_df.empty:
        print(f"\n✅ TOP 5 APPROVED LICENSE MODELS:")
        approved_display_cols = ['Model', 'RAG_Score', 'License', 'Languages', 'Memory_MB']
        print(approved_df[approved_display_cols].head(5).to_string(index=False))
        
        # Our recommendations
        print(f"\n🎯 FINAL RECOMMENDATIONS FOR RAG:")
        top3 = approved_df.head(3)
        for i, (_, model) in enumerate(top3.iterrows(), 1):
            print(f"\n{i}. {model['Model']}")
            print(f"   • RAG Score: {model['RAG_Score']} (Retrieval: {model['Retrieval']:.1f}, STS: {model['STS']:.1f})")
            print(f"   • Resources: {model['Memory_MB']}MB RAM, {model['Max_Tokens']} token context")
            print(f"   • License: {model['License']} ✅")
            print(f"   • Languages: {model['Languages']}")
            
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
                print(f"   • Languages: {model['Languages']}")
                
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
    
    # Parse command line arguments
    scrape_licenses = '--scrape-licenses' in sys.argv
    if scrape_licenses:
        sys.argv.remove('--scrape-licenses')
    
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
    df = analyze_mteb_data(csv_file, scrape_licenses=scrape_licenses)
    
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
    
    # Show license scraping summary if it was used
    if scrape_licenses:
        scraped_count = len([m for m in df['License_Source'] if m == 'Scraped from HF'])
        print(f"\n🔍 License Scraping Summary:")
        print(f"   Successfully scraped licenses for {scraped_count} models")
        print(f"   License cache contains {len(LICENSE_CACHE)} entries")

if __name__ == "__main__":
    main() 