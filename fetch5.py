#!/usr/bin/env python3
"""
MTEB data fetch with metadata filtering
Stage 1: Filter and show acceptable models
"""

import mteb
import pprint
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple

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

class RAGScorer:
    """RAG-focused scoring system using consistent tasks across all models"""
    
    def __init__(self):
        # Use only the 37 consistent RAG-relevant tasks found across all models
        self.consistent_tasks = {
            'retrieval': [
                'ArguAna', 'CQADupstackAndroidRetrieval', 'CQADupstackEnglishRetrieval',
                'CQADupstackGamingRetrieval', 'CQADupstackGisRetrieval', 'CQADupstackMathematicaRetrieval',
                'CQADupstackPhysicsRetrieval', 'CQADupstackProgrammersRetrieval', 'CQADupstackStatsRetrieval',
                'CQADupstackTexRetrieval', 'CQADupstackUnixRetrieval', 'CQADupstackWebmastersRetrieval',
                'CQADupstackWordpressRetrieval', 'FiQA2018', 'HotpotQA', 'HotpotQAHardNegatives',
                'MSMARCO', 'NFCorpus', 'QuoraRetrieval', 'SCIDOCS', 'SciFact', 'TRECCOVID',
                'Touche2020Retrieval.v3'
            ],
            'sts': [
                'BIOSSES', 'SICK-R', 'STS12', 'STS13', 'STS14', 'STS15', 'STS16',
                'STS17', 'STS22', 'STS22.v2', 'STSBenchmark'
            ],
            'reranking': [
                'MindSmallReranking', 'SciDocsRR'
            ]
        }
        
        # Weights for each category
        self.task_weights = {
            'retrieval': 0.60,  # Highest priority for RAG
            'sts': 0.30,        # Second priority
            'reranking': 0.10   # Nice to have
        }
        
        # Minimum requirements (must have scores in these categories)
        self.min_requirements = {
            'retrieval': 10,  # Need at least 10 retrieval scores for reliability
            'sts': 5,         # Need at least 5 STS scores
            'reranking': 0    # Reranking is optional
        }
        
        # Minimum total tasks required
        self.min_total_tasks = 15
    
    def calculate_category_score(self, df: pd.DataFrame, model: str, task_list: List[str]) -> Tuple[Optional[float], int]:
        """Calculate average score for a category and return count of available tasks"""
        if not task_list:
            return None, 0
        
        scores = []
        for task in task_list:
            if task in df.index and model in df.columns and not pd.isna(df.loc[task, model]):
                scores.append(float(df.loc[task, model]))
        
        avg_score = np.mean(scores) if scores else None
        return avg_score, len(scores)
    
    def is_model_valid(self, category_scores: Dict[str, Tuple[Optional[float], int]]) -> Tuple[bool, str]:
        """Check if model has sufficient consistent task data for fair RAG scoring"""
        
        # Check minimum requirements for each category
        for category, (score, count) in category_scores.items():
            min_required = self.min_requirements[category]
            if count < min_required:
                return False, f"Insufficient {category} tasks ({count} < {min_required} from consistent set)"
        
        # Check total task count
        total_tasks = sum(count for _, count in category_scores.values())
        if total_tasks < self.min_total_tasks:
            return False, f"Insufficient total consistent tasks ({total_tasks} < {self.min_total_tasks})"
        
        # Must have retrieval scores (core requirement for RAG)
        if category_scores['retrieval'][0] is None:
            return False, "No retrieval scores available from consistent task set"
        
        return True, "Valid"
    
    def calculate_rag_score(self, df: pd.DataFrame, model: str) -> Dict:
        """Calculate RAG score using only consistent tasks"""
        
        if model not in df.columns:
            return {
                'model': model,
                'rag_score': None,
                'status': 'Model not found in results',
                'task_breakdown': {},
                'task_counts': {}
            }
        
        # Calculate scores for each category using consistent tasks only
        category_scores = {}
        for category, task_list in self.consistent_tasks.items():
            score, count = self.calculate_category_score(df, model, task_list)
            category_scores[category] = (score, count)
        
        # Check if model meets minimum requirements
        is_valid, status = self.is_model_valid(category_scores)
        
        if not is_valid:
            return {
                'model': model,
                'rag_score': None,
                'status': status,
                'task_breakdown': {cat: score for cat, (score, _) in category_scores.items()},
                'task_counts': {cat: count for cat, (_, count) in category_scores.items()}
            }
        
        # Calculate weighted RAG score
        rag_score = 0.0
        total_weight = 0.0
        
        for category, (score, count) in category_scores.items():
            if score is not None:
                weight = self.task_weights[category]
                rag_score += score * weight
                total_weight += weight
        
        # Normalize by actual weights used
        if total_weight > 0:
            rag_score = rag_score / total_weight
        
        return {
            'model': model,
            'rag_score': round(rag_score, 4),
            'status': 'Valid - Fair Comparison',
            'task_breakdown': {cat: round(score, 4) if score else None for cat, (score, _) in category_scores.items()},
            'task_counts': {cat: count for cat, (_, count) in category_scores.items()}
        }
    
    def score_all_models(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score all models using consistent task methodology"""
        results = []
        
        # Print diagnostic info
        total_consistent = sum(len(tasks) for tasks in self.consistent_tasks.values())
        print(f"📊 Using Fair Comparison Method:")
        print(f"   • {len(self.consistent_tasks['retrieval'])} consistent retrieval tasks")
        print(f"   • {len(self.consistent_tasks['sts'])} consistent STS tasks") 
        print(f"   • {len(self.consistent_tasks['reranking'])} consistent reranking tasks")
        print(f"   • {total_consistent} total consistent RAG tasks")
        print()
        
        for model in df.columns:
            result = self.calculate_rag_score(df, model)
            results.append({
                'Model': result['model'],
                'RAG_Score': result['rag_score'],
                'Status': result['status'],
                'Retrieval_Score': result['task_breakdown'].get('retrieval'),
                'STS_Score': result['task_breakdown'].get('sts'),
                'Reranking_Score': result['task_breakdown'].get('reranking'),
                'Retrieval_Tasks': result['task_counts'].get('retrieval', 0),
                'STS_Tasks': result['task_counts'].get('sts', 0),
                'Reranking_Tasks': result['task_counts'].get('reranking', 0),
            })
        
        # Create DataFrame and sort by RAG score
        results_df = pd.DataFrame(results)
        
        # Separate valid and invalid models
        valid_models = results_df[results_df['Status'].str.contains('Valid', na=False)].copy()
        invalid_models = results_df[~results_df['Status'].str.contains('Valid', na=False)].copy()
        
        # Sort valid models by RAG score (descending)
        if not valid_models.empty:
            valid_models = valid_models.sort_values('RAG_Score', ascending=False)
        
        # Combine: valid models first, then invalid
        final_df = pd.concat([valid_models, invalid_models], ignore_index=True)
        
        return final_df

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
    
    # Stage 2: Fetch DataFrame for acceptable models only
    if acceptable_metas:
        print(f"\n🔍 Stage 2: Fetching scores DataFrame...")
        
        # Get model names for filtering
        acceptable_model_names = [meta.name for meta in acceptable_metas]
        print(f"Looking for scores for these {len(acceptable_model_names)} models...")
        
        # Fetch results DataFrame for ONLY our acceptable models
        print("Loading MTEB results for filtered models only...")
        results = mteb.load_results(models=acceptable_model_names)
        results_df = results.to_dataframe()
        print(f"Filtered DataFrame shape: {results_df.shape} (tasks x models)")
        
        # Check which models actually have results
        available_models = list(results_df.columns)
        missing_models = [name for name in acceptable_model_names if name not in available_models]
        
        print(f"\n📊 MODEL AVAILABILITY:")
        print(f"✅ Available in results: {len(available_models)}")
        print(f"❌ Missing from results: {len(missing_models)}")
        
        if missing_models:
            print(f"\n❌ Missing models:")
            for model in missing_models:
                print(f"   - {model}")
        
        if available_models:
            # Fix DataFrame structure - use task_name column as index
            if 'task_name' in results_df.columns:
                print(f"\n🔧 Restructuring DataFrame - using task_name as index...")
                
                # Set task_name as index and drop the task_name column
                results_df = results_df.set_index('task_name')
                
                # Update available_models to exclude 'task_name'
                available_models = [col for col in results_df.columns]
                
                print(f"\n📋 CORRECTED RESULTS DATAFRAME:")
                print(f"Shape: {results_df.shape} (tasks x models)")
                print(f"Models ({len(available_models)}): {available_models}")
                print(f"Tasks (first 10): {list(results_df.index[:10])}")
                
                print(f"\n🔍 RAW DATAFRAME DATA:")
                print("=" * 100)
                print(results_df)
                print("=" * 100)
                
                # Show some sample data
                print(f"\n📋 SAMPLE SCORES:")
                for model in available_models[:3]:  # Show first 3 models
                    print(f"\n{model}:")
                    model_scores = results_df[model].dropna()
                    for task_name, score in model_scores.head(10).items():
                        print(f"   {task_name}: {score}")
                
                # Stage 3: Apply RAG scoring
                print(f"\n🎯 Stage 3: Applying RAG-focused scoring system...")
                
                scorer = RAGScorer()
                rag_results = scorer.score_all_models(results_df)
                
                print(f"\n📊 RAG SCORING RESULTS:")
                print("=" * 100)
                
                # Show valid models
                valid_models = rag_results[rag_results['Status'].str.contains('Valid', na=False)]
                if not valid_models.empty:
                    print(f"\n✅ VALID MODELS WITH RAG SCORES ({len(valid_models)}):")
                    display_cols = ['Model', 'RAG_Score', 'Retrieval_Score', 'STS_Score', 'Reranking_Score', 'Retrieval_Tasks', 'STS_Tasks']
                    print(valid_models[display_cols].to_string(index=False))
                    
                    print(f"\n🏆 TOP 3 RAG RECOMMENDATIONS:")
                    for i, (_, model) in enumerate(valid_models.head(3).iterrows(), 1):
                        print(f"\n{i}. {model['Model']}")
                        print(f"   • RAG Score: {model['RAG_Score']}")
                        print(f"   • Breakdown: Retrieval={model['Retrieval_Score']:.3f}, STS={model['STS_Score']:.3f}, Reranking={model['Reranking_Score'] or 'N/A'}")
                        print(f"   • Task Coverage: {model['Retrieval_Tasks']} retrieval, {model['STS_Tasks']} STS, {model['Reranking_Tasks']} reranking")
                else:
                    print(f"\n❌ No models meet RAG scoring requirements!")
                
                # Show invalid models
                invalid_models = rag_results[~rag_results['Status'].str.contains('Valid', na=False)]
                if not invalid_models.empty:
                    print(f"\n⚠️  INVALID MODELS ({len(invalid_models)}):")
                    for _, model in invalid_models.iterrows():
                        print(f"   • {model['Model']}: {model['Status']}")
                
                print(f"\n💾 Saving RAG results...")
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y_%m_%d_%H%M")
                output_file = f"rag_results_{timestamp}.csv"
                rag_results.to_csv(output_file, index=False)
                print(f"   Results saved to: {output_file}")
                
            else:
                print(f"\n⚠️  No 'task_name' column found - DataFrame structure may be different")
                print(f"Available columns: {list(results_df.columns)}")
        else:
            print("\n❌ No acceptable models found in results DataFrame!")

if __name__ == "__main__":
    main() 