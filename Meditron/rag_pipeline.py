import os
import pandas as pd
import ast
from retriever import PubMedRetriever
from Generator import MedicalGenerator
from typing import List, Dict, Any
import logging
from tqdm import tqdm
from sklearn.metrics import classification_report
import torch
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def safe_parse_context(context_str):
    """Safely parse context string to list."""
    if isinstance(context_str, list):
        return context_str
    try:
        return ast.literal_eval(context_str)
    except Exception:
        return []


def clean_prediction(text: str) -> str:
    """Clean and standardize the prediction text."""
    text = text.lower().strip()
    if "yes" in text:
        return "yes"
    elif "no" in text:
        return "no"
    elif "maybe" in text:
        return "maybe"
    else:
        return "unknown"


def evaluate_predictions(true_labels: List[str], predictions: List[str]) -> Dict[str, float]:
    """Calculate evaluation metrics."""
    report = classification_report(true_labels, predictions, output_dict=True)
    return {
        'accuracy': report['accuracy'],
        'precision': report['weighted avg']['precision'],
        'recall': report['weighted avg']['recall'],
        'f1': report['weighted avg']['f1-score']
    }


def process_example(args):
    """Process a single example with the RAG pipeline."""
    idx, row, retriever, generator = args
    try:
        query = row['question']
        true_answer = row['answer'].lower()

        # Retrieve relevant contexts
        retrieved_results = retriever.retrieve(query, top_k=5)

        # Extract contexts and scores
        retrieved_contexts = []
        scores = []
        for result in retrieved_results:
            retrieved_contexts.append(result['text'])
            if 'score' in result:
                scores.append(result['score'])
            elif 'rerank_score' in result:
                scores.append(result['rerank_score'])
            else:
                scores.append(0.0)

        # Generate answer
        answer = generator.generate(query, retrieved_contexts)
        pred_cleaned = clean_prediction(answer)

        return {
            'idx': idx,
            'question': query,
            'true_answer': true_answer,
            'predicted_answer': pred_cleaned,
            'retrieved_contexts': retrieved_contexts,
            'retrieval_scores': scores,
            'dataset': row.get('dataset', 'unknown')  # Track which dataset this came from
        }
    except Exception as e:
        logger.error(f"Error processing example {idx}: {str(e)}")
        return None


def load_and_prepare_dataset(base_dir: str, dataset_name: str, max_examples: int = None) -> pd.DataFrame:
    """Load and prepare a dataset."""
    dataset_path = os.path.join(base_dir, "Dataset", f"pubmedqa_{dataset_name}.parquet")
    logger.info(f"Loading {dataset_name} dataset from {dataset_path}")

    try:
        df = pd.read_parquet(dataset_path)
        
        # Limit number of examples if specified
        if max_examples is not None:
            df = df.head(max_examples)
            logger.info(f"Limited to {max_examples} examples")
        
        df['dataset'] = dataset_name

        # Handle answer column
        if 'final_decision' in df.columns:
            df['answer'] = df['final_decision']
        elif 'label' in df.columns:
            df['answer'] = df['label']
        else:
            raise ValueError(f"No answer column found in {dataset_name} dataset")

        # Handle context column
        if 'context' in df.columns:
            df['context'] = df['context'].apply(
                lambda x: x['contexts'] if isinstance(x, dict) and 'contexts' in x else [])
        else:
            df['context'] = [[] for _ in range(len(df))]

        return df
    except Exception as e:
        logger.error(f"Failed to load {dataset_name} dataset: {str(e)}")
        raise


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run RAG pipeline on PubMedQA datasets')
    parser.add_argument('--datasets', nargs='+', choices=['labeled', 'artificial', 'all'], 
                      default=['all'], help='Datasets to process')
    parser.add_argument('--max-examples', type=int, help='Maximum number of examples to process per dataset')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of worker threads')
    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Initialize components
    retriever = PubMedRetriever()
    generator = MedicalGenerator()

    # Set up paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    # Load datasets
    try:
        datasets_to_load = ['labeled', 'artificial'] if 'all' in args.datasets else args.datasets
        dfs = []
        
        for dataset_name in datasets_to_load:
            df = load_and_prepare_dataset(base_dir, dataset_name, args.max_examples)
            dfs.append(df)
        
        # Combine datasets
        df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Total examples: {len(df)}")
        for dataset_name in datasets_to_load:
            count = len(df[df['dataset'] == dataset_name])
            logger.info(f"- {dataset_name}: {count} examples")
        
    except Exception as e:
        logger.error(f"Failed to load datasets: {str(e)}")
        raise

    # Process examples in parallel
    logger.info("Processing examples...")
    num_workers = min(args.num_workers, multiprocessing.cpu_count())
    logger.info(f"Using {num_workers} workers")

    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Create arguments for each example
        args_list = [(idx, row, retriever, generator) for idx, row in df.iterrows()]
        
        # Submit all tasks
        future_to_idx = {executor.submit(process_example, arg): arg[0] for arg in args_list}
        
        # Process results as they complete
        for future in tqdm(as_completed(future_to_idx), total=len(args_list), desc="Processing examples"):
            result = future.result()
            if result is not None:
                results.append(result)
                
                # Log progress
                if len(results) % 20 == 0:
                    current_predictions = [r['predicted_answer'] for r in results]
                    current_true_labels = [r['true_answer'] for r in results]
                    current_metrics = evaluate_predictions(current_true_labels, current_predictions)
                    logger.info(f"Processed {len(results)}/{len(df)} | Current Metrics: {current_metrics}")

    # Calculate final metrics
    if not results:
        logger.error("No predictions were generated!")
        return

    # Calculate overall metrics
    final_predictions = [r['predicted_answer'] for r in results]
    final_true_labels = [r['true_answer'] for r in results]
    final_metrics = evaluate_predictions(final_true_labels, final_predictions)
    
    logger.info("\nFinal Evaluation Metrics (Overall):")
    for metric, value in final_metrics.items():
        logger.info(f"{metric}: {value:.4f}")

    # Calculate metrics per dataset
    for dataset_name in datasets_to_load:
        dataset_results = [r for r in results if r['dataset'] == dataset_name]
        if dataset_results:
            dataset_predictions = [r['predicted_answer'] for r in dataset_results]
            dataset_true_labels = [r['true_answer'] for r in dataset_results]
            dataset_metrics = evaluate_predictions(dataset_true_labels, dataset_predictions)
            
            logger.info(f"\nMetrics for {dataset_name} dataset:")
            for metric, value in dataset_metrics.items():
                logger.info(f"{metric}: {value:.4f}")

    # Save detailed results
    results_df = pd.DataFrame(results)
    results_path = os.path.join(results_dir, "evaluation_results.csv")
    results_df.to_csv(results_path, index=False)
    logger.info(f"Results saved to {results_path}")

    # Save metrics
    metrics = {
        'overall': final_metrics,
        **{dataset: evaluate_predictions(
            [r['true_answer'] for r in results if r['dataset'] == dataset],
            [r['predicted_answer'] for r in results if r['dataset'] == dataset]
        ) for dataset in datasets_to_load}
    }
    
    metrics_path = os.path.join(results_dir, "metrics.json")
    pd.Series(metrics).to_json(metrics_path)
    logger.info(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
