# rag_pipeline.py

import os
import pandas as pd
import ast
from retriever import PubMedRetriever
from Generator import MedicalGenerator

def safe_parse_context(context_str):
    if isinstance(context_str, list):
        return context_str
    try:
        return ast.literal_eval(context_str)
    except Exception:
        return []

def clean_prediction(text):
    text = text.lower()
    if "yes" in text:
        return "yes"
    elif "no" in text:
        return "no"
    elif "maybe" in text:
        return "maybe"
    else:
        return "unknown"

def main():
    retriever = PubMedRetriever()
    generator = MedicalGenerator()

    # Correct path handling
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(base_dir, "Dataset", "pubmedqa_labeled.parquet")

    # Load dataset
    df = pd.read_parquet(dataset_path)
    df['contexts'] = df['contexts'].apply(safe_parse_context)

    correct = 0
    total = len(df)

    for idx, row in df.iterrows():
        query = row['question']
        true_answer = row['answer'].lower()

        # Use contexts from dataset
        contexts = row['contexts']

        # Generate the answer
        answer = generator.generate(query, contexts)
        pred_cleaned = clean_prediction(answer)

        if pred_cleaned == true_answer:
            correct += 1

        if idx % 20 == 0:
            print(f"Processed {idx}/{total} | Current Accuracy: {correct/(idx+1):.4f}")

    final_accuracy = correct / total
    print(f"\nFinal Accuracy on PubMedQA: {final_accuracy:.4f}")

if __name__ == "__main__":
    main()
