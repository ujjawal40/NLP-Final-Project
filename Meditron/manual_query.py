import logging
from retriever import PubMedRetriever
from Generator import MedicalGenerator
from typing import List, Dict, Any
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


def format_context(context: str, max_length: int = 200) -> str:
    """Format context for display."""
    if len(context) > max_length:
        return context[:max_length] + "..."
    return context


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Initialize components
    logger.info("Initializing RAG components...")
    retriever = PubMedRetriever()
    generator = MedicalGenerator()

    logger.info("RAG pipeline ready!")
    logger.info("Type 'exit' to quit")
    logger.info("Type 'help' for example questions")

    while True:
        try:
            # Get user query
            query = input("\nEnter your medical question: ").strip()

            if query.lower() == 'exit':
                break
            elif query.lower() == 'help':
                print("\nExample questions:")
                print("1. Do mitochondria play a role in remodelling lace plant leaves during programmed cell death?")
                print("2. Is there a relationship between sleep duration and cardiovascular disease?")
                print("3. Does vitamin D supplementation reduce the risk of COVID-19 infection?")
                continue
            elif not query:
                print("Please enter a question!")
                continue

            # Retrieve relevant contexts
            logger.info("Retrieving relevant contexts...")
            retrieved_results = retriever.retrieve(query, top_k=5)

            if not retrieved_results:
                logger.warning("No relevant contexts found!")
                continue

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

            # Log retrieval results
            logger.info(f"\nRetrieved {len(retrieved_contexts)} contexts:")
            for i, (context, score) in enumerate(zip(retrieved_contexts, scores)):
                logger.info(f"\nContext {i + 1} (Relevance Score: {score:.4f}):")
                logger.info(format_context(context))

            # Generate answer
            logger.info("\nGenerating answer...")
            answer = generator.generate(query, retrieved_contexts)
            pred_cleaned = clean_prediction(answer)

            # Print results
            print("\n" + "=" * 50)
            print(f"QUESTION: {query}")
            print(f"ANSWER: {answer}")
            print(f"CLEANED ANSWER: {pred_cleaned}")
            print("=" * 50)

            # Provide feedback option
            feedback = input("\nWas this answer helpful? (y/n): ").lower()
            if feedback == 'n':
                logger.info("Thank you for your feedback. We'll use this to improve the system.")

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            continue


if __name__ == "__main__":
    main()
