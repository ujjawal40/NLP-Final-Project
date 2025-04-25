import pandas as pd
from Generator import MeditronGenerator
from retriever import PubMedRetriever
from config import Config


class PubMedQAEvaluator:
    def __init__(self):
        self.config = Config()
        self.generator = MeditronGenerator()
        self.retriever = PubMedRetriever()

    def load_test_data(self):
        """Load labeled dataset for evaluation"""
        return pd.read_parquet(self.config.labeled_path)

    def evaluate_accuracy(self, sample_size=100):
        """Calculate accuracy on sample questions"""
        df = self.load_test_data().sample(sample_size)
        correct = 0

        for _, row in df.iterrows():
            contexts = self.retriever.search(row['question'])
            answer = self.generator.generate_answer(row['question'], contexts)

            if str(row['final_decision']).lower() in answer.lower():
                correct += 1

        accuracy = correct / sample_size
        print(f"Accuracy: {accuracy:.2%}")
        return accuracy


# Example usage
if __name__ == "__main__":
    evaluator = PubMedQAEvaluator()
    evaluator.evaluate_accuracy(50)  # Quick test with 50 samples