from rag_pipeline import PubMedRAG

rag = PubMedRAG()
test_cases = [
    "What are first-line treatments for hypertension?",
    "How does metformin affect blood sugar?",
    "What is the mortality rate of COVID-19?"
]

for question in test_cases:
    print(f"\n{' TESTING ':=^80}")
    print(f"QUESTION: {question}")
    result = rag(question)
    print(f"ANSWER: {result['answer']}")
    print(f"SOURCES: {[s['source'] for s in result['sources']]}")
    print("="*80)