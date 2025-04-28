# test_retrieval.py
from retriever import PubMedRetriever

retriever = PubMedRetriever()
results = retriever.retrieve("COVID-19 symptoms", k=2)

for i, doc in enumerate(results):
    print(f"Result {i+1}:")
    print(doc.page_content[:200] + "...")
    print(f"Metadata: {doc.metadata}\n")