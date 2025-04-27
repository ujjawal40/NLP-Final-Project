import time
from pathlib import Path
from dataloader import PubMedQADataLoader
from retriever import PubMedRetriever


def main():
    try:
        print("Starting index build process...")
        start_time = time.time()

        # Initialize loader with exact paths
        loader = PubMedQADataLoader()

        # Load data
        print("Loading dataset...")
        df = loader.load_parquets()
        print(f"Successfully loaded {len(df)} records")

        # Convert to documents
        print("Converting to documents...")
        documents = loader.to_documents(df)
        print(f"Created {len(documents)} document objects")

        # Build index
        print("Building FAISS index...")
        retriever = PubMedRetriever({
            "index_path": str(Path.home() / "NLP-Final-Project" / "Dataset" / "processed" / "vector_db"),
            "embedding_model": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
        })
        retriever.build_index(documents)

        print(f"✅ Build completed in {time.time() - start_time:.2f} seconds")

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise


if __name__ == "__main__":
    main()