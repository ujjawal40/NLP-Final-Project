import time
from dataloader import PubMedQADataLoader
from retriever import PubMedRetriever
from pathlib import Path


def main():
    print(" Starting index build process...")
    start_time = time.time()

    try:
        # Initialize components
        loader = PubMedQADataLoader()
        retriever = PubMedRetriever()

        print(" Loading dataset...")
        df = loader.load_parquets()
        print(f" Loaded {len(df)} records")

        print("Converting to documents...")
        documents = loader.to_documents(df)
        print(f" Created {len(documents)} document objects")

        print(" Building FAISS index...")
        retriever.build_index(documents)

        # Verify index was built
        index_path = Path("data/processed/vector_db.faiss")
        if index_path.exists():
            print(f" Success! Index built at {index_path}")
            print(f" Total time: {time.time() - start_time:.2f} seconds")
        else:
            raise FileNotFoundError("Index file not created")

    except Exception as e:
        print(f" Error during index build: {str(e)}")
        # Clean up partial files if needed
        if 'retriever' in locals() and hasattr(retriever, 'vector_db'):
            del retriever.vector_db
        raise


if __name__ == "__main__":
    main()