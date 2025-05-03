Meditron RAG: Retrieval-Augmented Medical Question Answering
This project implements a Retrieval-Augmented Generation (RAG) pipeline for medical question answering, leveraging domain-adapted large language models (LLMs) and a hybrid retrieval system over PubMed and related datasets.
Features
Hybrid Retrieval: Combines dense (vector) and sparse (BM25) retrieval for relevant medical context.
Reranking: Uses a cross-encoder to rerank retrieved contexts.
Domain LLM: Uses Meditron-7B for answer generation.
Batch and Interactive Modes: Evaluate on datasets or ask questions interactively.
Performance Visualization: Track retrieval and generation metrics.

git clone <your-repo-url>
cd Meditron
