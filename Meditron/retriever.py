# retriever.py

import torch
from pathlib import Path
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import numpy as np
from typing import List, Dict, Any
import nltk
from nltk.corpus import wordnet
import logging
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PubMedRetriever:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Load paths from yaml
        with open('path.yaml', 'r') as f:
            paths = yaml.safe_load(f)

        # Set up paths
        project_root = Path(__file__).parent.parent
        dataset_dir = project_root / "Dataset" / "processed" / "contexts_dataset"
        index_path = project_root / "Dataset" / "processed" / "vector_db" / "index.faiss"

        logger.info(f"Project root: {project_root}")
        logger.info(f"Dataset directory: {dataset_dir}")
        logger.info(f"Index path: {index_path}")

        try:
            self.dataset = load_from_disk(str(dataset_dir))
            self.dataset.load_faiss_index("embeddings", str(index_path))
            logger.info("Dataset and index loaded successfully")
        except Exception as e:
            logger.error(f"Error loading dataset or index: {str(e)}")
            raise

        # Initialize models - using medical-specific model
        self.embed_model = SentenceTransformer(
            "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
            device=self.device
        )

        # Initialize reranker with medical model
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=self.device)

        # Initialize BM25
        self._initialize_bm25()

        # Download NLTK data for query expansion
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')

    def _initialize_bm25(self):
        """Initialize BM25 with the dataset texts."""
        texts = self.dataset['text']
        tokenized_texts = [text.lower().split() for text in texts]
        self.bm25 = BM25Okapi(tokenized_texts)

    def _expand_query(self, query: str) -> str:
        """Expand query using WordNet synonyms."""
        words = query.lower().split()
        expanded_words = []

        # Only expand medical terms
        medical_terms = {
            'cancer': ['tumor', 'neoplasm', 'malignancy', 'carcinoma'],
            'hypertension': ['high blood pressure', 'elevated blood pressure'],
            'diabetes': ['diabetes mellitus', 'high blood sugar'],
            'covid': ['coronavirus', 'sars-cov-2', 'covid-19'],
            'heart': ['cardiac', 'cardiovascular'],
            'lung': ['pulmonary', 'respiratory'],
            'brain': ['cerebral', 'neurological'],
            'liver': ['hepatic'],
            'kidney': ['renal'],
            'blood': ['hematological', 'hematologic'],
            'bone': ['skeletal', 'osseous'],
            'muscle': ['muscular', 'myogenic'],
            'joint': ['articular'],
            'skin': ['dermal', 'cutaneous'],
            'eye': ['ocular', 'ophthalmic'],
            'ear': ['aural', 'otologic'],
            'nose': ['nasal', 'rhinologic'],
            'throat': ['pharyngeal', 'laryngeal'],
            'stomach': ['gastric'],
            'intestine': ['intestinal', 'enteric'],
            'pancreas': ['pancreatic'],
            'thyroid': ['thyroidal'],
            'adrenal': ['adrenocortical'],
            'pituitary': ['hypophyseal'],
            'immune': ['immunological', 'immunologic'],
            'infection': ['infectious', 'contagious'],
            'virus': ['viral', 'viral infection'],
            'bacteria': ['bacterial', 'bacterial infection'],
            'fungus': ['fungal', 'fungal infection'],
            'parasite': ['parasitic', 'parasitic infection'],
            'inflammation': ['inflammatory'],
            'pain': ['ache', 'discomfort'],
            'fever': ['pyrexia', 'hyperthermia'],
            'fatigue': ['tiredness', 'exhaustion'],
            'nausea': ['queasiness', 'sickness'],
            'vomiting': ['emesis', 'throwing up'],
            'diarrhea': ['loose stools', 'bowel movement'],
            'constipation': ['bowel obstruction', 'irregularity'],
            'headache': ['cephalgia', 'head pain'],
            'dizziness': ['vertigo', 'lightheadedness'],
            'seizure': ['convulsion', 'epileptic'],
            'stroke': ['cerebrovascular accident', 'brain attack'],
            'heart attack': ['myocardial infarction', 'cardiac arrest'],
            'asthma': ['bronchial asthma', 'reactive airway'],
            'arthritis': ['joint inflammation', 'rheumatism'],
            'osteoporosis': ['bone loss', 'bone thinning'],
            'depression': ['major depressive disorder', 'clinical depression'],
            'anxiety': ['anxiety disorder', 'nervousness'],
            'schizophrenia': ['psychotic disorder', 'mental illness'],
            'autism': ['autism spectrum disorder', 'ASD'],
            'alzheimer': ['dementia', 'cognitive decline'],
            'parkinson': ['parkinsonism', 'movement disorder']
        }

        for word in words:
            expanded_words.append(word)
            if word in medical_terms:
                expanded_words.extend(medical_terms[word])

        return " ".join(expanded_words)

    def _hybrid_search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """Perform hybrid search using both dense and sparse retrieval."""
        try:
            # Dense retrieval
            encoded_input = self.embed_model.encode(
                [query],
                convert_to_tensor=True,
                device=self.device,
                normalize_embeddings=True
            ).cpu().numpy()

            # Get nearest examples
            scores, examples = self.dataset.get_nearest_examples_batch(
                "embeddings",
                queries=encoded_input,
                k=top_k,
            )

            # Log the structure of examples
            logger.info(f"Examples type: {type(examples)}")
            logger.info(f"Examples structure: {examples.keys() if hasattr(examples, 'keys') else 'No keys'}")

            # Sparse retrieval (BM25)
            tokenized_query = query.lower().split()
            bm25_scores = self.bm25.get_scores(tokenized_query)
            top_bm25_indices = np.argsort(bm25_scores)[-top_k:][::-1]

            # Combine results
            combined_results = []
            seen_indices = set()

            # Add dense results
            for i, (score, example) in enumerate(zip(scores[0], examples[0])):
                if i not in seen_indices:
                    # Safely get text from example
                    text = example.get('text', '') if isinstance(example, dict) else str(example)
                    combined_results.append({
                        'text': text,
                        'score': float(score),
                        'source': 'dense'
                    })
                    seen_indices.add(i)

            # Add BM25 results
            for idx in top_bm25_indices:
                if idx not in seen_indices:
                    combined_results.append({
                        'text': self.dataset['text'][idx],
                        'score': float(bm25_scores[idx]),
                        'source': 'sparse'
                    })
                    seen_indices.add(idx)

            return combined_results
        except Exception as e:
            logger.error(f"Error in hybrid search: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return []

    def _rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """Rerank candidates using cross-encoder."""
        if not candidates:
            return []

        # Prepare pairs for reranking
        pairs = [(query, candidate['text']) for candidate in candidates]

        # Get reranking scores
        rerank_scores = self.reranker.predict(pairs)

        # Combine original scores with reranking scores
        for candidate, score in zip(candidates, rerank_scores):
            candidate['rerank_score'] = float(score)
            # Adjust weights to favor reranking more
            candidate['final_score'] = 0.8 * candidate['rerank_score'] + 0.2 * candidate['score']

        # Sort by final score
        reranked = sorted(candidates, key=lambda x: x['final_score'], reverse=True)
        return reranked[:top_k]

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Enhanced retrieval with query expansion, hybrid search, and reranking.

        Args:
            query: The search query
            top_k: Number of results to return

        Returns:
            List of dictionaries containing retrieved texts and their scores
        """
        # Expand query
        expanded_query = self._expand_query(query)
        logger.info(f"Original query: {query}")
        logger.info(f"Expanded query: {expanded_query}")

        # Perform hybrid search
        candidates = self._hybrid_search(expanded_query, top_k=20)

        # Rerank results
        final_results = self._rerank(query, candidates, top_k)

        return final_results
