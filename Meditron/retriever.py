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

        # Comprehensive medical terms dictionary
        medical_terms = {
            # General Medical Terms
            'disease': ['disorder', 'condition', 'syndrome', 'illness', 'ailment'],
            'treatment': ['therapy', 'intervention', 'management', 'regimen', 'protocol'],
            'diagnosis': ['assessment', 'evaluation', 'identification', 'detection'],
            'symptom': ['manifestation', 'indication', 'sign', 'clinical feature'],
            'patient': ['subject', 'individual', 'case', 'person'],
            'clinical': ['medical', 'therapeutic', 'healthcare'],
            'medical': ['clinical', 'therapeutic', 'healthcare'],
            'health': ['wellness', 'wellbeing', 'physical condition'],
            
            # Anatomy and Physiology
            'cell': ['cellular', 'cytological', 'tissue unit'],
            'tissue': ['cellular structure', 'biological material'],
            'organ': ['anatomical structure', 'body part'],
            'system': ['physiological system', 'bodily system'],
            'function': ['physiological role', 'biological activity'],
            'structure': ['anatomical organization', 'morphology'],
            'development': ['growth', 'maturation', 'progression'],
            'regulation': ['control', 'modulation', 'homeostasis'],
            
            # Molecular Biology
            'gene': ['genetic sequence', 'DNA segment', 'genomic region'],
            'protein': ['polypeptide', 'amino acid chain', 'molecular structure'],
            'enzyme': ['catalyst', 'biological catalyst', 'protein catalyst'],
            'receptor': ['binding site', 'molecular target', 'signal transducer'],
            'pathway': ['signaling cascade', 'metabolic route', 'biological process'],
            'mutation': ['genetic change', 'sequence alteration', 'genomic variation'],
            'expression': ['gene activity', 'transcription', 'translation'],
            'regulation': ['control', 'modulation', 'homeostasis'],
            
            # Immunology
            'immune': ['immunological', 'immunologic', 'defense system'],
            'antibody': ['immunoglobulin', 'immune protein', 'defense molecule'],
            'antigen': ['immune target', 'foreign substance', 'immune trigger'],
            'inflammation': ['inflammatory response', 'immune reaction'],
            'infection': ['pathogenic invasion', 'disease transmission'],
            'vaccine': ['immunization', 'preventive treatment'],
            'immunity': ['immune protection', 'defense mechanism'],
            'autoimmune': ['self-immune', 'auto-reactive'],
            
            # Pharmacology
            'drug': ['medication', 'pharmaceutical', 'therapeutic agent'],
            'therapy': ['treatment', 'intervention', 'management'],
            'dose': ['dosage', 'amount', 'concentration'],
            'administration': ['delivery', 'application', 'introduction'],
            'pharmacokinetics': ['drug metabolism', 'drug disposition'],
            'pharmacodynamics': ['drug action', 'drug effect'],
            'side effect': ['adverse effect', 'complication', 'reaction'],
            'interaction': ['drug interaction', 'combined effect'],
            
            # Oncology
            'cancer': ['malignancy', 'neoplasm', 'tumor'],
            'tumor': ['neoplasm', 'growth', 'mass'],
            'metastasis': ['spread', 'dissemination', 'secondary growth'],
            'angiogenesis': ['blood vessel formation', 'vascularization'],
            'apoptosis': ['cell death', 'programmed cell death'],
            'proliferation': ['cell growth', 'multiplication', 'expansion'],
            'invasion': ['tissue infiltration', 'spread'],
            'malignant': ['cancerous', 'aggressive', 'invasive'],
            
            # Cardiovascular
            'heart': ['cardiac', 'cardiovascular', 'myocardial'],
            'blood': ['hematological', 'circulatory', 'vascular'],
            'vessel': ['vascular structure', 'blood vessel'],
            'pressure': ['blood pressure', 'vascular pressure'],
            'circulation': ['blood flow', 'vascular flow'],
            'cardiac': ['heart-related', 'myocardial'],
            'vascular': ['blood vessel', 'circulatory'],
            'hypertension': ['high blood pressure', 'elevated pressure'],
            
            # Neurology
            'brain': ['cerebral', 'neurological', 'neural'],
            'nerve': ['neural', 'neuronal', 'nervous'],
            'neuron': ['nerve cell', 'neural cell'],
            'synapse': ['neural connection', 'nerve junction'],
            'neurotransmitter': ['neural messenger', 'chemical signal'],
            'cognitive': ['mental', 'intellectual', 'brain function'],
            'behavioral': ['psychological', 'mental', 'cognitive'],
            'sensory': ['sensation', 'perception', 'sensory system'],
            
            # Endocrinology
            'hormone': ['endocrine signal', 'chemical messenger'],
            'endocrine': ['hormonal', 'glandular'],
            'metabolism': ['metabolic process', 'biochemical process'],
            'glucose': ['blood sugar', 'carbohydrate'],
            'insulin': ['pancreatic hormone', 'glucose regulator'],
            'thyroid': ['thyroid gland', 'endocrine organ'],
            'adrenal': ['adrenal gland', 'endocrine organ'],
            'pituitary': ['pituitary gland', 'master gland'],
            
            # Respiratory
            'lung': ['pulmonary', 'respiratory', 'alveolar'],
            'airway': ['respiratory tract', 'breathing passage'],
            'respiration': ['breathing', 'ventilation'],
            'oxygen': ['O2', 'respiratory gas'],
            'ventilation': ['breathing', 'air movement'],
            'asthma': ['bronchial asthma', 'reactive airway'],
            'pneumonia': ['lung infection', 'pulmonary infection'],
            'bronchitis': ['airway inflammation', 'bronchial inflammation'],
            
            # Gastroenterology
            'stomach': ['gastric', 'digestive organ'],
            'intestine': ['bowel', 'digestive tract'],
            'liver': ['hepatic', 'hepatobiliary'],
            'pancreas': ['pancreatic', 'digestive organ'],
            'digestion': ['digestive process', 'food breakdown'],
            'absorption': ['nutrient uptake', 'molecular uptake'],
            'secretion': ['glandular output', 'fluid production'],
            'motility': ['movement', 'peristalsis'],
            
            # Musculoskeletal
            'muscle': ['muscular', 'myogenic', 'muscle tissue'],
            'bone': ['skeletal', 'osseous', 'bone tissue'],
            'joint': ['articular', 'articulation'],
            'cartilage': ['connective tissue', 'articular tissue'],
            'tendon': ['connective tissue', 'muscle attachment'],
            'ligament': ['connective tissue', 'joint stabilizer'],
            'fracture': ['bone break', 'skeletal injury'],
            'arthritis': ['joint inflammation', 'articular disease'],
            
            # Dermatology
            'skin': ['dermal', 'cutaneous', 'integumentary'],
            'epidermis': ['outer skin layer', 'skin surface'],
            'dermis': ['skin layer', 'cutaneous layer'],
            'rash': ['skin eruption', 'dermatitis'],
            'inflammation': ['inflammatory response', 'tissue reaction'],
            'wound': ['injury', 'tissue damage'],
            'healing': ['repair', 'tissue regeneration'],
            'scar': ['tissue repair', 'healed wound'],
            
            # Ophthalmology
            'eye': ['ocular', 'ophthalmic', 'visual organ'],
            'retina': ['retinal', 'visual layer'],
            'cornea': ['corneal', 'eye surface'],
            'vision': ['sight', 'visual perception'],
            'glaucoma': ['eye pressure', 'optic nerve damage'],
            'cataract': ['lens opacity', 'vision clouding'],
            'macula': ['retinal region', 'central vision'],
            'optic': ['visual', 'sight-related'],
            
            # Otolaryngology
            'ear': ['aural', 'otologic', 'hearing organ'],
            'nose': ['nasal', 'rhinologic', 'olfactory organ'],
            'throat': ['pharyngeal', 'laryngeal', 'upper airway'],
            'hearing': ['auditory', 'sound perception'],
            'balance': ['equilibrium', 'vestibular function'],
            'smell': ['olfaction', 'olfactory sense'],
            'taste': ['gustation', 'gustatory sense'],
            'voice': ['vocal', 'speech production'],
            
            # Urology
            'kidney': ['renal', 'nephric', 'urinary organ'],
            'bladder': ['urinary bladder', 'storage organ'],
            'urine': ['urinary output', 'kidney product'],
            'prostate': ['prostatic', 'male gland'],
            'dialysis': ['kidney treatment', 'renal replacement'],
            'nephritis': ['kidney inflammation', 'renal inflammation'],
            'cystitis': ['bladder inflammation', 'urinary inflammation'],
            'urolithiasis': ['kidney stones', 'urinary calculi'],
            
            # Obstetrics and Gynecology
            'pregnancy': ['gestation', 'prenatal period'],
            'fetus': ['developing baby', 'unborn child'],
            'placenta': ['placental', 'pregnancy organ'],
            'ovary': ['ovarian', 'female gonad'],
            'uterus': ['uterine', 'womb'],
            'menstruation': ['menstrual cycle', 'period'],
            'fertility': ['reproductive capacity', 'conception ability'],
            'contraception': ['birth control', 'family planning'],
            
            # Pediatrics
            'child': ['pediatric', 'young patient'],
            'infant': ['newborn', 'baby'],
            'development': ['growth', 'maturation'],
            'growth': ['development', 'maturation'],
            'vaccination': ['immunization', 'preventive care'],
            'nutrition': ['diet', 'feeding'],
            'milestone': ['developmental stage', 'growth marker'],
            'pediatric': ['child-related', 'children's health'],
            
            # Geriatrics
            'aging': ['senescence', 'elderly'],
            'elderly': ['geriatric', 'senior'],
            'degeneration': ['deterioration', 'decline'],
            'dementia': ['cognitive decline', 'memory loss'],
            'frailty': ['weakness', 'debility'],
            'longevity': ['life span', 'life expectancy'],
            'senescence': ['aging', 'biological aging'],
            'geriatric': ['elderly care', 'aging-related'],
            
            # Emergency Medicine
            'emergency': ['urgent', 'acute'],
            'trauma': ['injury', 'physical damage'],
            'shock': ['circulatory failure', 'hemodynamic collapse'],
            'resuscitation': ['revival', 'emergency care'],
            'critical': ['severe', 'life-threatening'],
            'acute': ['sudden', 'severe'],
            'stabilization': ['stabilizing', 'emergency care'],
            'triage': ['patient sorting', 'priority assessment'],
            
            # Radiology
            'imaging': ['radiological', 'diagnostic imaging'],
            'x-ray': ['radiograph', 'radiation image'],
            'mri': ['magnetic resonance', 'magnetic imaging'],
            'ct': ['computed tomography', 'tomographic imaging'],
            'ultrasound': ['sonography', 'sound imaging'],
            'radiation': ['radiological', 'ionizing radiation'],
            'contrast': ['imaging agent', 'radiological dye'],
            'scan': ['imaging study', 'diagnostic scan'],
            
            # Laboratory Medicine
            'test': ['laboratory test', 'diagnostic test'],
            'sample': ['specimen', 'biological sample'],
            'analysis': ['laboratory analysis', 'testing'],
            'result': ['test result', 'laboratory finding'],
            'reference': ['normal range', 'standard value'],
            'quality': ['test quality', 'laboratory quality'],
            'validation': ['test validation', 'method validation'],
            'calibration': ['standardization', 'measurement calibration'],
            
            # Public Health
            'epidemiology': ['disease study', 'population health'],
            'prevention': ['preventive care', 'disease prevention'],
            'screening': ['early detection', 'preventive testing'],
            'surveillance': ['monitoring', 'disease tracking'],
            'outbreak': ['disease outbreak', 'epidemic'],
            'pandemic': ['global outbreak', 'worldwide epidemic'],
            'vaccination': ['immunization', 'preventive vaccination'],
            'quarantine': ['isolation', 'containment'],
            
            # Medical Research
            'study': ['research', 'investigation'],
            'trial': ['clinical trial', 'research study'],
            'protocol': ['study protocol', 'research plan'],
            'methodology': ['research method', 'study design'],
            'analysis': ['data analysis', 'statistical analysis'],
            'result': ['finding', 'outcome'],
            'conclusion': ['finding', 'determination'],
            'publication': ['research paper', 'scientific article']
        }

        # Add original words
        expanded_words.extend(words)
        
        # Add expanded terms
        for word in words:
            if word in medical_terms:
                expanded_words.extend(medical_terms[word])
        
        # Add bigrams for better context matching
        bigrams = [' '.join(words[i:i+2]) for i in range(len(words)-1)]
        for bigram in bigrams:
            if bigram in medical_terms:
                expanded_words.extend(medical_terms[bigram])
        
        # Remove duplicates while preserving order
        seen = set()
        expanded_words = [x for x in expanded_words if not (x in seen or seen.add(x))]
        
        expanded_query = " ".join(expanded_words)
        logger.info(f"Expanded query: {expanded_query}")
        return expanded_query

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
            # Adjust weights to favor reranking more for medical queries
            candidate['final_score'] = 0.9 * candidate['rerank_score'] + 0.1 * candidate['score']
        
        # Sort by final score
        reranked = sorted(candidates, key=lambda x: x['final_score'], reverse=True)
        
        # Log top results for debugging
        logger.info("Top 3 reranked results:")
        for i, result in enumerate(reranked[:3]):
            logger.info(f"Rank {i+1} - Score: {result['final_score']:.4f}")
            logger.info(f"Text: {result['text'][:200]}...")
        
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
