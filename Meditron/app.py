import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import torch
import sys
import os
import logging
import warnings
import config

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Suppress warnings
warnings.filterwarnings('ignore')
logging.getLogger('streamlit').setLevel(logging.ERROR)

# Try importing the required modules
try:
    from retriever import PubMedRetriever
    from Generator import MedicalGenerator as Generator
except ImportError as e:
    st.error(f"Error importing required modules: {str(e)}")
    st.error("Please make sure all required files are in the correct location.")
    st.error(f"Current directory: {current_dir}")
    st.error(f"Python path: {sys.path}")
    st.stop()

import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json

# Set page config
st.set_page_config(
    page_title="Meditron Medical QA System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to control text size
st.markdown("""
    <style>
    .response-text {
        font-size: 16px !important;
        line-height: 1.5;
    }
    .context-text {
        font-size: 14px !important;
        line-height: 1.4;
        color: #666;
    }
    .stMarkdown {
        font-size: 16px !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'Home'
if 'query_history' not in st.session_state:
    st.session_state.query_history = []
if 'evaluation_metrics' not in st.session_state:
    st.session_state.evaluation_metrics = {
        'retrieval_scores': [],
        'response_times': [],
        'query_lengths': []
    }

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", 
    ['Home', 'Project Overview', 'Dataset', 'Technical Details', 'Model Architecture', 'Demo', 'Performance Metrics'])
st.session_state.page = page

# Initialize models
@st.cache_resource
def load_models():
    with st.spinner("Loading models... This may take a few minutes."):
        try:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0" if config.DEVICE == "cuda" else ""
            retriever = PubMedRetriever()
            generator = Generator()
            return retriever, generator
        except Exception as e:
            st.error(f"Error initializing models: {str(e)}")
            st.stop()

# Load models
try:
    retriever, generator = load_models()
except Exception as e:
    st.error(f"Error loading models: {str(e)}")
    st.stop()

# Home page
if st.session_state.page == 'Home':
    st.title("🏥 Meditron Medical QA System")
    st.markdown("""
    ## Welcome to Our Medical Question-Answering System
    
    This application demonstrates a state-of-the-art medical question-answering system powered by:
    - **Meditron-7B**: A specialized medical language model
    - **PubMedBERT**: For biomedical text understanding
    - **Sentence Transformers**: For semantic search capabilities
    
    ### Key Features
    - Real-time medical question answering
    - Context-aware responses based on medical literature
    - High-accuracy information retrieval
    - Performance monitoring and visualization
    
    Navigate through the sidebar to explore different aspects of the system.
    """)

# Project Overview page
elif st.session_state.page == 'Project Overview':
    st.title("Project Overview")
    st.markdown("""
    ## Objective
    Our goal was to create an advanced medical question-answering system that leverages:
    1. Large Language Models (LLMs) specialized in medical knowledge
    2. Efficient information retrieval from medical literature
    3. Semantic search capabilities for accurate context matching
    
    ## System Architecture
    The system consists of three main components:
    
    ### 1. Retriever Component
    - Uses **PubMedBERT** for understanding medical queries
    - Implements semantic search using **Sentence Transformers**
    - Utilizes FAISS for efficient similarity search
    
    ### 2. Generator Component
    - Powered by **Meditron-7B**, a medical domain LLM
    - Fine-tuned on medical literature and conversations
    - Optimized for medical question answering
    
    ### 3. Integration Layer
    - Streamlit-based user interface
    - Real-time performance monitoring
    - Query history tracking
    """)

# Dataset page
elif st.session_state.page == 'Dataset':
    st.title("Dataset Description")
    st.markdown("""
    ## PubMedQA Dataset
    
    Our system is trained and evaluated on the PubMedQA dataset, which consists of:
    
    ### Dataset Statistics
    - **Total Examples**: 1,000,000+ medical questions and answers
    - **Source**: PubMed abstracts and medical literature
    - **Types**: Research questions, clinical queries, and medical concepts
    
    ### Data Processing Pipeline
    1. **Text Extraction**
       - Parsing PubMed abstracts
       - Cleaning and normalizing text
       - Extracting relevant contexts
    
    2. **Embedding Generation**
       - Using Sentence Transformers for text embedding
       - Dimension: 768 (base model)
       - Stored in FAISS index for efficient retrieval
    
    3. **Quality Control**
       - Removing duplicates and near-duplicates
       - Ensuring medical relevance
       - Validating context quality
    """)

# Add new Technical Details page
elif st.session_state.page == 'Technical Details':
    st.title("Technical Implementation Details")
    
    st.markdown("""
    ## 1. Document Processing Pipeline
    
    ### Text Extraction and Preprocessing
    - **Source**: PubMed abstracts in XML format
    - **Processing Steps**:
        1. XML parsing using `xml.etree.ElementTree`
        2. Text cleaning and normalization
        3. Sentence segmentation using `nltk.sent_tokenize`
        4. Special character handling and whitespace normalization
    
    ### Document Chunking
    - **Chunk Size**: 512 tokens
    - **Overlap**: 128 tokens
    - **Rationale**: Balance between context preservation and retrieval granularity
    
    ## 2. Embedding Generation
    
    ### Sentence Transformers Implementation
    ```python
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-mpnet-base-v2')
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)
    ```
    
    ### Embedding Details
    - **Model**: all-mpnet-base-v2
    - **Embedding Size**: 768 dimensions
    - **Batch Processing**: 32 documents per batch
    - **Normalization**: L2 normalization applied
    - **GPU Acceleration**: Using CUDA for faster processing
    
    ## 3. FAISS Index Construction
    
    ### Index Architecture
    ```python
    dimension = 768  # Embedding dimension
    index = faiss.IndexFlatL2(dimension)  # Base index
    index = faiss.IndexIDMap2(index)  # Add ID mapping
    ```
    
    ### Index Properties
    - **Type**: L2 distance-based similarity
    - **Storage**: Memory-mapped for efficient access
    - **ID Mapping**: Preserves document IDs for retrieval
    - **Size**: {len(documents)} documents indexed
    
    ### Performance Optimization
    - **GPU Acceleration**: FAISS GPU support enabled
    - **Batch Processing**: 1000 vectors per batch
    - **Memory Management**: Streaming index building
    
    ## 4. Retrieval System
    
    ### Query Processing
    ```python
    # Query embedding
    query_vector = model.encode([query])
    
    # FAISS search
    D, I = index.search(query_vector, k=top_k)
    ```
    
    ### Similarity Search
    - **Method**: L2 distance computation
    - **Top-K**: Retrieving top 5 most similar documents
    - **Distance Threshold**: 0.7 (configurable)
    
    ### Context Reranking
    - **Model**: cross-encoder/ms-marco-MiniLM-L-6-v2
    - **Batch Size**: 32 pairs per batch
    - **Score Normalization**: Min-max scaling applied
    
    ## 5. Response Generation
    
    ### Context Processing
    - **Selection**: Top 2 contexts after reranking
    - **Truncation**: Smart sentence boundary detection
    - **Maximum Length**: 300 characters per context
    
    ### Prompt Engineering
    ```python
    prompt = f'''Based on the following medical contexts, please provide a concise answer.
    
    Context 1: {context1}
    Context 2: {context2}
    
    Question: {query}
    Answer:'''
    ```
    
    ### Generation Parameters
    ```python
    generation_config = {
        'max_new_tokens': 100,
        'temperature': 0.7,
        'top_p': 0.9,
        'num_beams': 1,
        'do_sample': True,
        'early_stopping': True
    }
    ```
    
    ## 6. Performance Optimization
    
    ### Memory Management
    - CUDA cache clearing
    - Garbage collection
    - Efficient tensor handling
    
    ### Speed Optimization
    - Batch processing
    - GPU acceleration
    - Inference mode for generation
    
    ### Resource Usage
    - GPU Memory: ~15GB peak
    - CPU Usage: 4-8 cores
    - Disk I/O: Memory-mapped files
    """)

# Model Architecture page
elif st.session_state.page == 'Model Architecture':
    st.title("Model Architecture")
    st.markdown("""
    ## Core Components
    
    ### 1. Sentence Transformers
    - **Model**: all-mpnet-base-v2
    - **Purpose**: Generate semantic embeddings
    - **Architecture**: Transformer-based encoder
    - **Output**: 768-dimensional vectors
    
    ### 2. PubMedBERT
    - **Base Model**: BERT architecture
    - **Training**: Specialized on biomedical text
    - **Features**: 
        - Domain-specific vocabulary
        - Medical entity recognition
        - Contextual understanding
    
    ### 3. Meditron-7B
    - **Architecture**: 7 billion parameter model
    - **Training**: Medical domain specialization
    - **Features**:
        - Medical knowledge integration
        - Context-aware generation
        - High accuracy in medical responses
    
    ## Information Flow
    1. Query → Sentence Transformer → Embedding
    2. Embedding → FAISS Index → Relevant Contexts
    3. Contexts + Query → Meditron-7B → Generated Answer
    """)

# Demo page
elif st.session_state.page == 'Demo':
    st.title("Live Demo")
    
    # Query input
    query = st.text_area("Enter your medical question:", height=100)
    
    if st.button("Submit Query"):
        if query:
            with st.spinner("Processing your query..."):
                start_time = datetime.now()
                
                # Get retrieval results
                retrieval_results = retriever.retrieve(query)
                
                # Generate response
                response = generator.generate(query, retrieval_results)
                
                # Calculate metrics
                end_time = datetime.now()
                response_time = (end_time - start_time).total_seconds()
                
                # Update metrics
                st.session_state.evaluation_metrics['retrieval_scores'].append(
                    np.mean([r['score'] for r in retrieval_results])
                )
                st.session_state.evaluation_metrics['response_times'].append(response_time)
                st.session_state.evaluation_metrics['query_lengths'].append(len(query.split()))
                
                # Add to query history
                st.session_state.query_history.append({
                    'query': query,
                    'response': response,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                
                # Display response with controlled formatting
                st.subheader("Response:")
                st.markdown(f'<div class="response-text">{response}</div>', unsafe_allow_html=True)
                
                # Display retrieval results
                st.subheader("Retrieved Contexts:")
                for i, result in enumerate(retrieval_results[:2]):
                    with st.expander(f"Context {i + 1} (Score: {result['score']:.2f})"):
                        st.markdown(f'<div class="context-text">{result["text"][:300]}</div>', unsafe_allow_html=True)

# Performance Metrics page
elif st.session_state.page == 'Performance Metrics':
    st.title("Performance Metrics")
    
    if st.session_state.evaluation_metrics['retrieval_scores']:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_retrieval = np.mean(st.session_state.evaluation_metrics['retrieval_scores'])
            st.metric("Average Retrieval Score", f"{avg_retrieval:.2f}")
            
        with col2:
            avg_response_time = np.mean(st.session_state.evaluation_metrics['response_times'])
            st.metric("Average Response Time", f"{avg_response_time:.2f}s")
            
        with col3:
            avg_query_length = np.mean(st.session_state.evaluation_metrics['query_lengths'])
            st.metric("Average Query Length", f"{avg_query_length:.1f} words")
        
        # Performance over time
        st.subheader("Performance Trends")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=st.session_state.evaluation_metrics['response_times'],
            name="Response Time (s)",
            mode='lines+markers'
        ))
        fig.update_layout(
            title="Response Time Trend",
            xaxis_title="Query Number",
            yaxis_title="Time (seconds)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Query History
        st.subheader("Recent Queries")
        for i, history_item in enumerate(reversed(st.session_state.query_history[-5:])):
            with st.expander(f"Query {len(st.session_state.query_history) - i}"):
                st.write(f"**Query:** {history_item['query']}")
                st.write(f"**Response:** {history_item['response']}")
                st.write(f"**Time:** {history_item['timestamp']}")
    else:
        st.info("No queries processed yet. Try the Demo page to generate some metrics!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with ❤️ using Streamlit, Meditron-7B, and PubMedBERT</p>
</div>
""", unsafe_allow_html=True)