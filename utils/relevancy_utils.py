import os
import asyncio
import logging

# Ragas and LangChain Imports
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import ResponseRelevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# LangChain Model Imports
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger('relevancy_utils')
response_relevancy_scorer = None # Initialize as None

# --- INITIALIZATION BLOCK ---
try:
    logger.info("Starting Ragas scorer initialization...")
    
    # 1. Check API Key
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY environment variable not set. Please set it to proceed.")

    # 2. Initialize the LLM (ChatGroq)
    llm = ChatGroq(
        model="llama-3.3-70b-versatile", 
        temperature=0.0, 
        max_retries=2,
        groq_api_key=groq_api_key,
        n=1 # Set n=1 for the base model config
    )
    logger.debug("ChatGroq LLM initialized.")

    # 3. Initialize the Embedding Model (HuggingFace)
    embedding_model = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    logger.debug("HuggingFace Embeddings initialized.")

    # 4. Wrap Models for Ragas Compatibility
    evaluator_llm = LangchainLLMWrapper(llm) 
    evaluator_embeddings = LangchainEmbeddingsWrapper(embedding_model) 

    # 5. Initialize the Ragas Scorer
    response_relevancy_scorer = ResponseRelevancy(
        llm=evaluator_llm, 
        embeddings=evaluator_embeddings,
        strictness=1 # <-- FINAL FIX: Ensures Ragas requests n=1 from Groq.
    )
    logger.info("Ragas ResponseRelevancy scorer initialized successfully.")

except Exception as e:
    logger.error(f"Ragas Initialization failed. Root cause: {type(e).__name__}: {str(e)}", exc_info=True)
    response_relevancy_scorer = None
    # Re-raise a clean, custom error for the calling synchronous function
    raise RuntimeError(f"Ragas scorer failed to initialize. Check API key and dependencies. Root Error: {str(e)}")


async def async_calculate_relevancy(question: str, answer: str) -> float:
    """
    The asynchronous core function to compute the Ragas score.
    """
    # Check if initialization was successful
    if response_relevancy_scorer is None:
        raise Exception("Ragas scorer failed to initialize. Check API key and dependencies.")
        
    sample = SingleTurnSample(
        user_input=question, 
        response=answer,
        retrieved_contexts=[]
    )
    
    # Calculate the score
    result = response_relevancy_scorer.single_turn_score(sample)
    
    # Extract the score
    if isinstance(result, dict):
        return float(result.get('response_relevancy', 0.0))
    
    try:
        return float(result)
    except (TypeError, ValueError):
        logger.error(f"Unexpected result type from Ragas: {type(result)}")
        return 0.0


def calculate_relevancy_score(question: str, answer: str) -> float:
    """
    The synchronous wrapper function for use in Flask routes.
    """
    try:
        # Execute the asynchronous function
        score = asyncio.run(async_calculate_relevancy(question, answer))
        return score
    except Exception as e:
        # Raise the error as a RuntimeError, which is caught by the routes.py
        raise RuntimeError(f"Ragas calculation failed: {str(e)}")
    
