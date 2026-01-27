import os
import asyncio
import logging

# Ragas and LangChain Imports
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import Faithfulness
from ragas.llms import LangchainLLMWrapper

# LangChain Model Imports
from langchain_groq import ChatGroq

logger = logging.getLogger('faithfulness_utils')
faithfulness_scorer = None 

# --- INITIALIZATION BLOCK ---
try:
    logger.info("Initializing Faithfulness scorer...")
    
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY environment variable not set.")

    # We use the same high-quality LLM for the 'Judge' role
    llm = ChatGroq(
        model="llama-3.3-70b-versatile", 
        temperature=0.0, 
        max_retries=2,
        groq_api_key=groq_api_key
    )

    evaluator_llm = LangchainLLMWrapper(llm) 

    # Note: Faithfulness only needs an LLM (to extract and verify claims). 
    # It does NOT require an embedding model.
    faithfulness_scorer = Faithfulness(llm=evaluator_llm)
    
    logger.info("Faithfulness scorer initialized successfully.")

except Exception as e:
    logger.error(f"Faithfulness Initialization failed: {str(e)}", exc_info=True)
    faithfulness_scorer = None

async def async_calculate_faithfulness(question: str, answer: str, contexts: list) -> float:
    """
    Core async logic for Ragas faithfulness calculation.
    """
    if faithfulness_scorer is None:
        raise Exception("Faithfulness scorer is not initialized.")
        
    sample = SingleTurnSample(
        user_input=question, 
        response=answer,
        retrieved_contexts=contexts
    )
    
    # In newer Ragas versions, single_turn_ascore is the standard async method
    score = await faithfulness_scorer.single_turn_ascore(sample)
    return float(score)

def calculate_faithfulness_score(question: str, answer: str, context_text: str) -> float:
    """
    Synchronous wrapper for the Flask route.
    """
    try:
        # Convert the single context string into the list format Ragas expects
        contexts = [context_text]
        score = asyncio.run(async_calculate_faithfulness(question, answer, contexts))
        return score
    except Exception as e:
        logger.error(f"Faithfulness Calculation Error: {str(e)}")
        # Fallback to a scalar check if the version returns a result object
        raise RuntimeError(f"Faithfulness calculation failed: {str(e)}")