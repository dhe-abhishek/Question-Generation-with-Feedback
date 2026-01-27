import os
import asyncio
import logging
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import AnswerCorrectness, AnswerSimilarity
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger('correctness_utils')
_correctness_scorer = None 

def get_scorer():
    global _correctness_scorer
    if _correctness_scorer is not None:
        return _correctness_scorer

    try:
        logger.info("Initializing AnswerCorrectness Scorer (Fixed)...")
        
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found.")

        # 1. Initialize models
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0, groq_api_key=api_key)
        embedding_model = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

        evaluator_llm = LangchainLLMWrapper(llm)
        evaluator_embeddings = LangchainEmbeddingsWrapper(embedding_model)

        # 2. Fix: AnswerSimilarity only needs embeddings in this version
        similarity_scorer = AnswerSimilarity(embeddings=evaluator_embeddings)
        
        # 3. AnswerCorrectness gets the LLM and the similarity scorer
        _correctness_scorer = AnswerCorrectness(
            llm=evaluator_llm,
            answer_similarity=similarity_scorer,
            weights=[0.4, 0.6] 
        )
        
        logger.info("AnswerCorrectness Scorer successfully initialized.")
        return _correctness_scorer

    except Exception as e:
        logger.error(f"CRITICAL: Scorer setup failed: {str(e)}", exc_info=True)
        raise RuntimeError(f"Initialization Error: {str(e)}")

async def async_calculate_correctness(question: str, answer: str, ground_truth: str) -> float:
    scorer = get_scorer()
    sample = SingleTurnSample(
        user_input=question, 
        response=answer,
        reference=ground_truth
    )
    # Ragas v0.2+ uses single_turn_ascore
    result = await scorer.single_turn_ascore(sample)
    return float(result)

def calculate_correctness_score(question: str, answer: str, ground_truth: str) -> float:
    try:
        # standard sync wrapper for Flask
        return asyncio.run(async_calculate_correctness(question, answer, ground_truth))
    except Exception as e:
        logger.error(f"Calculation Error: {str(e)}")
        raise RuntimeError(str(e))