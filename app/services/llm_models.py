import time
import logging
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
# NOTE: In a real environment, you would need to install:
# pip install openai deepseek-client
# For this demonstration, we are mocking the API interaction.

logger = logging.getLogger('llm_models')

# --- Abstract Base Class (Interface) ---

class LLMBase(ABC):
    """Base class defining the interface for all LLM wrappers."""
    def __init__(self, model_name: str, api_key: str = ""):
        self.model_name = model_name
        self.api_key = api_key
        
    @abstractmethod
    def generate_content(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        """
        Generates content and returns raw text and metadata.
        Returns: (raw_text_output, metadata_dict)
        """
        pass

# --- Concrete Implementations (MOCKED) ---

class GeminiProModel(LLMBase):
    """Wrapper for the user's existing Gemini 2.5 Pro model."""
    def generate_content(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        start_time = time.time()
        
        # MOCK API RESPONSE for demonstration.
        # This simulates a typical successful, structured JSON response.
        mock_json = """
            {
              "questions": [
                {"question_number": 1, "question_text": "What is the primary driver of plate tectonics?", "options": {"A": "Solar winds", "B": "Mantle convection", "C": "Tidal forces"}},
                {"question_number": 2, "question_text": "Apply the process of osmosis to a plant cell placed in a hypertonic solution.", "options": {}},
                {"question_number": 3, "question_text": "Which chemical reaction describes photosynthesis?", "options": {"A": "CO2 + H2O → C6H12O6 + O2", "B": "C6H12O6 + O2 → CO2 + H2O", "C": "2H2 + O2 → 2H2O"}}
              ],
              "answer_key": ["B", "The cell would lose water and plasmolyze.", "A"],
              "bloom_levels": ["Remembering", "Applying", "Analyzing"]
            }
            """
            
        # The model wraps JSON in a markdown block
        raw_output = f"Here is the content for your {self.model_name}:\n```json\n{mock_json}\n```"
        
        latency = time.time() - start_time + 1.5 
        metadata = { "latency": latency, "tokens_used": len(raw_output.split()) * 1.8 }
        logger.info(f"Gemini Pro call successful. Latency: {latency:.2f}s")
        return raw_output, metadata

class OpenAIChatModel(LLMBase):
    """Wrapper for an OpenAI model (e.g., GPT-4o)."""
    def generate_content(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        start_time = time.time()
        time.sleep(1.0) # Simulate API latency
        
        # MOCK API RESPONSE for demonstration
        mock_json = """
            {
              "questions": [
                {"question_number": 1, "question_text": "If a user generates 3 MCQs, what is the probability they target Bloom's Level 'Applying'?", "options": {"A": "1/3", "B": "0", "C": "Cannot be determined"}},
                {"question_number": 2, "question_text": "Describe the core difference between inductive and deductive reasoning.", "options": {}},
                {"question_number": 3, "question_text": "Identify the logical flaw in the 'appeal to emotion' fallacy.", "options": {"A": "Targets feelings, not facts", "B": "Is a form of hasty generalization", "C": "Only applicable in legal arguments"}}
              ],
              "answer_key": ["C", "Inductive moves from specific to general; deductive from general to specific.", "A"],
              "bloom_levels": ["Analyzing", "Understanding", "Evaluating"]
            }
            """
        # OpenAI models often use JSON mode and may or may not wrap it, but we prepare for wrapping
        raw_output = f"```json\n{mock_json}\n```\n"

        latency = time.time() - start_time
        metadata = { "latency": latency, "tokens_used": len(raw_output.split()) * 1.6 }
        logger.info(f"OpenAI call finished. Latency: {latency:.2f}s")
        return raw_output, metadata

class DeepseekModel(LLMBase):
    """Wrapper for a Deepseek model."""
    def generate_content(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        start_time = time.time()
        time.sleep(1.8) # Simulate API latency (slightly slower)

        # MOCK API RESPONSE for demonstration
        # This is where the actual Deepseek API call would occur.
        # It receives the 'prompt' (which contains the context file's content)
        # and returns the generated question data structured as JSON.
        #
        # EXAMPLE REAL CODE STRUCTURE (conceptually):
        # from deepseek import Deepseek
        # client = Deepseek(api_key=self.api_key)
        # response = client.chat.completions.create(
        #     model=self.model_name,
        #     messages=[{"role": "user", "content": prompt}], # 'prompt' contains the context file text
        #     response_format={"type": "json_object"}
        # )
        # mock_json = response.choices[0].message.content
        #
        # For now, we use a mock placeholder to allow the rest of the framework to run:
        mock_json = """
            {
              "questions": [
                {"question_number": 1, "question_text": "How does Deepseek integrate with the LLMBase class?", "options": {"A": "Through a direct API call", "B": "By reading a local file", "C": "Via a shared database"}},
                {"question_number": 2, "question_text": "Explain why mocking is necessary for initial framework development.", "options": {}},
                {"question_number": 3, "question_text": "Identify the key difference in latency observed between the Deepseek and OpenAI mock calls.", "options": {"A": "Deepseek is slower (1.8s vs 1.0s)", "B": "OpenAI is slower", "C": "They are identical"}}
              ],
              "answer_key": ["A", "Mocking allows development of parsing and evaluation logic without spending on API credits or needing real API access.", "A"],
              "bloom_levels": ["Analyzing", "Understanding", "Remembering"]
            }
            """
        # Deepseek often includes a conversational preamble.
        raw_output = f"I have successfully generated the content based on your prompt. Here is the output block:\n```json\n{mock_json}\n```"

        latency = time.time() - start_time
        metadata = { "latency": latency, "tokens_used": len(raw_output.split()) * 1.3 }
        logger.info(f"Deepseek call finished. Latency: {latency:.2f}s")
        return raw_output, metadata


# --- Model Registry ---

MODEL_REGISTRY = {
    "Gemini_2_5_Pro": GeminiProModel(
        model_name="gemini-2.5-pro",
        api_key="AIzaSyAKyMVv9IZhcIF9vUuu1jI1oWAV-bAIKhw" # Use your production key
    ),
    "GPT_4o": OpenAIChatModel(
        model_name="gpt-4o", 
        api_key="OPENAI_API_KEY_HERE"
    ),
    "Deepseek_Coder": DeepseekModel(
        model_name="deepseek-coder", 
        api_key="DEEPSEEK_API_KEY_HERE"
    ),
    # The Mock model is now removed, but you can add it back if needed for testing:
    # "Mock_Failsafe": MockLLM(model_name="mock-v1", api_key=""),
}

def get_registered_models() -> Dict[str, LLMBase]:
    """Retrieves all models for the evaluation run."""
    return MODEL_REGISTRY
