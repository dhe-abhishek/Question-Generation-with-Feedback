from pydantic import BaseModel, Field
from typing import List, Dict, Union, Any

# --- Pydantic Schema for LLM Generated Output ---

class QuestionStructure(BaseModel):
    """Defines the structure of a single question (can be MCQ or LA)."""
    question_number: int = Field(description="The sequential number of the question.")
    question_text: str = Field(description="The full text of the question.")
    options: Dict[str, str] = Field(description="A dictionary of options (e.g., {'A': 'Opt 1', 'B': 'Opt 2'}). Empty for Long Answer.")

class LLMOutputSchema(BaseModel):
    """The root schema the LLM must adhere to."""
    questions: List[QuestionStructure] = Field(description="Array of generated question objects.")
    answer_key: List[str] = Field(description="Array of correct answers (e.g., ['A', 'The core is molten'] corresponding to questions).")
    bloom_levels: List[str] = Field(description="Array of Bloom's Taxonomy levels (e.g., ['Analyzing', 'Remembering']).")

# --- Pydantic Schema for Evaluation Data (Result of Comparison) ---

class ModelMetric(BaseModel):
    """Defines the metrics collected for a single LLM model."""
    Format_Adherence: bool = Field(description="Did the model return valid JSON matching the schema?")
    Question_Count_Match: bool = Field(description="Did the model generate the exact required number of questions?")
    Latency_Seconds: float = Field(description="Time taken to get the API response.")
    Mock_Tokens_Used: int = Field(description="A mocked metric for token usage based on output size.")
    Parse_Error: Union[str, None] = Field(description="Details of any parsing or validation error.")
    Accuracy_Score: float = Field(description="Placeholder for the final human/judge accuracy score.")
    
class EvaluationData(BaseModel):
    """The root data structure for the final comparison results."""
    run_id: str
    target_questions: int
    context_file: str
    models: Dict[str, ModelMetric]