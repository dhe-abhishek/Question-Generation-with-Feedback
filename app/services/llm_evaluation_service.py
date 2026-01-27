import logging
import json
import re
from typing import Dict, Any, Tuple
from pydantic import ValidationError

from app.services.llm_models import get_registered_models, LLMBase
from utils.pydantic_schema import LLMOutputSchema, ModelMetric
from utils.json_utils import extract_json_block, robust_json_fix # Assuming these exist

logger = logging.getLogger('llm_evaluation_service')

class LLMEvaluationService:
    """
    Manages the execution of prompts across multiple LLMs and validates their output.
    """

    def __init__(self, target_questions: int):
        self.target_questions = target_questions
        self.models = get_registered_models()
        logger.info(f"Initialized LLMEvaluationService with {len(self.models)} models. Target questions: {target_questions}")

    def _craft_prompt(self, text_content: str, question_type: str = '1', blooms_level_choice: str = '2') -> str:
        """
        Generates the detailed, structured prompt for the LLMs.
        """
        q_type_desc = {'1': 'Multiple-Choice Questions (MCQs)', '4': 'Long Answer Questions'}.get(question_type, 'Questions')
        blooms_desc = {'1': 'Remembering', '2': 'Understanding', '3': 'Applying', '4': 'Analyzing', '5': 'Evaluating', '6': 'Creating'}.get(blooms_level_choice, 'Understanding')
        
        prompt = (
            f"Generate exactly **{self.target_questions} {q_type_desc}** from the provided text. "
            f"Each question must target the Bloom's Taxonomy level of **'{blooms_desc}'**. "
            f"For MCQs, ensure 4 options (A, B, C, D). For Long Answer, use an empty options dictionary: {{}}.\n\n"
            f"**STRICT OUTPUT REQUIREMENT:** "
            f"Return the output as a **single, valid JSON object** enclosed in a markdown block (```json...```). "
            f"The object must strictly follow the schema: `LLMOutputSchema` (containing 'questions', 'answer_key', and 'bloom_levels' arrays).\n\n"
            f"--- CONTEXT DOCUMENT TEXT ---\n{text_content}\n--- END CONTEXT ---\n"
        )
        logger.debug(f"Prompt crafted for {self.target_questions} questions at BL-{blooms_level_choice}.")
        return prompt

    def run_evaluation(self, text_content: str, run_parameters: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, ModelMetric]]:
        """
        Executes the comparison run and validates outputs.
        
        Returns: 
            raw_outputs: Dict of {model_alias: raw_text}
            validated_metrics: Dict of {model_alias: ModelMetric}
        """
        all_raw_outputs: Dict[str, str] = {}
        all_validated_metrics: Dict[str, ModelMetric] = {}
        
        try:
            prompt = self._craft_prompt(
                text_content=text_content,
                question_type=run_parameters.get('question_type', '1'),
                blooms_level_choice=run_parameters.get('blooms_level', '2')
            )
            
            for alias, llm_instance in self.models.items():
                logger.info(f"Starting API call for model: {alias} ({llm_instance.model_name})")
                raw_output, metadata = llm_instance.generate_content(prompt)
                all_raw_outputs[alias] = raw_output
                
                # --- Validation and Metric Collection ---
                metrics = ModelMetric(
                    Format_Adherence=False,
                    Question_Count_Match=False,
                    Latency_Seconds=metadata.get('latency', 0.0),
                    Mock_Tokens_Used=metadata.get('tokens_used', 0),
                    Parse_Error=None,
                    Accuracy_Score=0.0 # Will be populated manually/via judge
                )
                
                try:
                    # 1. Extract JSON block (using utility from existing project structure)
                    json_str = extract_json_block(raw_output)
                    if not json_str:
                        raise ValueError("Could not extract JSON block from raw output.")
                    
                    # 2. Attempt robust parsing and loading
                    data = robust_json_fix(json_str)
                    if not data:
                        raise ValueError("Failed to robustly parse JSON string.")
                        
                    # 3. Pydantic validation
                    validated_data = LLMOutputSchema.model_validate(data)
                    metrics.Format_Adherence = True
                    
                    # 4. Question Count Check
                    num_generated_q = len(validated_data.questions)
                    if num_generated_q == self.target_questions:
                        metrics.Question_Count_Match = True
                    else:
                        metrics.Parse_Error = f"Q-Count Mismatch: Expected {self.target_questions}, got {num_generated_q}"
                        logger.warning(f"{alias} Q-Count Mismatch: {metrics.Parse_Error}")

                    logger.info(f"Validation successful for {alias}. Count match: {metrics.Question_Count_Match}")
                        
                except (json.JSONDecodeError, ValidationError, ValueError) as e:
                    metrics.Parse_Error = str(e)[:150] # Truncate error for display
                    logger.error(f"Validation failed for {alias}: {metrics.Parse_Error}")
                
                all_validated_metrics[alias] = metrics
                
            return all_raw_outputs, all_validated_metrics
            
        except Exception as e:
            logger.critical(f"Critical error during evaluation run: {str(e)}", exc_info=True)
            return all_raw_outputs, all_validated_metrics
            
