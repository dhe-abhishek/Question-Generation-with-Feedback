import logging
import json
from typing import List, Dict, Optional, Any
from config import Config
from google import genai
from google.genai.errors import APIError 

logger = logging.getLogger('question_generation')

class QuestionGenerator:
    """
    Service class responsible for generating questions using the Google Gemini API.
    """
    
    # Define system instruction for the LLM
    SYSTEM_INSTRUCTION = (\
        "You are an expert educational content creator. Your task is to generate questions and "
        "a corresponding answer key based on the provided text and user specifications. "
        "The output MUST be a JSON object matching the provided schema. "
        "Do not include any text, headers, or explanations outside of the JSON object. "
        "Ensure the JSON is perfectly formatted and ready to be parsed by Python's json.loads()."\
    )

    def __init__(self, model_name: str = Config.GEMINI_MODEL):
        """Initializes the Gemini client."""
        try:
            # The API key should be available via an environment variable or set in Config
            self.client = genai.Client()
            self.model_name = model_name
            logger.info(f"QuestionGenerator initialized with model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini Client: {e}", exc_info=True)
            self.client = None
            
    def _get_question_schema(self, question_type: str) -> Dict[str, Any]:
        """
        Returns the appropriate JSON schema for the requested question type.
        """

        # -----------------------------------------------
        # Common schema for MCQ (default)
        # -----------------------------------------------
        mcq_question_item = {
            "type": "object",
            "properties": {
                "question_number": {"type": "integer", "description": "The sequential number of the question."},
                "blooms_level": {"type": "integer", "description": "The Bloom's Taxonomy level (1-6)."},
                "question": {"type": "string", "description": "The full text of the question."},
                "options": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "An array of 4 possible answers."
                },
                "correct_option_letter": {"type": "string", "description": "The correct option letter (A, B, C, or D)."}
            },
            "required": ["question_number", "blooms_level", "question", "options", "correct_option_letter"]
        }

        # Common answer key structure
        answer_key_item = {
            "type": "object",
            "properties": {
                "question_number": {"type": "integer", "description": "The sequential number of the question."},
                "correct_answer": {"type": "string", "description": "The correct answer value, matching the correct_option_letter or the FIB answer."}
            },
            "required": ["question_number", "correct_answer"]
        }

        # Main wrapper schema
        schema = {
            "type": "object",
            "properties": {
                "questions": {
                    "type": "array",
                    "items": mcq_question_item,  # Will be replaced dynamically per question type
                    "description": "A list of generated questions."
                },
                "answer_key": {
                    "type": "array",
                    "items": answer_key_item,
                    "description": "A list of correct answers corresponding to the questions."
                }
            },
            "required": ["questions", "answer_key"]
        }

        # -----------------------------------------------
        # TYPE 1 → Multiple Choice Questions (already default)
        # -----------------------------------------------
        if question_type == '1':
            schema['properties']['questions']['items'] = mcq_question_item

        # -----------------------------------------------
        # TYPE 2 → Fill-in-the-Blank Questions (FIB)
        # -----------------------------------------------
        elif question_type == '2':
            fib_question_item = {
                "type": "object",
                "properties": {
                    "question_number": {"type": "integer", "description": "The sequential number of the question."},
                    "blooms_level": {"type": "integer", "description": "The Bloom's Taxonomy level (1-6)."},
                    "question": {"type": "string", "description": "The sentence with one word or phrase replaced by [BLANK]."},
                },
                "required": ["question_number", "blooms_level", "question"]
            }
            schema['properties']['questions']['items'] = fib_question_item

            # Update answer key description for FIBs
            schema['properties']['answer_key']['items']['properties']['correct_answer']['description'] = (
                "The text that correctly fills the [BLANK]."
            )

        # -----------------------------------------------
        # TYPE 3 → Short Answer Questions (SAQ)
        # -----------------------------------------------
        elif question_type == '3':
            # Short answers are concise, 1–3 sentences
            saq_question_item = {
                "type": "object",
                "properties": {
                    "question_number": {"type": "integer", "description": "The sequential number of the question."},
                    "blooms_level": {"type": "integer", "description": "The Bloom's Taxonomy level (1-6)."},
                    "question": {"type": "string", "description": "A brief conceptual or factual question answerable in 1–3 sentences."},
                },
                "required": ["question_number", "blooms_level", "question"]
            }

            schema['properties']['questions']['items'] = saq_question_item

            # Update answer key description for SAQ
            schema['properties']['answer_key']['items']['properties']['correct_answer']['description'] = (
                "A short, precise, and accurate model answer (1–3 sentences)."
            )

        # -----------------------------------------------
        # TYPE 4 → Long Answer Questions (LAQ)
        # -----------------------------------------------
        elif question_type == '4':
            # Long answers are detailed, multi-paragraph, analytical
            laq_question_item = {
                "type": "object",
                "properties": {
                    "question_number": {"type": "integer", "description": "The sequential number of the question."},
                    "blooms_level": {"type": "integer", "description": "The Bloom's Taxonomy level (1-6)."},
                    "question": {"type": "string", "description": "An open-ended, analytical question requiring a detailed explanation."},
                },
                "required": ["question_number", "blooms_level", "question"]
            }

            schema['properties']['questions']['items'] = laq_question_item

            # Update answer key description for LAQ
            schema['properties']['answer_key']['items']['properties']['correct_answer']['description'] = (
                "A comprehensive, well-structured model answer covering key points and reasoning."
            )

        # -----------------------------------------------
        # Unsupported Question Type
        # -----------------------------------------------
        else:
            logger.warning(f"Unsupported question type: {question_type}. Defaulting to MCQ schema.")
            schema['properties']['questions']['items'] = mcq_question_item

        return schema


    def _craft_prompt(self, text_content: str, num_questions: int, question_type: str, blooms_level_choice: str) -> str:
        """
        Creates the detailed prompt for the Gemini model.
        """
        
        type_map = {'1': 'multiple-choice questions (MCQs)',
                    '2': 'fill-in-the-blank questions (FIBs)',
                    '3': 'Short Answer (SA)',
                    '4': 'Long Answer (LA)'}
        q_type_desc = type_map.get(question_type, 'multiple-choice questions (MCQs)')
        
        blooms_map = {
            '1': 'Remembering', '2': 'Understanding', '3': 'Applying',
            '4': 'Analyzing', '5': 'Evaluating', '6': 'Creating', 'all': 'any appropriate level'
        }
        blooms_desc = blooms_map.get(blooms_level_choice, 'any appropriate level')

        
        prompt = (
            f"Generate {num_questions} {q_type_desc} from the text provided below. "
            f"Each question must be mapped to a Bloom's Taxonomy level, specifically targeting "
            f"**{blooms_desc}** (Level {blooms_level_choice}).\n\n"
        )
        
        # --- MCQ ---
        if question_type == '1':
            prompt += (
                "For MCQs, ensure each question has exactly 4 options (A, B, C, D) and specify "
                "the correct option letter. The questions array should contain all questions "
                "and their options, and the answer_key array should contain the correct answers.\n\n"
            )
            
        # --- Fill-in-the-Blank ---
        elif question_type == '2':
            prompt += (
                "For FIBs, write the question as a sentence with one word or short phrase replaced by '[BLANK]'. "
                "The questions array should contain the FIB sentences. The answer_key array must "
                "contain the exact word or phrase that fills the blank for each question.\n\n"
            )
        
        # --- Short Answer ---
        elif question_type == '3':
            prompt += (
                "For Short Answer (SA), generate concise, fact-based or concept-based questions "
                "that can be answered in one to three sentences. Each question should assess understanding or "
                "application of a specific idea from the text. The 'questions' array should contain these questions, "
                "and the 'answer_key' array must include a short, precise, and accurate model answer for each.\n\n"
            )

        # --- Long Answer ---
        elif question_type == '4':
            prompt += (
                "For Long Answer (LA), generate open-ended, analytical, or explanatory questions "
                "that require detailed responses of one or more paragraphs. These should test higher-order "
                "thinking skills like analysis, evaluation, or creation. The 'questions' array should contain "
                "these detailed questions, and the 'answer_key' array must include a comprehensive, well-structured "
                "model answer or explanation for each question.\n\n"
            )
            
        # --- Self-Correction and JSON Mandate ---
        prompt += (
            "**--- VERIFICATION (Internal Step) ---**\n"
            "**Before generating the final output**, internally verify that:\n"
            "1. Every question is fully answerable using ONLY the provided text (No Hallucination).\n"
            "2. The answer key is 100% factually accurate for the corresponding question.\n"
            "3. The required Bloom's level is accurately assessed by each question.\n\n"
            "**--- FINAL OUTPUT MANDATE ---**\n"
            "Provide the output as a **single, valid JSON object**. Do not include any text, notes, or explanations outside of the JSON.\n"
            "The JSON structure must be:\n"
            "```json\n"
            f'{{\n  "questions": [/* Array of {q_type_desc} objects or strings */],\n'
            f'  "answer_key": [/* Array of correct answers */],\n'
            f'  "bloom_levels": [/* Array of Bloom\'s levels (e.g., "Applying") for each question */]\n'
            f'}}\n'
            "```\n\n"
        )
        
        # --- Append the source text ---
        prompt += f"--- TEXT CONTENT ---\n{text_content}\n--- END TEXT ---\n"
        
        return prompt

    def _generate_structured_content(self, prompt: str, schema: Dict[str, Any]) -> tuple[Optional[Dict], Optional[str]]:
        """
        Calls the Gemini API to generate structured JSON content.
        
        Returns:
            tuple: (structured_results, error_message)
        """
        if not self.client:
            return None, "Gemini client not initialized. Check API Key configuration."

        try:
            # The structure for generation config with JSON schema
            generation_config = genai.types.GenerateContentConfig(
                system_instruction=self.SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
                response_schema=schema,
            )
            
            logger.info(f"Generating structured content using model {self.model_name}...")
            
            # Call the API
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[prompt],
                config=generation_config
            )
            
            # The response text should be a valid JSON string matching the schema
            json_text = response.text
            
            # Robustly parse the JSON text
            try:
                structured_results = json.loads(json_text)
                return structured_results, None
            except json.JSONDecodeError as jde:
                logger.error(f"JSON Decode Error: {jde} - Raw text: {json_text[:200]}...")
                return None, f"Could not decode JSON response from model. {jde}"

        except APIError as e:
            error_msg = f"Gemini API Error: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return None, error_msg
        except Exception as e:
            error_msg = f"An unexpected error occurred during API call: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return None, error_msg


    def generate_questions(self, text_content: str, num_questions: int, question_type: str, blooms_level_choice: str) -> tuple[Optional[List], Optional[List], Optional[str]]:
        """
        Generates a list of questions and their corresponding answer key using the Gemini API.

        This method is the public interface used by app.routes to fix the AttributeError.

        Args:
            text_content (str): The source text to generate questions from.
            num_questions (int): The number of questions to generate.
            question_type (str): The type of question ('1' for MCQ, '2' for FIB, etc.).
            blooms_level_choice (str): The target Bloom's level ('all' or '1' through '6').

        Returns:
            tuple: (questions_list, answer_key_list, error_message)
        """
        
        # 1. Craft Prompt
        prompt = self._craft_prompt(text_content, num_questions, question_type, blooms_level_choice)
        
        # 2. Get Structured Schema
        schema = self._get_question_schema(question_type)
        
        # 3. Generate Content
        structured_results, error = self._generate_structured_content(prompt, schema)
        
        if error:
            return None, None, error
            
        # 4. Extract and Return Questions and Key
        questions = structured_results.get('questions', [])
        answer_key = structured_results.get('answer_key', [])
        
        if not questions or not answer_key:
            return None, None, "Generated content was missing questions or answer key."
            
        logger.info(f"Successfully separated {len(questions)} questions and {len(answer_key)} answer key items.")
        
        # FIX: Ensure a three-element tuple is returned for the successful case
        return questions, answer_key, None
    
    
    def reframe_question(self, text_content, original_question, original_answer, feedback, question_type='1'):
        """
        Reframes ONLY ONE specific question using source text and user feedback.
        Ensures the original format (MCQ, FIB, etc.) is maintained.
        """
        # Dynamically select schema based on the actual question type
        schema = self._get_question_schema(question_type)
        
        prompt = (
            f"You are an expert educational editor. A user has requested to REGENERATE ONLY ONE specific question.\n"
            f"STRICT CONTEXT: {text_content}\n\n"
            f"ORIGINAL QUESTION TO FIX: {original_question}\n"
            f"CURRENT ANSWER: {original_answer}\n"
            f"USER FEEDBACK/REASON: {feedback}\n\n"
            "INSTRUCTIONS:\n"
            "1. Rewrite ONLY the specific question provided above. Do NOT generate a new list.\n"
            "2. Maintain the exact same Bloom's Taxonomy level and format.\n"
            "3. If the feedback says the answer is wrong, use the context to provide the correct one.\n"
            "4. Return the result as a JSON object with 'questions' and 'answer_key' arrays containing EXACTLY ONE item."
        )
        
        # Generate content using the targeted schema
        structured_results, error = self._generate_structured_content(prompt, schema)
        
        if error or not structured_results or not structured_results.get('questions'):
            return None, None, error or "AI failed to reframe the content."
        
        raw_q = structured_results['questions'][0]
        raw_a = structured_results['answer_key'][0]
            
        # Return only the first element of the arrays
        # 1. Normalize Question Dictionary
        # We use the specific order: question_number, blooms_level, question, options, correct_option_letter
        new_q = {
            'question_number': raw_q.get('question_number'),
            'blooms_level': raw_q.get('blooms_level'),
            'question': raw_q.get('question'),
            'options': raw_q.get('options')
        }
        # Handle MCQ specific key name
        if question_type == '1':
            # Ensure we use 'correct_option_letter' even if LLM returns 'correct_answer'
            new_q['correct_option_letter'] = raw_q.get('correct_option_letter') or raw_q.get('correct_answer') or raw_q.get('correct_option')

        # 2. Normalize Answer Dictionary
        # Order: question_number, correct_answer
        new_a = {
            'question_number': raw_a.get('question_number'),
            'correct_answer': raw_a.get('correct_answer') or raw_a.get('correct_option')
        }
        
        return new_q, new_a, None