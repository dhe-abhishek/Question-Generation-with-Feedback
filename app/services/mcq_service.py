import google.generativeai as genai
import os
import json
import logging
from config import Config # Assuming Config is accessible for API key

logger = logging.getLogger('mcq_service')

# Use a specific format for the model to output a JSON object,
# which is much more reliable for programmatic parsing than plain text.
GENERATION_PROMPT = """
You are an expert educator tasked with generating multiple-choice questions (MCQs) from a provided text.
Generate {num_questions} unique MCQs, each with 4 options (A, B, C, D) and an identified correct answer.
Crucially, you must map each question to a specific level of Bloom's Taxonomy (Level 1-6) and output a JSON object.

Bloom's Taxonomy Levels:
1: Remembering (Recalling facts/basic concepts)
2: Understanding (Explaining ideas/concepts)
3: Applying (Using information in new situations)
4: Analyzing (Drawing connections among ideas)
5: Evaluating (Justifying a stand or decision)
6: Creating (Producing new or original work)

The output MUST be a single, valid JSON array of objects. DO NOT include any preamble, explanation, or text outside of the JSON block.

Text to generate questions from:
---
{text_content}
---

JSON Output Format:
[
    {{
        "question": "Question text goes here?",
        "options": ["Option A", "Option B", "Option C", "Option D"],
        "correct_answer": "Option Letter (A, B, C, or D)",
        "blooms_level": "Level Number (1 to 6)"
    }},
    ...
]
"""

def generate_mcqs_from_text(text_content: str, num_questions: int) -> list:
    """
    Generates MCQs using the Gemini model based on the provided text content.
    Returns a list of MCQ dictionaries or an empty list on failure.
    """
    if "ERROR:" in text_content:
        logger.error("Cannot generate MCQs: Text content contains error from extraction step.")
        return []

    try:
        # Configuration
        google_api_key = os.getenv('GOOGLE_API_KEY')
        if not google_api_key:
            logger.error("GOOGLE_API_KEY is not set.")
            return []
            
        genai.configure(api_key=google_api_key)
        
        # Use a model that supports JSON mode and better reasoning
        model_name = 'gemini-2.5-pro' # Use a strong model for complex tasks like Bloom's Taxonomy mapping
        model = genai.GenerativeModel(model_name)

        prompt = GENERATION_PROMPT.format(num_questions=num_questions, text_content=text_content)
        
        logger.info(f"Generating {num_questions} MCQs using {model_name}...")
        
        # Generate with JSON response format
        generation_config = genai.types.GenerationConfig(
            response_mime_type="application/json",
        )
        
        
        # Generate the content with JSON output
        response = model.generate_content(prompt, generation_config=generation_config)
        
        # Parse the JSON response
        logger.info("Printing response text...")
        print(response.text)
        mcqs = json.loads(response.text)
        logger.info(f"Successfully generated and parsed {len(mcqs)} MCQs.")
        return mcqs

    except Exception as e:
        logger.error(f"Error during question generation: {e}", exc_info=True)
        return []

