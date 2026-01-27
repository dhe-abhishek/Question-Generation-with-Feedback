import re
import logging
import google.generativeai as genai
from utils.file_utils import extract_text_from_pdf
from config import Config

logger = logging.getLogger(__name__)

class AnswerGenerationService:
    
    @staticmethod
    def generate_answer_from_context(pdf_path, question, blooms_level):
        """
        Generate answer from PDF context based on question and Bloom's taxonomy level
        using Gemini 2.5 Pro model.
        """
        try:
            logger.info(f"Starting answer generation - PDF: {pdf_path}, Question: {question}, Bloom's: {blooms_level}")

            # Extract text from PDF
            text, error = extract_text_from_pdf(pdf_path)
            if error:
                logger.error(f"PDF extraction error: {error}")
                return None, error

            logger.info(f"PDF text extracted, length: {len(text) if text else 0}")

            if not text or len(text.strip()) < 50:
                return None, "Extracted text is too short or empty. Please check the PDF content."

            # Configure Gemini
            try:
                logger.info(f"Configuring Gemini with API key: {Config.GEMINI_API_KEY[:10]}... and model: {Config.GEMINI_MODEL}")
                genai.configure(api_key=Config.GEMINI_API_KEY)
                model = genai.GenerativeModel(Config.GEMINI_MODEL)
            except Exception as e:
                logger.error(f"Gemini configuration error: {str(e)}")
                return None, f"AI model configuration failed: {str(e)}"

            # Prepare prompt based on Bloom's taxonomy level
            bloom_instructions = {
                "remember": "Recall and remember specific facts, terms, and basic concepts from the context. Answer should directly quote or reference exact information from the text.",
                "understand": "Explain ideas and concepts from the context in your own words. Demonstrate comprehension by interpreting, summarizing, or paraphrasing the relevant information.",
                "apply": "Use information from the context in new situations. Show how concepts can be implemented or used to solve problems based on the methods described.",
                "analyze": "Draw connections among ideas in the context. Break down complex information into components, identify relationships, and organize concepts systematically.",
                "evaluate": "Make judgments and critiques based on the context. Justify positions and assess quality using evidence and criteria from the text.",
                "create": "Generate new ideas, plans, or proposals based on concepts from the context. Combine elements in innovative ways while staying grounded in the provided information."
            }

            instruction = bloom_instructions.get(blooms_level, bloom_instructions["understand"])

            prompt = f"""
            CONTEXT FROM DOCUMENT:
            {text[:6000]}  # Limit context length for API constraints

            QUESTION: {question}

            BLOOM'S TAXONOMY LEVEL: {blooms_level.upper()}
            INSTRUCTION: {instruction}

            CRITICAL REQUIREMENTS:
            1. Answer MUST be based ONLY on the provided context above
            2. Do NOT add any external knowledge, information, or personal opinions
            3. If the context doesn't contain enough information to answer, state what is available from the context
            4. Structure your answer according to the Bloom's level requirement
            5. Be precise and reference specific parts of the context where applicable

            ANSWER:
            """

            logger.info(f"Prompt prepared, sending to Gemini...")

            # Generate answer using Gemini
            try:
                response = model.generate_content(prompt)
                
                logger.info(f"Gemini response received: {response}")
                
                if not response or not response.text:
                    logger.error("Empty response from Gemini")
                    return None, "Received empty response from AI model"
                
                answer = response.text.strip()
                logger.info(f"Answer generated successfully, length: {len(answer)}")
                
                return answer, None

            except Exception as e:
                logger.error(f"Gemini API error: {str(e)}", exc_info=True)
                return None, f"AI model error: {str(e)}"

        except Exception as e:
            logger.error(f"Unexpected error in answer generation: {str(e)}", exc_info=True)
            return None, f"An unexpected error occurred: {str(e)}"