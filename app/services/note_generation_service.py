import logging
from config import Config
import google.generativeai as genai

logger = logging.getLogger('note_service')

class NoteGenerationService:
    def __init__(self):
        # Configure the Gemini API using the model from your Config
        genai.configure(api_key=Config.GEMINI_API_KEY)
        self.model = genai.GenerativeModel(Config.GEMINI_MODEL)

    def generate_notes(self, text_content, learner_level, additional_links=None):
        """
        Generates detailed notes based on the learner level.
        """
        try:
            # Define level-specific instructions
            prompts = {
                "average": (
                    "Create notes for an AVERAGE learner. Use the simplest language possible. "
                    "Include: 1) A clear text-based Mind Map. 2) Simple diagrams represented in text/ASCII. "
                    "3) Basic numerical examples if applicable. 4) A real-world practical example. "
                    "5) Easy memory tricks (mnemonics) to remember the core concepts."
                ),
                "intermediate": (
                    "Create notes for an INTERMEDIATE learner. Use standard academic terminology. "
                    "Include: 1) A detailed Mind Map of concept relationships. 2) Logic flowcharts. "
                    "3) Moderate numerical examples with step-by-step solutions. 4) Professional practical examples. "
                    "5) A deeper explanation of 'how' and 'why' the concepts work together."
                ),
                "advanced": (
                    "Create notes for an ADVANCED learner. Use technical and precise language. "
                    "Include: 1) A complex systems Mind Map. 2) Advanced numerical proofs or high-level calculations. "
                    "3) Edge-case practical examples or industry-standard applications. 4) Theoretical analysis. "
                    "5) At the end, provide a 'Deep Dive' section with questions or topics to motivate further reading."
                )
            }

            instruction = prompts.get(learner_level, prompts["average"])
            
            # Construct the final prompt
            full_prompt = f"""
            Task: {instruction}
            
            Content Source:
            {text_content}
            
            Additional Reference Context:
            {additional_links if additional_links else "None provided"}
            
            Format the output with clear Markdown headers for readability.
            """

            response = self.model.generate_content(full_prompt)
            
            if response and response.text:
                return response.text, None
            return None, "Model failed to generate a response."

        except Exception as e:
            logger.error(f"Error in NoteGenerationService: {str(e)}")
            return None, str(e)