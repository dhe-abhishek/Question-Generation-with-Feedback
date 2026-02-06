import logging
import json
from openai import OpenAI
from app.database import db, QuestionEvaluation
from config import Config

logger = logging.getLogger('question_evaluator')

class QuestionEvaluator:
    """
    Service responsible for evaluating generated questions using the Groq (Llama 3) model.
    Assesses questions on the 7 dimensions defined in the original evaluator.
    """

    def __init__(self):
        try:
            # Groq uses an OpenAI-compatible interface
            self.client = OpenAI(
                api_key=Config.GROQ_API_KEY,
                base_url=Config.GROQ_BASE_URL
            )
            self.model = Config.GROQ_MODEL
            logger.info(f"QuestionEvaluator initialized with Groq model: {self.model}")
        except Exception as e:
            logger.error(f"Failed to initialize Groq client: {e}")
            self.client = None

    def evaluate_and_save(self, question_id, context, question_text, answer_text):
        """
        Triggers the Groq LLM to evaluate a question based on linguistic and 
        factual consistency parameters, then saves scores to the DB.
        """
        if not self.client:
            logger.error("Groq client not initialized. Skipping evaluation.")
            return

        # Prompt reframed to include original parameters
        prompt = f"""
        You are a strict academic auditor evaluating the quality of an AI-generated question for an educational assessment. 
        Your goal is to ensure the question is scientifically accurate, linguistically perfect, and strictly derived from the provided context.

        [Context]: {context[:5000]}
        [Question]: {question_text}
        [Answer]: {answer_text}

        ### Detailed Evaluation Rubric:

        1. Fluency:
        - (5) Perfect: Professional, error-free academic language.
        - (4) Very Good: Minor punctuation or stylistic choice that doesn't impact flow.
        - (3) Average: Grammatically correct but contains awkward phrasing.
        - (2) Poor: Frequent grammatical slips or non-native phrasing.
        - (1) Unusable: Significant errors that hinder comprehension.

        2. Clarity:
        - (5) Crystal Clear: Single, unambiguous interpretation.
        - (4) Clear: Obvious meaning, though a word choice could be slightly more precise.
        - (3) Functional: Meaning is clear only after reading it twice.
        - (2) Vague: Uses vague pronouns (it, they, this) without clear referents.
        - (1) Confusing: Multiple interpretations possible; logically muddy.

        3. Conciseness:
        - (5) Optimal: Every word adds value; no "fluff."
        - (4) Good: Mostly efficient, perhaps one redundant adjective.
        - (3) Wordy: Contains 1-2 phrases that could be shortened.
        - (2) Repetitive: Uses the same words or ideas multiple times in one sentence.
        - (1) Bloated: Extremely wordy; feels like "filler" text.

        4. Relevance:
        - (5) Critical: Focuses on a core scientific/educational concept or "big idea."
        - (4) Important: Focuses on a secondary but necessary concept.
        - (3) Relevant: Focuses on a factual detail that is technically in the text.
        - (2) Trivial: Focuses on an insignificant footnote or "unimportant" date/number.
        - (1) Irrelevant: Topic is not logically connected to the main context.

        5. Consistency (Factual Alignment):
        - (5) Flawless: Facts in the question are 100% mirrored in the context.
        - (4) Strong: Conceptually correct, but uses a synonym not found in the text.
        - (3) Acceptable: No direct contradiction, but frames the fact slightly differently.
        - (2) Weak: Skews a fact or oversimplifies a complex relationship in the text.
        - (1) Contradictory: Directly goes against facts stated in the context.

        6. Answerability (Source-Based):
        - (5) Explicit: The exact answer is stated clearly in a single location in the context.
        - (4) Direct Inference: Answer requires connecting two adjacent sentences in the text.
        - (3) Multi-hop: Requires connecting information from different paragraphs in the text.
        - (2) External Hint: Partially answerable, but requires minor outside general knowledge.
        - (1) Unanswerable: The context does not contain the information needed to answer.

        7. Answer Consistency:
        - (5) Perfect Match: The provided Answer is the most accurate response to the Question.
        - (4) Strong Match: The answer is correct but could be formatted better.
        - (3) Partial: The answer is technically correct but misses a key nuance of the question.
        - (2) Mismatched: The answer addresses the topic but doesn't actually answer the specific question.
        - (1) Incorrect: The answer is wrong or logically unrelated to the question.

        ### Instructions:
        - Provide a 1-sentence 'reason' justifying why the specific score was chosen over a higher or lower one.
        - Be a strict judge. If there is any doubt, lean toward the lower score.
        - Output strictly valid JSON.

        ### Response Format:
        {{
        "evaluations": {{
            "fluency": {{"reason": "...", "score": 5}},
            "clarity": {{"reason": "...", "score": 4}},
            "conciseness": {{"reason": "...", "score": 5}},
            "relevance": {{"reason": "...", "score": 3}},
            "consistency": {{"reason": "...", "score": 5}},
            "answerability": {{"reason": "...", "score": 4}},
            "answer_consistency": {{"reason": "...", "score": 5}}
        }},
        "final_scores": {{
            "fluency": 5, "clarity": 4, "conciseness": 5, "relevance": 3, "consistency": 5, "answerability": 4, "answer_consistency": 5
        }}
        }}
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a scientific evaluator. Output strictly valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                # Groq supports JSON mode for structured output
                response_format={"type": "json_object"},
                temperature=0.1
            )

            # Extract and parse JSON content
            content = response.choices[0].message.content
            raw_data = json.loads(content)
            print("*******************************")
            print(raw_data)
            print("*******************************")
            scores = raw_data.get('final_scores', {})

            # Create Database Record using the original parameters
            evaluation = QuestionEvaluation(
                question_id=question_id,
                model_used=self.model,
                fluency=scores.get('fluency'),
                clarity=scores.get('clarity'),
                conciseness=scores.get('conciseness'),
                relevance=scores.get('relevance'),
                consistency=scores.get('consistency'),
                answerability=scores.get('answerability'),
                answer_consistency=scores.get('answer_consistency')
            )

            db.session.add(evaluation)
            db.session.commit()
            logger.info(f"Successfully saved Groq evaluation for question ID: {question_id}")
            return scores

        except Exception as e:
            db.session.rollback()
            logger.error(f"Error during Groq evaluation for question {question_id}: {str(e)}")
            return None