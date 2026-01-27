import json
import re
import google.generativeai as genai
from app.database import db, Question, McqOption, TextAnswer, QuestionType, BloomsTaxonomy, Quiz
from config import Config

class QuizService:
    @staticmethod
    def process_and_save_quiz(pdf_text, filename):
        """
        Processes text extracted from a PDF, generates questions using AI, 
        and saves a new Quiz entry with its specific mapping of questions.
        """
        genai.configure(api_key=Config.GEMINI_API_KEY)
        model = genai.GenerativeModel(Config.GEMINI_MODEL)

        try:
            # 1. Create a NEW entry in the 'quizzes' table for this generation session
            # We no longer delete old questions because we want a historical record of every quiz generated.
            new_quiz = Quiz(title=filename)
            db.session.add(new_quiz)
            db.session.flush()  # Flush assigns an ID to new_quiz so we can link questions to it

            prompt = f"""
            Extract questions from the text. 
            Return ONLY a JSON array. 
            Important: Use only the code for blooms_code (e.g., "BL-2").
            
            Format:
            [
              {{
                "text": "...",
                "type": "MCQ", 
                "blooms_code": "BL-2",
                "mcq": {{"A": "..", "B": "..", "C": "..", "D": "..", "correct": "A"}},
                "text_answer": null
              }}
            ]
            TEXT: {pdf_text}
            """

            response = model.generate_content(prompt)
            json_match = re.search(r'\[.*\]', response.text, re.DOTALL)
            
            if not json_match:
                print("Error: AI response did not contain a valid JSON array.")
                return False
                
            data = json.loads(json_match.group())

            for item in data:
                # 2. Lookup Taxonomy and Question Type
                bloom_code = item['blooms_code'].split(':')[0].strip()
                blooms = BloomsTaxonomy.query.filter_by(level_code=bloom_code).first()
                q_type = QuestionType.query.filter_by(type_name=item['type']).first()

                # 3. Create the Question entry
                new_q = Question(
                    question_text=item['text'],
                    type_id=q_type.id if q_type else None,
                    blooms_id=blooms.id if blooms else None,
                    source_document=filename
                )
                db.session.add(new_q)
                db.session.flush()  # Assigns ID to new_q

                # 4. LINK Question to this specific Quiz (Mapping)
                # This automatically populates the quiz_question_mapping association table
                new_quiz.questions.append(new_q)

                # 5. Save MCQ options or Text Answer details
                if item['type'] == 'MCQ' and item.get('mcq'):
                    opts = item['mcq']
                    db.session.add(McqOption(
                        question_id=new_q.id,
                        option_a=opts['A'], 
                        option_b=opts['B'],
                        option_c=opts['C'], 
                        option_d=opts['D'],
                        correct_option=opts['correct']
                    ))
                else:
                    db.session.add(TextAnswer(
                        question_id=new_q.id, 
                        answer_content=item.get('text_answer') or "No answer provided by AI."
                    ))

            # Commit all changes: The Quiz, all Questions, and the Mappings
            db.session.commit()
            return True
            
        except Exception as e:
            db.session.rollback()
            print(f"Database Error in QuizService: {e}")
            return False