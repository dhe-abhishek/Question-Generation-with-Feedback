import json
import re
import google.generativeai as genai
from app.database import db, Question, McqOption, TextAnswer, QuestionType, BloomsTaxonomy
from config import Config

class QuizService:
    @staticmethod
    def process_and_save_quiz(pdf_text, filename):
        genai.configure(api_key=Config.GEMINI_API_KEY)
        model = genai.GenerativeModel(Config.GEMINI_MODEL)

        # 1. PREVENT REPETITION: Delete old questions for this specific file
        Question.query.filter_by(source_document=filename).delete()
        db.session.commit()

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

        try:
            response = model.generate_content(prompt)
            json_match = re.search(r'\[.*\]', response.text, re.DOTALL)
            data = json.loads(json_match.group())

            for item in data:
                # 2. SMART BLOOM LOOKUP
                # This ensures "BL-2" matches even if the AI sends "BL-2: Understanding"
                bloom_code = item['blooms_code'].split(':')[0].strip()
                blooms = BloomsTaxonomy.query.filter_by(level_code=bloom_code).first()
                
                q_type = QuestionType.query.filter_by(type_name=item['type']).first()

                new_q = Question(
                    question_text=item['text'],
                    type_id=q_type.id if q_type else None,
                    blooms_id=blooms.id if blooms else None,
                    source_document=filename
                )
                db.session.add(new_q)
                db.session.flush()

                if item['type'] == 'MCQ' and item['mcq']:
                    opts = item['mcq']
                    db.session.add(McqOption(
                        question_id=new_q.id,
                        option_a=opts['A'], option_b=opts['B'],
                        option_c=opts['C'], option_d=opts['D'],
                        correct_option=opts['correct']
                    ))
                else:
                    db.session.add(TextAnswer(
                        question_id=new_q.id, 
                        answer_content=item['text_answer']
                    ))

            db.session.commit()
            return True
        except Exception as e:
            db.session.rollback()
            print(f"Error: {e}")
            return False