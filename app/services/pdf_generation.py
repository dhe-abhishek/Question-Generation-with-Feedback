import os
import re
import logging
from fpdf import FPDF
from config import Config
from typing import List, Dict, Optional

logger = logging.getLogger('pdf_generation')

def safe_text(text: Optional[str]) -> str:
    """Ensure text is properly encoded for PDF generation (latin-1)"""
    if text is None:
        return ""
    # Convert to string, strip whitespace, and handle encoding
    text = str(text).strip()
    # fpdf uses 'latin-1' encoding. Replace unsupported chars.
    return text.encode('latin-1', 'replace').decode('latin-1')

def _merge_data(questions: List[Dict], answer_key: List[Dict]) -> List[Dict]:
    """
    Merges the separate questions and answer_key lists into a single,
    unified list of question objects based on their 'id'.
    """
    merged_data = {}
    
    # Create a lookup map from the questions list
    for q in questions:
        q_id = q.get('id') or q.get('question_number')
        if q_id is not None:
            merged_data[q_id] = q.copy()

    for ans in answer_key:
        ans_id = ans.get('id') or ans.get('question_number')
        if ans_id in merged_data:
            # Match the key name based on your schema
            merged_data[ans_id]['answer'] = ans.get('correct_answer')
        else:
            logger.warning(f"Found answer for non-existent question ID: {ans_id}")
            
    # Return a list of the merged objects, sorted by ID
    return sorted(merged_data.values(), key=lambda x: x.get('id', 0))

def save_questions_to_text_file(questions: List[Dict], answer_key: List[Dict], filename: str) -> tuple[str | None, str | None]:
    """
    Saves the generated questions and answers to a text file.
    (Renamed from save_mcqs_to_file for clarity)
    """
    
    # FIX: Merge the questions and answers first
    mcqs = _merge_data(questions, answer_key)
    if not mcqs:
        logger.warning("No merged data to save to text file.")
        return None, "No questions were generated to save."
        
    logger.info(f"Saving merged questions to text file: {filename}")
    
    blooms_mapping = {
        '1': 'Remembering', '2': 'Understanding', '3': 'Applying', 
        '4': 'Analyzing', '5': 'Evaluating', '6': 'Creating'
    }
    
    file_path = os.path.join(Config.RESULTS_FOLDER, filename)
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            for mcq in mcqs:
                # FIX: Support both 'id' and 'question_number'
                q_num = mcq.get('id') or mcq.get('question_number', 'N/A')
                f.write(f"--- Question {q_num} ---\n")
                
                blooms_level = str(mcq.get('blooms_level', ''))
                blooms_text = blooms_mapping.get(blooms_level, 'N/A')
                f.write(f"Bloom's Level: {blooms_level} ({blooms_text})\n")
                
                question_text = mcq.get('question', 'Error: Missing Question Text')
                f.write(f"Question: {question_text}\n")
                
                # Check if it's an MCQ and has options
                if 'options' in mcq and isinstance(mcq['options'], list):
                    f.write("Options:\n")
                    options_list = mcq['options']
                    
                    # ✅ Automatically add A), B), C), D) labels
                    option_labels = ['A', 'B', 'C', 'D']
                    for idx, option_text in enumerate(options_list):
                        label = option_labels[idx] if idx < len(option_labels) else chr(65 + idx)
                        f.write(f"  {label}) {option_text}\n")
                    
                # FIX: Get the answer from the merged 'answer' key
                answer_value = mcq.get('answer', 'N/A')
                f.write(f"Answer: {answer_value}\n\n")

        logger.info(f"Successfully saved {len(mcqs)} questions to text file.")
        return file_path, None

    except Exception as e:
        logger.error(f"Error saving MCQs to text file: {str(e)}", exc_info=True)
        return None, f"Failed to save questions to text file: {str(e)}"


def create_pdf(
    questions: List[Dict],
    answer_key: List[Dict],
    base_filename: str,
    question_type_code: str = '1'
    ) -> tuple[str | None, str | None]:
    """
    Generates a PDF from a list of structured questions and a separate answer key.

    Args:
        questions (List[Dict]): List of question dictionaries from generator.
        answer_key (List[Dict]): List of answer dictionaries from generator.
        base_filename (str): The original uploaded filename (e.g., "my_doc.pdf").
        question_type_code (str): The code for the question type ('1' = MCQ, '2' = Short Answer/FIB).

    Returns:
        tuple: (generated_pdf_filename, error_message)
    """

    # Merge questions + answers
    mcqs = _merge_data(questions, answer_key)
    if not mcqs:
        logger.warning("No merged data to create PDF.")
        return None, "No questions were generated to create a PDF."

    # Prepare output path
    pdf_filename = f"generated_mcqs_{os.path.splitext(base_filename)[0]}_questions.pdf"
    file_path = os.path.join(Config.RESULTS_FOLDER, pdf_filename)
    logger.info(f"Creating PDF file: {file_path}")

    try:
        pdf = FPDF()
        pdf.add_page()

        # Header
        question_type_name = Config.QUESTION_TYPES.get(question_type_code, 'Assessment')
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, txt=safe_text(f"{question_type_name} from Source Document"), ln=1, align="C")
        pdf.ln(5)

        blooms_mapping = {
            '1': 'Remembering', '2': 'Understanding', '3': 'Applying',
            '4': 'Analyzing', '5': 'Evaluating', '6': 'Creating'
        }

        for mcq in mcqs:
            q_num = mcq.get('id') or mcq.get('question_number', 'N/A')

            # Bloom's Level Header
            blooms_level = str(mcq.get('blooms_level', ''))
            blooms_text = blooms_mapping.get(blooms_level, f'Level {blooms_level}')
            pdf.set_font("Arial", size=10)
            pdf.set_fill_color(240, 240, 240)
            pdf.cell(0, 6, txt=safe_text(f"[BL-{blooms_level}: {blooms_text}]"), ln=1, align="L", fill=True)

            # Question Text
            question_text = mcq.get('question', 'Question Missing')
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 6, txt=safe_text(f"Question {q_num}:"), ln=1, align="L")
            pdf.set_font("Arial", size=11)
            max_width = pdf.w - pdf.l_margin - pdf.r_margin
            pdf.multi_cell(max_width, 6, txt=safe_text(question_text))
            pdf.ln(2)

            # 3. Options (for MCQs)
            if question_type_code == '1' and 'options' in mcq and isinstance(mcq['options'], list):
                pdf.set_font("Arial", 'I', 11)
                pdf.cell(0, 6, txt=safe_text("Options:"), ln=1, align="L")
                pdf.set_font("Arial", size=11)

                options_list = mcq.get('options', [])
                option_labels = ['A', 'B', 'C', 'D']

                for idx, option_text in enumerate(options_list):
                    label = option_labels[idx] if idx < len(option_labels) else chr(65 + idx)
                    try:
                        # Print label + wrapped text
                        pdf.set_font("Arial", 'B', 11)
                        pdf.cell(10, 6, txt=f"{label})", ln=0)
                        pdf.set_font("Arial", size=11)
                        pdf.multi_cell(max_width - 10, 6, txt=safe_text(option_text))
                        pdf.ln(1)
                    except Exception as e:
                        logger.warning(f"Skipping option rendering due to layout issue: {e}")
                        pdf.multi_cell(max_width, 6, txt=safe_text(f"{label}) [Rendering issue skipped]"))

                pdf.ln(2)

            # 4. Correct Answer or Short Answer
            pdf.set_font("Arial", 'B', 12)
            answer_label = "Correct Answer:" if question_type_code == '1' else "Answer:"
            answer_value = mcq.get('answer', 'N/A')

            pdf.cell(0, 8, txt=safe_text(answer_label), ln=1, align="L")
            pdf.set_font("Arial", size=11)
            pdf.multi_cell(max_width, 6, txt=safe_text(answer_value))
            pdf.ln(6)

        # Save PDF
        pdf.output(file_path, 'F')
        logger.info(f"PDF successfully generated at {file_path}")
        return pdf_filename, None

    except Exception as e:
        logger.error(f"Error generating PDF: {str(e)}", exc_info=True)
        if os.path.exists(file_path):
            os.remove(file_path)
        return None, str(e)


# This class seems to be a duplicate/older version of create_pdf.
# It's safer to leave it but ensure it's not being called.
# The main 'create_pdf' function above is what routes.py calls.
class PDFGenerator:
    """
    DEPRECATED? Handles the generation of PDF documents from structured question data.
    """
    def __init__(self):
        pass 

    def generate_pdf(self, mcqs: List[Dict], filename: str) -> str:
        logger.warning("Called deprecated PDFGenerator.generate_pdf")
        # Reroute to the functional method, but it's missing the answer key
        new_filename, error = create_pdf(mcqs, [], filename, '1') 
        if error:
            return None
        return new_filename

