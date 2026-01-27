import pandas as pd
import os
from rouge import Rouge
import nltk
from config import Config
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
import logging

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

logger = logging.getLogger('file_utils')

def read_questions_from_file(file_path):
    """Read questions from a text file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            
        # Split by newlines and filter out empty lines
        questions = [q.strip() for q in content.split('\n') if q.strip()]
        return questions
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {str(e)}")
        return []

def calculate_rouge_l(candidate_questions, reference_questions):
    """Calculate ROUGE-L score between candidate and reference questions."""
    try:
        rouge = Rouge()
        
        # Convert lists to strings for ROUGE calculation
        candidate_text = " ".join(candidate_questions)
        reference_text = " ".join(reference_questions)
        
        scores = rouge.get_scores(candidate_text, reference_text)
        return scores[0]['rouge-l']['f']
    except Exception as e:
        logger.error(f"Error calculating ROUGE-L: {str(e)}")
        return 0.0

def calculate_meteor(candidate_questions, reference_questions):
    """Calculate METEOR score between candidate and reference questions."""
    try:
        total_score = 0.0
        count = 0
        
        # For each candidate question, find the best matching reference question
        for cand_q in candidate_questions:
            best_score = 0.0
            cand_tokens = word_tokenize(cand_q.lower())
            
            for ref_q in reference_questions:
                ref_tokens = word_tokenize(ref_q.lower())
                
                try:
                    score = meteor_score([ref_tokens], cand_tokens)
                    best_score = max(best_score, score)
                except Exception:
                    continue
            
            total_score += best_score
            count += 1
        
        return total_score / count if count > 0 else 0.0
    except Exception as e:
        logger.error(f"Error calculating METEOR: {str(e)}")
        return 0.0

def save_scores_to_excel(ref_filename, cand_filename, rouge_score, meteor_score, excel_path=None):
    """Save evaluation scores to Excel file, appending new results."""
    try:
        if excel_path is None:
            excel_path = os.path.join(Config.RESULTS_FOLDER, 'evaluation_scores.xlsx')
        
        # Create results directory if it doesn't exist
        os.makedirs(os.path.dirname(excel_path), exist_ok=True)
        
        # Create new data row
        new_data = {
            'Reference_File': ref_filename,
            'Candidate_File': cand_filename,
            'ROUGE_L_Score': rouge_score,
            'METEOR_Score': meteor_score,
            'Timestamp': pd.Timestamp.now()
        }
        
        # Check if file exists
        if os.path.exists(excel_path):
            # Append to existing file
            df_existing = pd.read_excel(excel_path)
            df_new = pd.DataFrame([new_data])
            df_updated = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            # Create new file
            df_updated = pd.DataFrame([new_data])
        
        # Save to Excel
        df_updated.to_excel(excel_path, index=False)
        logger.info(f"Scores saved to {excel_path}")
        
        return excel_path
        
    except Exception as e:
        logger.error(f"Error saving scores to Excel: {str(e)}")
        # Fallback: create basic Excel file
        try:
            df = pd.DataFrame([{
                'Reference_File': ref_filename,
                'Candidate_File': cand_filename,
                'ROUGE_L_Score': rouge_score,
                'METEOR_Score': meteor_score,
                'Timestamp': pd.Timestamp.now()
            }])
            df.to_excel(excel_path, index=False)
            return excel_path
        except Exception as e2:
            logger.error(f"Failed to create Excel file: {str(e2)}")
            raise e2