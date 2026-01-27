import pandas as pd
import os
import logging
from config import Config
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
import nltk
import numpy as np
from collections import Counter

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

logger = logging.getLogger('question_evaluator_utils')

# Try to import rouge, with fallback
try:
    from rouge import Rouge
    ROUGE_AVAILABLE = True
except ImportError:
    try:
        from rouge_score import rouge_scorer
        ROUGE_AVAILABLE = True
        ROUGE_SCORER = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    except ImportError:
        ROUGE_AVAILABLE = False
        logger.warning("ROUGE library not available. Install with: pip install rouge-score")

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
    """Calculate ROUGE-L score between candidate and reference questions using proper alignment."""
    if not ROUGE_AVAILABLE:
        logger.error("ROUGE library not available")
        return 0.0
        
    try:
        # For question-level comparison, we need to align questions first
        # Since we have the same number of questions, we can do 1-to-1 comparison
        total_rouge_l = 0.0
        count = 0
        
        # Use the standard ROUGE implementation
        rouge = Rouge()
        
        # If we have different numbers of questions, we need to handle alignment
        if len(candidate_questions) != len(reference_questions):
            logger.warning("Different number of questions between candidate and reference. Using document-level ROUGE.")
            # Fallback to document-level ROUGE
            candidate_text = " ".join(candidate_questions)
            reference_text = " ".join(reference_questions)
            scores = rouge.get_scores(candidate_text, reference_text)
            return scores[0]['rouge-l']['f']
        
        # 1-to-1 question alignment (assuming same order)
        for i in range(min(len(candidate_questions), len(reference_questions))):
            cand_q = candidate_questions[i]
            ref_q = reference_questions[i]
            
            scores = rouge.get_scores(cand_q, ref_q)
            total_rouge_l += scores[0]['rouge-l']['f']
            count += 1
        
        return total_rouge_l / count if count > 0 else 0.0
        
    except Exception as e:
        logger.error(f"Error calculating ROUGE-L: {str(e)}")
        # Fallback to simple implementation
        return simple_rouge_l(candidate_questions, reference_questions)
    
    
def calculate_sentence_rouge_l(candidate_questions, reference_questions):
    """Calculate ROUGE-L at sentence level with proper alignment."""
    try:
        if not ROUGE_AVAILABLE:
            return simple_rouge_l(candidate_questions, reference_questions)
            
        rouge = Rouge()
        total_f1 = 0.0
        count = 0
        
        # Handle different lengths by using the minimum
        min_length = min(len(candidate_questions), len(reference_questions))
        
        for i in range(min_length):
            cand_sent = candidate_questions[i]
            ref_sent = reference_questions[i]
            
            # Calculate ROUGE for this sentence pair
            scores = rouge.get_scores(cand_sent, ref_sent)
            total_f1 += scores[0]['rouge-l']['f']
            count += 1
        
        return total_f1 / count if count > 0 else 0.0
        
    except Exception as e:
        logger.error(f"Error in sentence-level ROUGE: {str(e)}")
        return simple_rouge_l(candidate_questions, reference_questions)

def calculate_meteor(candidate_questions, reference_questions):
    """Calculate METEOR score with proper question alignment."""
    try:
        total_score = 0.0
        count = 0
        
        # If different number of questions, use best-match approach
        if len(candidate_questions) != len(reference_questions):
            logger.warning("Different number of questions. Using best-match METEOR.")
            return calculate_meteor_best_match(candidate_questions, reference_questions)
        
        # 1-to-1 question alignment (assuming same order)
        for i in range(min(len(candidate_questions), len(reference_questions))):
            cand_q = candidate_questions[i]
            ref_q = reference_questions[i]
            
            cand_tokens = word_tokenize(cand_q.lower())
            ref_tokens = word_tokenize(ref_q.lower())
            
            try:
                score = meteor_score([ref_tokens], cand_tokens)
                total_score += score
                count += 1
            except Exception as e:
                logger.debug(f"METEOR calculation warning for pair {i}: {str(e)}")
                continue
        
        return total_score / count if count > 0 else 0.0
        
    except Exception as e:
        logger.error(f"Error calculating METEOR: {str(e)}")
        return 0.0

def calculate_meteor_best_match(candidate_questions, reference_questions):
    """Calculate METEOR using best matching reference for each candidate."""
    try:
        total_score = 0.0
        count = 0
        
        for cand_q in candidate_questions:
            best_score = 0.0
            cand_tokens = word_tokenize(cand_q.lower())
            
            for ref_q in reference_questions:
                ref_tokens = word_tokenize(ref_q.lower())
                
                try:
                    score = meteor_score([ref_tokens], cand_tokens)
                    if score > best_score:
                        best_score = score
                except Exception:
                    continue
            
            total_score += best_score
            count += 1
        
        return total_score / count if count > 0 else 0.0
        
    except Exception as e:
        logger.error(f"Error in best-match METEOR: {str(e)}")
        return 0.0

def simple_rouge_l(candidate_questions, reference_questions):
    """A simple ROUGE-L implementation that does proper question alignment."""
    try:
        total_f1 = 0.0
        count = 0
        
        # 1-to-1 question alignment
        for i in range(min(len(candidate_questions), len(reference_questions))):
            cand_q = candidate_questions[i].lower()
            ref_q = reference_questions[i].lower()
            
            # Calculate LCS for this question pair
            lcs = lcs_length(cand_q, ref_q)
            
            precision = lcs / len(cand_q) if cand_q else 0
            recall = lcs / len(ref_q) if ref_q else 0
            
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
                total_f1 += f1
                count += 1
        
        return total_f1 / count if count > 0 else 0.0
        
    except Exception as e:
        logger.error(f"Error in simple ROUGE-L: {str(e)}")
        return 0.0

def lcs_length(x, y):
    """Calculate the length of the longest common subsequence between two strings."""
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]

def save_scores_to_excel(ref_filename, cand_filename, rouge_score, meteor_score, excel_path=None):
    """Save evaluation scores to Excel file, appending new results."""
    try:
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
        raise e
    
