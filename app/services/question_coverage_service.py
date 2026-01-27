import logging
from typing import List, Dict, Optional, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re # Ensure re is imported

logger = logging.getLogger('question_coverage_service')

class QuestionCoverageService:
    """
    Service class for calculating the relevance (coverage) of generated questions 
    against the source context using TF-IDF and cosine similarity.
    """

    @staticmethod
    def extract_questions_from_text(questions_text: str) -> List[str]:
        """
        Extracts individual question blocks from a raw block of text (typically from a PDF).
        
        This method is designed to handle the multi-line, structured format 
        (e.g., 'Question X: ... Answer: ... Correct Answer: ...').
        """
        
        # New robust regex pattern: 
        # 1. Start capturing at the beginning of a question: (Question\s*\d+:\s*)
        # 2. Use non-greedy match (.*?) to capture everything after the question number.
        # 3. Stop capturing right before the next question number or the end of the string ($).
        # re.DOTALL is critical to make '.' match newline characters.
        pattern = re.compile(r'(Question\s*\d+:\s*.*?)(?=\nQuestion\s*\d+:|$)', re.DOTALL | re.IGNORECASE)
        
        question_blocks = pattern.findall(questions_text)
        
        if not question_blocks:
            logger.warning("No questions found using the structured regex pattern.")
            return []

        cleaned_questions = []
        for block in question_blocks:
            # 1. Strip overall whitespace
            block = block.strip()
            
            # 2. Remove the 'Correct Answer' line and everything after it 
            # (as this is noise for coverage analysis).
            block = re.sub(r'Correct Answer:.*', '', block, flags=re.DOTALL | re.IGNORECASE)
            
            # 3. Remove the Bloom's Level tag if present (e.g., [BL-1: Remembering])
            block = re.sub(r'\[BL-\d:.*?\]', '', block, flags=re.IGNORECASE)

            # 4. Remove generic formatting like 'Short Answer (SA) from Source Document'
            block = re.sub(r'(?:Short Answer \(SA\)|Multiple Choice Question \(MCQ\)) from Source Document', '', block, flags=re.IGNORECASE)
            
            # 5. Remove page breaks if present (e.g., --- PAGE X ---)
            block = re.sub(r'---\s*PAGE\s*\d+\s*---', '', block, flags=re.IGNORECASE)

            # 6. Clean up excessive newlines and whitespace after cleaning
            block = re.sub(r'\n\s*\n', '\n', block).strip()
            
            if block:
                cleaned_questions.append(block)

        logger.info(f"Extracted and cleaned {len(cleaned_questions)} question blocks from the PDF text.")
        return cleaned_questions


    @staticmethod
    def calculate_relevance_scores(context_text: str, questions_list: List[str]) -> Optional[List[Dict[str, Any]]]:
        """
        Calculates the relevance of each question to the context using TF-IDF and 
        Cosine Similarity.
        """
        if not context_text or not questions_list:
            logger.error("Context text or questions list cannot be empty.")
            return None

        # Combine the context and all questions into a single corpus
        corpus = [context_text] + questions_list
        
        try:
            # Initialize TF-IDF Vectorizer
            # Use 'english' stop words and tune parameters for better performance
            vectorizer = TfidfVectorizer(stop_words='english', max_df=0.85, min_df=2)
            tfidf_matrix = vectorizer.fit_transform(corpus)
        except ValueError as e:
            # This often happens if all documents are too short or empty after cleaning.
            logger.error(f"Error fitting TF-IDF vectorizer (likely documents too sparse): {e}")
            return None

        # The first vector is the context (Document 0)
        context_vector = tfidf_matrix[0:1]
        
        # The subsequent vectors are the questions (Document 1 onwards)
        question_vectors = tfidf_matrix[1:]

        # Calculate Cosine Similarity between the context and each question
        similarity_scores = cosine_similarity(context_vector, question_vectors).flatten()

        # Compile results
        results = []
        for i, question_text in enumerate(questions_list):
            results.append({
                'question': question_text,
                # FIX: Removed * 100 to ensure the score is between 0 and 1
                'relevance_score': round(float(similarity_scores[i]), 4)
            })
            
        # Sort results by relevance score (highest first)
        results.sort(key=lambda x: x['relevance_score'], reverse=True)

        return results