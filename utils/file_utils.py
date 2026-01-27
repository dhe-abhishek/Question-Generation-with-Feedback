import os
import logging
from werkzeug.utils import secure_filename
from config import Config
import json
import pdfplumber
import docx
import re
import fitz
import matplotlib.pyplot as plt
#from app.services.mcq_generation_service import generate_mcqs_from_text
from app.services.pdf_generation import save_questions_to_text_file, create_pdf
from typing import List, Dict, Optional

logger = logging.getLogger('file_utils')

def allowed_file(filename):
    """Checks if a file's extension is allowed."""
    allowed = '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS
    logger.debug(f"File {filename} allowed: {allowed}")
    return allowed

def save_uploaded_file(file):
    """Saves an uploaded file to the uploads directory"""
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(Config.UPLOAD_FOLDER, filename)
        
        # Ensure upload directory exists
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        file.save(file_path)
        
        logger.info(f"File saved to: {file_path}")
        return file_path, filename, None
        
    return None, None, "Invalid file format"

def get_file_extension(filename):
    """Extracts the file extension from a filename."""
    # Split the filename into a base and extension
    return os.path.splitext(filename)[1].lstrip('.').lower()

def generate_mcqs_from_file(file_path, num_questions, model_id, model_dir,blooms_level_choice):
    """Main function to generate MCQs from a file"""
    logger.info(f"Generating MCQs from file: {file_path}")
    
    # Extract text from file
    text = extract_text_from_file(file_path)
    if not text:
        logger.error("Could not extract text from file")
        return None, "Could not extract text from the uploaded file"
    
    # Load the model
    model, error = load_local_model(model_id, model_dir)
    if error:
        logger.error(f"Model loading error: {error}")
        return None, f'Error loading model: {error}'

    # Generate MCQs
    logger.info("Generating MCQs...")
    json_mcqs = generate_mcqs_from_text(text, num_questions, model, blooms_level_choice)
    
    if not json_mcqs:
        logger.error("Failed to generate MCQs")
        return None, 'Failed to generate MCQs from the provided text'
        
    mcqs = json.loads(json_mcqs)
    logger.info(f"Successfully generated {len(mcqs)} MCQs")
    
    return mcqs, None

""" 
def save_mcq_results(mcqs, base_filename):
    #Save MCQ results to text and PDF files
    logger.info(f"Saving MCQ results for: {base_filename}")
    
    # Ensure results directory exists
    from config import Config
    os.makedirs(Config.RESULTS_FOLDER, exist_ok=True)
    
    # Save results to files
    txt_filename = f"generated_mcqs_{base_filename}.txt"
    pdf_filename = f"generated_mcqs_{base_filename}.pdf"
    
    if not save_mcqs_to_file(mcqs, txt_filename):
        logger.error("Failed to save text file")
        return None, None, 'Error saving text file'
        
    if not create_pdf(mcqs, pdf_filename):
        logger.error("Failed to create PDF")
        return None, None, 'Error creating PDF file'
    
    return txt_filename, pdf_filename, None
 """

def save_mcq_results(questions: List[Dict], answer_key: List[Dict], base_filename: str) -> tuple[str | None, str | None]:
    """
    Saves MCQ results to a text file by calling the dedicated service.
    
    Returns:
        tuple: (txt_filename, error)
    """
    logger.info(f"Saving TXT results for: {base_filename}")
    
    os.makedirs(Config.RESULTS_FOLDER, exist_ok=True)
    
    # Create a unique name for the text file
    txt_filename = f"generated_mcqs_{os.path.splitext(base_filename)[0]}_key.txt"
    
    # FIX: Pass both questions and answer_key to the text file saver
    txt_filepath, error = save_questions_to_text_file(
        questions=questions, 
        answer_key=answer_key, 
        filename=txt_filename
    )
    
    if error:
        logger.error(f"Failed to save text file: {error}")
        return None, error
        
    logger.info(f"TXT file saved: {txt_filename}")
    # Return only the text filename and error status
    return txt_filename, None


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts text content from a PDF file using pymupdf (fitz).
    """
    text_content = ''
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text_content += page.get_text()
        doc.close()
        
        # Preprocess the extracted text (from your notebook code)
        import re
        # Remove extra whitespace and newline characters
        cleaned_text = re.sub(r'\s+', ' ', text_content).strip()

        # Simple header/footer removal based on line length heuristic (optional but good for cleaning)
        # This is the same logic from your notebook:
        lines = cleaned_text.split('\n')
        cleaned_lines = []
        for line in lines:
            if len(line) > 50: # Adjust the length threshold as needed
                cleaned_lines.append(line)

        cleaned_text = ' '.join(cleaned_lines)
        return cleaned_text, None
    except FileNotFoundError:
        error_msg = f"PDF file not found at: {pdf_path}"
        logger.error(error_msg)
        return None, error_msg
    except Exception as e:
        error_msg = f"Error during PDF text extraction: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return None, error_msg


    except FileNotFoundError:
        logger.error(f"PDF file not found at: {pdf_path}")
        return "ERROR: File not found"
    except Exception as e:
        logger.error(f"Error during PDF text extraction: {e}", exc_info=True)
        return f"ERROR: An error occurred during text extraction: {str(e)}"

def extract_mcqs_from_generated_pdf(pdf):
    """Extracts MCQs from PDFs generated by this system"""
    mcqs = []
    current_mcq = {}
    
    for page in pdf.pages:
        text = page.extract_text()
        if not text:
            continue
            
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            
            # Detect question pattern (e.g., "Question 1:")
            question_match = re.match(r'Question\s+(\d+):\s*(.*?)\s*(?:\[BL-(\d+)\])?', line, re.IGNORECASE)
            if question_match:
                # Save previous question if exists
                if current_mcq:
                    mcqs.append(current_mcq)
                
                # Start new question
                current_mcq = {
                    'question': question_match.group(2),
                    'options': [],
                    'correct_answer': None,
                    'blooms_level': question_match.group(3) if question_match.group(3) else None
                }
                continue
                
            # Detect options (A), B), etc.)
            option_match = re.match(r'([A-D])\)\s*(.*)', line)
            if option_match and current_mcq:
                current_mcq['options'].append(option_match.group(2))
                continue
                
            # Detect correct answer
            correct_match = re.match(r'Correct Answer:\s*([A-D])', line, re.IGNORECASE)
            if correct_match and current_mcq:
                current_mcq['correct_answer'] = correct_match.group(1).upper()
                continue
                
            # Detect Bloom's level in the question text if not already captured
            if current_mcq and not current_mcq.get('blooms_level'):
                blooms_match = re.search(r'\[BL-(\d+)\]', line)
                if blooms_match:
                    current_mcq['blooms_level'] = blooms_match.group(1)
    
    # Add the last question
    if current_mcq:
        mcqs.append(current_mcq)
    
    return mcqs

def save_data_cloud_results(cloud_data, original_filename):
    """
    Saves the generated WordCloud image to the results folder.

    Args:
        cloud_data (dict): Contains 'wordcloud_object'.
        original_filename (str): Base name for the result file.

    Returns:
        tuple: (cloud_filename, error)
    """
    wordcloud_object = cloud_data.get('wordcloud_object')
    if not wordcloud_object:
        return None, "WordCloud object not found in data."

    try:
        # Create a unique filename for the image
        base_name = os.path.splitext(original_filename)[0]
        image_filename = f"{base_name}_wordcloud.png"
        save_path = os.path.join(Config.RESULTS_FOLDER, image_filename)

        # 1. Create a figure (similar to your original code)
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111) # Add a single subplot
        
        # 2. Display the WordCloud object
        ax.imshow(wordcloud_object, interpolation='bilinear')
        
        # 3. Turn off axes
        ax.axis('off')

        # 4. Save the figure to the specified path (Crucial Step!)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        
        # Close the figure to free up memory (important in a server environment)
        plt.close(fig) 
        
        return image_filename, None

    except Exception as e:
        return None, f"Error saving WordCloud image: {str(e)}"
    
    
def cleanup_file(file_path: str):
    """
    Removes a file from the filesystem.
    This is used to clean up temporary uploads after processing.
    """
    if file_path and os.path.exists(file_path):
        try:
            os.remove(file_path)
            logger.info(f"Cleaned up file: {file_path}")
        except OSError as e:
            # Handle cases where the file might be locked or permissions are an issue
            logger.error(f"Error cleaning up file {file_path}: {e}")
