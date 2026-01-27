import os
import logging
import re
from flask import render_template, request, send_file, Blueprint, flash, redirect, url_for
from config import Config
from utils.file_utils import allowed_file, save_uploaded_file, save_mcq_results, extract_text_from_pdf, save_data_cloud_results, cleanup_file
#from app.services.mcq_service import generate_mcqs_from_text
from app.services.data_cloud_service import generate_data_cloud_from_text
from utils.validation import validate_num_questions, validate_file_and_params
from app.services.pdf_generation import create_pdf # Import the PDF creation utility
from app.services.question_generation import QuestionGenerator # Import the service class
from werkzeug.utils import secure_filename
import pdfplumber


# Create a Blueprint for the main routes
main_blueprint = Blueprint('main', __name__)
logger = logging.getLogger('routes')

@main_blueprint.route('/')
def home():
    """Renders the home page with all system features."""
    logger.info("Rendering home page")
    return render_template('home.html')

@main_blueprint.route('/mcq-generator')
def question_generator():
    """Renders the question generator page."""
    logger.info("Rendering Question generator page")
    return render_template('index.html')


 
@main_blueprint.route('/generate', methods=['POST'])
def generate_questions():
    """Handles the file upload and question generation process."""
    logger.info("Received Question generation request")
    
    file_path = None  # Initialize to handle cleanup on errors
    
    try:
        # --- 1. Parameter Retrieval and Validation ---
        file = request.files.get('file')
        num_questions_str = request.form.get('num_questions', '5')
        question_type = request.form.get('question_type', '1')  # Default to MCQ
        blooms_level_choice = request.form.get('blooms_level_choice', 'all')
        
        # Use the utility function to validate file and number of questions
        error = validate_file_and_params(file, num_questions_str, Config.MAX_QUESTIONS, Config.ALLOWED_EXTENSIONS)
        if error:
            flash(error)
            return redirect(url_for('main.question_generator'))

        # Parse valid parameters
        num_questions, _ = validate_num_questions(num_questions_str, max_val=Config.MAX_QUESTIONS)
        
        # --- 2. Save File and Extract Text ---
        file_path, filename, error = save_uploaded_file(file)
        if error:
            flash(error)
            return redirect(url_for('main.question_generator'))
            
        # extract_text_from_pdf is assumed to handle all supported types here
        # FIX: Unpack 2 values (text_content, error)
        text_content, error = extract_text_from_pdf(file_path)
        if error:
            flash(f'Error extracting text: {error}')
            cleanup_file(file_path) # Use cleanup utility
            return redirect(url_for('main.question_generator'))

        # --- 3. Generate Questions ---
        generator = QuestionGenerator(Config.GEMINI_MODEL)
        
        # Unpack 3 values (questions, answer_key, error)
        questions_list, answer_key_list, error = generator.generate_questions(
            text_content, num_questions, question_type, blooms_level_choice
        )
        
        # Clean up uploaded file immediately after use
        cleanup_file(file_path)
        
        if error:
            logger.error(f"Question generation failed: {error}")
            flash(f'Error generating questions: {error}')
            return redirect(url_for('main.question_generator'))

        # --- 4. Save and Render Results ---
        
        # FIX: Pass all required arguments to file generators
        
        # Generate the PDF and get its filename
        pdf_filename, pdf_error = create_pdf(questions_list, answer_key_list, filename, question_type)
        if pdf_error:
            flash(f"Error creating PDF file: {pdf_error}")
            # Continue to generate TXT file if possible
            
        # Generate the TXT file and get its filename
        txt_filename, txt_error = save_mcq_results(questions_list, answer_key_list, filename)
        if txt_error:
            flash(f"Error creating TXT file: {txt_error}")
            # If PDF also failed, redirect
            if pdf_error:
                return redirect(url_for('main.question_generator'))

        
        logger.info(f"Successfully generated {len(questions_list)} questions.")
        
        # Render the results page with download links
        return render_template(
            'results.html', 
            questions=questions_list, # This might be used to display on page, keep it
            pdf_filename=pdf_filename, 
            txt_filename=txt_filename # Use txt_filename
        )

    except Exception as e:
        logger.error(f"Error during question generation process: {str(e)}", exc_info=True)
        cleanup_file(file_path) # Ensure cleanup on unexpected crash
        flash(f'An unexpected server error occurred: {str(e)}')
        return redirect(url_for('main.question_generator'))
 
 
 
@main_blueprint.route('/download/<filename>')
def download_file(filename):
    """Handles file downloads for generated MCQs."""
    logger.info(f"Download request for: {filename}")
    
    # Construct full absolute path and normalize slashes
    file_path = os.path.normpath(os.path.join(Config.RESULTS_FOLDER, filename))
    
    # Security check
    if not os.path.exists(file_path) or '..' in filename or filename.startswith(('/', '\\')):
        logger.error(f"File not found or invalid path: {file_path}")
        flash('File not found')
        return redirect(url_for('main.question_generator'))
        
    logger.info(f"Sending file: {file_path}")
    return send_file(file_path, as_attachment=True)


@main_blueprint.route('/data-cloud')
def data_cloud():
    """Renders the data visualization and analytics page"""
    logger.info("Rendering Data Cloud visualization page")
    return render_template('data_cloud.html')


@main_blueprint.route('/generate_data_cloud', methods=['POST'])
def generate_data_cloud():
    """Handles the file upload and Data Cloud generation process."""
    logger.info("Received Data Cloud generation request")
    
    file_path = None  # Initialize to handle cleanup on errors
    
    # 1. Basic File Check
    if 'file' not in request.files:
        logger.error("No file part in request")
        flash('No file part in the request.')
        return redirect(url_for('main.data_cloud'))
    
    file = request.files['file']

    # If the user submits an empty part without a filename.
    if file.filename == '':
        flash('No selected file.')
        return redirect(url_for('main.data_cloud'))

    if file and allowed_file(file.filename):
        try:
            # 2. Save Uploaded File
            file_path, filename, error = save_uploaded_file(file)
            if error:
                flash(error)
                return redirect(url_for('main.data_cloud'))

            # 3. Extract Text from File
            text_content = extract_text_from_pdf(file_path)
            
            # Remove uploaded file immediately after extraction to keep 'uploads' clean
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
                file_path = None  # Mark as cleaned up
                 
            if "ERROR:" in text_content:
                flash(f"Error during file processing: {text_content.replace('ERROR: ', '')}")
                return redirect(url_for('main.data_cloud'))
            
            # 4. Generate Data Cloud using the service
            cloud_data = generate_data_cloud_from_text(text_content)

            if not cloud_data:
                flash('Failed to generate Data Cloud. Please check your configuration and the server logs for details.')
                return redirect(url_for('main.data_cloud'))
                
            # 5. Save results (image file)
            cloud_filename, error = save_data_cloud_results(cloud_data, filename)
            
            # Check if there was an error saving files
            if error:
                flash(f'Error saving results: {error}')
                return redirect(url_for('main.data_cloud'))

            logger.info(f"Generated Data Cloud for file {filename}.")
            
            # 6. Render Results Page
            return render_template('data_cloud_results.html', 
                                 cloud_data=cloud_data,
                                 cloud_filename=cloud_filename)

        except Exception as e:
            logger.error(f"Error during Data Cloud generation process: {str(e)}", exc_info=True)
            # Ensure the uploaded file is cleaned up in case of a crash
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
            flash(f'An unexpected server error occurred: {str(e)}')
            return redirect(url_for('main.data_cloud'))
    else:
        flash('Invalid file type. Please upload a supported file.')
        return redirect(url_for('main.data_cloud'))


# Additional routes for other features
@main_blueprint.route('/blooms-checker')
def blooms_checker():
    """Renders the Bloom's Level Checker page."""
    logger.info("Rendering Bloom's Level Checker page")
    return render_template('blooms_checker.html', models=Config.AVAILABLE_MODELS)


@main_blueprint.route('/question-generation-analytics')
def question_generation_analytics():
    """Renders the question generation analytics page."""
    logger.info("Rendering Question generation analytics page")
    return render_template('aqg_analytics.html')





#vialli
logger = logging.getLogger(__name__)

def extract_text_from_file(file_path):
    """Extracts text content from PDF, DOCX, or TXT files."""
    logger.info(f"Extracting text from: {file_path}")
    
    ext = file_path.rsplit('.', 1)[1].lower()
    text = ""
    
    try:
        if ext == 'pdf':
            logger.debug("Processing PDF file")
            with pdfplumber.open(file_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                    if i >= 10:  # Limit pages for performance
                        break
            logger.info(f"Extracted {len(text)} chars from PDF")
            
        elif ext == 'docx':
            logger.debug("Processing DOCX file")
            doc = docx.Document(file_path)
            text = ' '.join([para.text for para in doc.paragraphs])
            logger.info(f"Extracted {len(text)} chars from DOCX")
            
        elif ext == 'txt':
            logger.debug("Processing TXT file")
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            logger.info(f"Extracted {len(text)} chars from TXT")
            
        else:
            logger.error(f"Unsupported file extension: {ext}")
            return None
            
    except Exception as e:
        logger.error(f"Error extracting text from file: {e}", exc_info=True)
        return None
        
    return text if text.strip() else None


def analyze_question_blooms_level(document_text, question_text, selected_blooms_level):
    """
    Enhanced Bloom's level analysis based on precise cognitive process distinctions.
    """
    logger.info(f"Analyzing question for Bloom's level: {selected_blooms_level}")
    
    try:
        # Revised Bloom's level mapping with precise distinctions
        blooms_processes = {
    'Remembering': {
        'keywords': ['recall', 'list', 'define', 'identify', 'name', 'state', 'recognize', 'select', 'match', 'memorize', 'what is the', 'what are the', 'who was', 'when did', 'where is'],
        # MORE SPECIFIC PATTERNS: Only catch simple, factual "what is" questions.
        # Use ^ to anchor at start, and avoid patterns that lead to complex comparisons.
        'question_patterns': [
            r'^what is the (name|value|date|capital|definition)', # Specific factual requests
            r'^what are the (three|four|key) (steps|parts|types)', # Requesting a list
            r'^who (is|was) (the|a)', # Simple "who" questions
            r'^when did (the|.*) (happen|occur|start)', # Simple "when" questions
            r'^where is (the|.*) (located|found)', # Simple "where" questions
            r'^name (all|the|three)', # Direct command to list
            r'^list (the|all|three)', # Direct command to list
            r'^define', # Direct command to define
            r'^what was the (first|last|result)' # Factual recall of an outcome
        ],
        'verbs': ['is', 'are', 'was', 'were', 'did', 'does', 'has', 'have'],
        'weight': 1.0,
        'complexity_threshold': 0.2
    },
    'Understanding': {
    'keywords': ['explain', 'summarize', 'interpret', 'paraphrase', 'classify', 'describe', 'discuss', 'restate', 'translate', 'outline', 'main idea', 'in your own words', 'difference between'],
    # ADDED: 'difference between' - for basic conceptual distinctions
    'question_patterns': [
        r'explain (why|how)', 
        r'summarize',
        r'what does (.*) mean',
        r'in your own words',
        r'how would you describe',
        r'what is the main idea',
        r'what is the purpose of',
        r'give an example of',
        r'what can you infer from',
        r'how does (.*) work',
        r'how (.*) works',
        r'why is (.*) important',
        r'how would you explain (.*)',
        r'what happens when',
        # ADDED: Basic difference questions
        r'what is the difference between', # For basic conceptual distinctions
        r'what are the differences between', # For basic conceptual distinctions
        r'distinguish between', # Direct instruction for basic differentiation
        r'how is (.*) different from', # Comparative understanding
    ],
    'verbs': ['explain', 'describe', 'summarize', 'interpret', 'discuss', 'paraphrase', 'distinguish', 'breakdown'],
    'weight': 1.2,
    'complexity_threshold': 0.4
},
    'Applying': {
        'keywords': ['use', 'apply', 'implement', 'solve', 'demonstrate', 'show how', 'employ', 'illustrate', 'execute', 'calculate', 'model'],
        'question_patterns': [
            r'how would you use (.*) to', # Asks for application of a tool/concept
            r'what would happen if', # Asks to predict an outcome based on rules
            r'how would you solve (this|the following) problem', # Direct problem-solving
            r'demonstrate how (to|you)', # Asks for a demonstration of a procedure
            r'apply (the|this) (principle|rule|law) (to|for)', # Direct application
            r'solve for', # Mathematical/scientific application
            r'use (.*) to (show|demonstrate|solve)', # Using a tool for a task
            r'calculate the', # Numerical application
            r'perform (the|a) (calculation|procedure|experiment)', 
            r'carry out (the|this) (task|process)',
        ],
        'verbs': ['use', 'apply', 'solve', 'demonstrate', 'implement', 'calculate'],
        'weight': 1.3,
        'complexity_threshold': 0.5
    },
    'Analyzing': {
    'keywords': ['analyze', 'compare', 'contrast', 'differentiate', 'examine', 'investigate', 'categorize', 'organize', 'deduce', 'distinguish', 'relationship', 'cause', 'effect', 'similarities between'],

    'question_patterns': [
        r'what are the similarities between', # Compare/contrast
        r'compare and contrast', # Direct instruction
        r'analyze (how|why)', # Requests analysis of process or reason
        r'why do you think', # Requests analysis of motives or causes
        r'what evidence supports', # Requests analysis of supporting details
        r'how is (.*) related to', # Requests analysis of relationships
        r'what factors contribute to', # Requests analysis of causal factors
        r'break down', # Direct instruction to analyze components
        r'examine the causes of', # Requests causal analysis
        r'what is the relationship between', # Requests relational analysis
        r'why is (.*) different from', # explicit differentiation
        r'how would you categorize (.*)', # classification
        r'what components make up (.*)', # decomposition
        r'what is the underlying cause of', # deeper causal analysis
        r'how does (.*) influence (.*)', # relational analysis
        r'what assumptions underlie (.*)', # critical analysis
    ],
    'verbs': ['analyze', 'compare', 'contrast', 'examine', 'investigate', 'categorize', 'differentiate'],
    'weight': 1.4,
    'complexity_threshold': 0.7
},
    'Evaluating': {
        'keywords': ['evaluate', 'judge', 'critique', 'justify', 'defend', 'argue', 'assess', 'rate', 'recommend', 'appraise', 'prioritize', 'opinion', 'do you agree'],
        'question_patterns': [
            r'do you agree (with|that)', # Requests a judgment and justification
            r'what is your opinion (on|about)', # Requests a personal judgment
            r'how effective (is|was)', # Requests an assessment of effectiveness
            r'justify your (answer|position)', # Requests defense of a stance
            r'defend your (position|argument)', # Requests defense of a stance
            r'critique (the|this)', # Requests critical assessment
            r'evaluate (the|this) (decision|method)', # Requests an evaluation
            r'which is (better|more effective)', # Requests a comparative judgment
            r'what would you recommend', # Requests a justified suggestion
            r'assess the (value|validity)', # Requests an assessment
            r'rate the importance of', # Requests a prioritized judgment
            r'which option (is|would be) best',     # choice + justification
            r'how would you improve (.*)',          # evaluative + constructive
            r'do the benefits outweigh the risks',  # judgment calls

        ],
        'verbs': ['evaluate', 'judge', 'critique', 'justify', 'defend', 'assess', 'recommend'],
        'weight': 1.5,
        'complexity_threshold': 0.8
    },
    'Creating': {
        'keywords': ['create', 'design', 'develop', 'generate', 'produce', 'hypothesize', 'plan', 'construct', 'invent', 'compose', 'formulate', 'propose', 'what if'],
        'question_patterns': [
            r'how would you design', # Requests original design
            r'what would you create', # Requests original creation
            r'can you propose (an|a)', # Requests a proposal
            r'develop a (plan|model|solution)', # Requests development of a new plan
            r'create a (solution|product|story)', # Requests creation of something new
            r'hypothesize what would happen if', # Requests forming a hypothesis
            r'design (a|an)', # Direct instruction to design
            r'invent (a|an)', # Direct instruction to invent
            r'compose (a|an)', # Direct instruction to compose
            r'formulate a (theory|plan)', # Direct instruction to formulate
            r'what if you could', # Prompts creative thinking
            r'how would you modify (.*) to',        # creation by modification
            r'what new (method|model|idea) could',  # explicit novelty
            r'develop an alternative to (.*)',      # alternative creation
        ],
        'verbs': ['create', 'design', 'develop', 'propose', 'invent', 'hypothesize', 'construct'],
        'weight': 1.6,
        'complexity_threshold': 0.9
    }
}
        
        question_lower = question_text.lower().strip()
        document_lower = document_text.lower() if document_text else ""
        
        # Preprocess question to remove extra spaces and punctuation
        question_clean = re.sub(r'[^\w\s]', ' ', question_lower)
        question_clean = re.sub(r'\s+', ' ', question_clean).strip()
        
        # Score each Bloom's level with multiple strategies
        level_scores = {}
        for level, data in blooms_processes.items():
            score = 0
            
            # 1. Exact keyword matching with context awareness
            for keyword in data['keywords']:
                # Use word boundaries to avoid partial matches
                if re.search(r'\b' + re.escape(keyword) + r'\b', question_clean):
                    score += 1.5 * data['weight']
            
            # 2. Pattern matching with regex (higher confidence)
            for pattern in data['question_patterns']:
                if re.search(pattern, question_lower):
                    score += 2.0 * data['weight']
            
            # 3. Verb analysis (check the first word or early words for a verb)
            question_words = question_clean.split()
            # Check the first 2-3 words for a key verb
            for i, word in enumerate(question_words[:3]):
                if word in data['verbs']:
                    score += 1.0 * data['weight']
                    break
            
            # 4. Document context analysis - STRONGLY weighted for higher-order thinking
            if document_text:
                doc_score = _analyze_document_context(question_text, document_lower, level)
                # Applying, Analyzing, Evaluating, Creating benefit more from a source document
                if level in ['Applying', 'Analyzing', 'Evaluating', 'Creating']:
                    doc_score *= 1.5
                score += doc_score
            
            # 5. Question complexity analysis
            complexity_score = _analyze_question_complexity(question_text, level, data['complexity_threshold'])
            score += complexity_score
            
            level_scores[level] = max(score, 0)
        
        # Handle edge cases and special patterns
        level_scores = _handle_special_cases(question_text, level_scores)
        
        # Normalize scores and determine the detected level
        detected_level, confidence = _determine_level_with_confidence(level_scores)
        
        # Additional validation checks
        confidence = _validate_classification(confidence, question_text, detected_level, document_text)
        
        # Check if it matches the selected level
        matches = (detected_level == selected_blooms_level)
        
        # Generate detailed explanation
        explanation = _generate_detailed_explanation(
            question_text, detected_level, selected_blooms_level, 
            matches, level_scores, document_text, confidence
        )
        
        return {
            "matches_level": matches,
            "confidence": round(confidence, 2),
            "actual_level": detected_level,
            "explanation": explanation,
            "scores": {k: round(v, 2) for k, v in level_scores.items()}
        }
        
    except Exception as e:
        logger.error(f"Error in Bloom's level analysis: {e}", exc_info=True)
        return {
            "matches_level": False,
            "confidence": 0.0,
            "actual_level": "Error",
            "explanation": f"Analysis failed: {str(e)}",
            "scores": {}
        }

def _analyze_document_context(question_text, document_text, level):
    """
    Analyzes if the question requires document content analysis and scores accordingly.
    Higher scores for levels that require sourcing from the text (Analyze, Evaluate).
    """
    score = 0
    question_lower = question_text.lower()
    
    # Document reference patterns - crucial for Analyze/Evaluate
    doc_references = [
        r'according to (the|this) (text|document|passage|reading|author)',
        r'based on (the|this) (text|document|passage|reading)',
        r'from the (text|document|passage|reading)',
        r'as described in',
        r'as stated in',
        r'the (text|document|passage|reading) (says|states|describes|implies)',
        r'what evidence in the text'
    ]
    
    for pattern in doc_references:
        if re.search(pattern, question_lower):
            # This is a strong indicator of Analysis or Evaluation
            if level in ['Analyzing', 'Evaluating']:
                score += 2.5
            else:
                score += 1.0
            break
    
    # For Creating, check if it asks to extend or modify something *from the document*
    if level == 'Creating':
        if document_text and ('create a new' in question_lower or 'design an alternative' in question_lower):
            # Check if the question references a concept from the doc
            content_terms = re.findall(r'\b([A-Z][a-z]+|[0-9]+)\b', question_text)
            for term in content_terms:
                if term.lower() in document_text:
                    score += 1.5
                    break
    
    return score

def _analyze_question_complexity(question_text, level, threshold):
    """
    Analyzes question complexity and adjusts score based on level expectations.
    """
    words = question_text.split()
    word_count = len(words)
    
    # Calculate complexity metrics
    sentence_complexity = min(word_count / 10, 1.0)
    
    # Check for complex sentence structures indicative of higher-order thinking
    has_subordination = any(word in question_text.lower() for word in ['because', 'although', 'while', 'if', 'when', 'unless', 'since'])
    has_coordination = any(word in question_text.lower() for word in ['and', 'but', 'or', 'however', 'therefore', 'thus'])
    has_conditionals = any(word in question_text.lower() for word in ['would', 'could', 'might', 'should'])
    
    structural_complexity = 0.4 if has_subordination else 0.1
    structural_complexity += 0.2 if has_coordination else 0.0
    structural_complexity += 0.2 if has_conditionals else 0.0
    
    total_complexity = (sentence_complexity + structural_complexity) / 2
    
    # Adjust score based on whether complexity matches level expectations
    # e.g., a highly complex question is unlikely to be just 'Remembering'
    if total_complexity >= threshold:
        return 0.8
    elif total_complexity < threshold - 0.2:
        return -0.8  # Stronger penalty for mismatch
    else:
        return 0.0

def _handle_special_cases(question_text, level_scores):
    """
    Handles special cases and edge patterns that might confuse the classifier.
    """
    question_lower = question_text.lower()
    
    # "What did X do" or "What happened" are almost always Remembering
    if re.search(r'^what (did|happened)', question_lower):
        level_scores['Remembering'] += 2.5
        for level in ['Understanding', 'Applying', 'Analyzing', 'Evaluating', 'Creating']:
            level_scores[level] = max(level_scores[level] - 1.5, 0)
    
    # Questions starting with "How" require careful parsing
    if question_lower.startswith('how '):
        if 'how to' in question_lower: # Applying
            level_scores['Applying'] += 1.5
        elif 'how would' in question_lower or 'how could' in question_lower: # Applying or Creating
            level_scores['Applying'] += 1.0
            level_scores['Creating'] += 1.0
        elif 'how does' in question_lower or 'how did' in question_lower: # Understanding or Analyzing
            level_scores['Understanding'] += 1.0
            level_scores['Analyzing'] += 0.5
    
    # Questions with "Why" typically indicate Understanding or Analyzing
    if question_lower.startswith('why '):
        level_scores['Understanding'] += 1.0
        level_scores['Analyzing'] += 1.5 # Weighted more towards Analysis
    
    return level_scores

def _determine_level_with_confidence(level_scores):
    """
    Determines the Bloom's level with confidence scoring.
    """
    if not level_scores or sum(level_scores.values()) == 0:
        return "Remembering", 0.5
    
    max_score = max(level_scores.values())
    detected_level = max(level_scores, key=level_scores.get)
    
    total_score = sum(level_scores.values())
    if total_score == 0:
        return detected_level, 0.5
    
    confidence = max_score / total_score
    
    sorted_scores = sorted(level_scores.values(), reverse=True)
    if len(sorted_scores) > 1:
        score_gap = sorted_scores[0] - sorted_scores[1]
        gap_ratio = score_gap / max(1, sorted_scores[0])
        confidence = min(confidence + gap_ratio * 0.3, 0.95)
    
    return detected_level, confidence

def _validate_classification(confidence, question_text, detected_level, document_text):
    """
    Performs additional validation checks on the classification.
    """
    words = question_text.split()
    word_count = len(words)
    
    if word_count < 4:
        confidence *= 0.7
    
    if word_count > 15:
        confidence = min(confidence * 1.1, 0.95)
    
    if '?' in question_text:
        confidence = min(confidence * 1.05, 0.95)
    
    # High confidence if a complex question is NOT classified as Remembering
    if detected_level != 'Remembering' and word_count > 8:
        confidence = min(confidence * 1.1, 0.95)
    
    if document_text and len(document_text) > 100:
        confidence = min(confidence * 1.05, 0.95)
    
    return max(0.1, min(confidence, 0.95))

def _generate_detailed_explanation(question, detected_level, selected_level, matches, scores, document_text, confidence):
    """
    Generates a comprehensive explanation of the analysis.
    """
    if matches:
        base = f"✓ CORRECT. The question is classified as '{detected_level}'."
    else:
        base = f"✗ INCORRECT. The question is classified as '{detected_level}', not '{selected_level}'."
    
    conf_text = f" Confidence: {confidence*100:.1f}%."
    score_text = " Score breakdown: " + ", ".join([f"{l}:{s:.1f}" for l, s in scores.items()]) + "."
    
    word_count = len(question.split())
    quality = "Well-structured" if word_count > 6 and '?' in question else "Needs more specificity"
    quality_text = f" Question quality: {quality} ({word_count} words)."
    
    doc_text = " Document context was utilized." if document_text and len(document_text) > 0 else " No document provided for context."
    
    reasoning = _get_classification_reasoning(question, detected_level, scores)
    
    return f"{base}{conf_text}{score_text}{quality_text}{doc_text} {reasoning}"

def _get_classification_reasoning(question, detected_level, scores):
    """
    Provides specific reasoning for the classification based on Bloom's distinctions.
    """
    question_lower = question.lower()
    reasoning = "Reasoning: "

    if detected_level == 'Remembering':
        if re.search(r'\b(who|what|when|where|list|name|define)\b', question_lower):
            reasoning += "Question asks for direct recall of factual information."
        else:
            reasoning += "Question structure and keywords indicate a request for retrieval of known information."

    elif detected_level == 'Understanding':
        if re.search(r'\b(explain|summarize|in your own words|main idea)\b', question_lower):
            reasoning += "Question requires explaining ideas or concepts, not just recalling them."
        else:
            reasoning += "Question asks for interpretation or demonstration of comprehension."

    elif detected_level == 'Applying':
        if re.search(r'\b(use|apply|solve|demonstrate|calculate)\b', question_lower):
            reasoning += "Question requires using knowledge or a procedure in a specific situation or problem."
        else:
            reasoning += "Question prompts the application of learned material in a new context."

    elif detected_level == 'Analyzing':
        if re.search(r'\b(difference|analyze|compare|contrast|relationship|cause|effect)\b', question_lower):
            reasoning += "Question requires breaking down information into parts and examining relationships."
        else:
            reasoning += "Question prompts deconstruction of concepts to find underlying structure or motives."

    elif detected_level == 'Evaluating':
        if re.search(r'\b(evaluate|judge|critique|justify|defend|opinion)\b', question_lower):
            reasoning += "Question requires making a judgment based on criteria and standards."
        else:
            reasoning += "Question prompts justification of a decision or critical assessment of a value."

    elif detected_level == 'Creating':
        if re.search(r'\b(create|design|develop|propose|invent|what if)\b', question_lower):
            reasoning += "Question requires synthesizing elements into a new, coherent whole or proposing original ideas."
        else:
            reasoning += "Question prompts the generation of new ideas, products, or ways of viewing things."

    # Add a note if the decision was close
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    if len(sorted_scores) > 1:
        first_score = sorted_scores[0][1]
        second_score = sorted_scores[1][1]
        if first_score - second_score < 1.0: # Scores were close
            next_best = sorted_scores[1][0]
            reasoning += f" This was a close decision with '{next_best}'."

    return reasoning
    """
    Provides specific reasoning for the classification decision.
    """
    question_lower = question.lower()
    reasoning = []
    
    if detected_level == 'Remembering':
        if any(word in question_lower for word in ['what', 'who', 'when', 'where']):
            reasoning.append("Uses basic information-seeking words")
        if 'list' in question_lower or 'name' in question_lower:
            reasoning.append("Requests specific factual recall")
    
    elif detected_level == 'Understanding':
        if 'explain' in question_lower:
            reasoning.append("Asks for explanation of concepts")
        if 'describe' in question_lower:
            reasoning.append("Requests description rather than simple recall")
    
    # Add more specific reasoning for other levels...
    
    if reasoning:
        return "Reasoning: " + "; ".join(reasoning) + "."
    return "Classification based on keyword and pattern analysis."


@main_blueprint.route('/check-blooms-level', methods=['POST'])
def check_blooms_level():
    """Handles the Bloom's level checking process."""
    logger.info("Received Bloom's level check request")
    
    if 'file' not in request.files:
        logger.error("No file part in request")
        flash('No file part in the request')
        return redirect(url_for('main.blooms_checker'))
    
    file = request.files['file']
    logger.debug(f"File received: {file.filename}")

    if file.filename == '':
        logger.error("No file selected")
        flash('No file selected')
        return redirect(url_for('main.blooms_checker'))
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(Config.UPLOAD_FOLDER, filename)
        
        logger.info(f"Processing file: {filename}")
        
        # Ensure upload directory exists
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        file.save(file_path)
        logger.debug(f"File saved to: {file_path}")

        # Extract text from the document
        document_text = extract_text_from_file(file_path)
        logger.debug(f"Extracted text length: {len(document_text) if document_text else 0}")

        if not document_text:
            logger.error("Could not extract text from file")
            flash('Could not extract text from the uploaded file')
            return redirect(url_for('main.blooms_checker'))

        # Get form data - FIXED: Use correct field names from your form
        question_text = request.form.get('question', '').strip()
        blooms_level = request.form.get('blooms_level', '').strip()
        
        if not question_text:
            logger.error("No question provided")
            flash('Please provide a question to analyze')
            return redirect(url_for('main.blooms_checker'))
            
        if not blooms_level:
            logger.error("No Bloom's level selected")
            flash('Please select a Bloom\'s Taxonomy level')
            return redirect(url_for('main.blooms_checker'))
            
        # Map Bloom's level codes to full names - FIXED: Match your form values
        blooms_mapping = {
            'remember': 'Remembering',
            'understand': 'Understanding', 
            'apply': 'Applying',
            'analyze': 'Analyzing',
            'evaluate': 'Evaluating',
            'create': 'Creating'
        }
        
        selected_blooms_name = blooms_mapping.get(blooms_level, blooms_level)
        
        # Analyze the question using AI
        logger.info("Analyzing question with AI model...")
        analysis_result = analyze_question_blooms_level(
            document_text, question_text, selected_blooms_name
        )
        
        logger.info(f"Analysis complete: {analysis_result}")
        
        # Render results
        return render_template('blooms_result.html', 
                             question=question_text,
                             selected_level=selected_blooms_name,
                             analysis=analysis_result,
                             filename=filename)
            
    logger.error(f"Invalid file format: {file.filename}")
    flash('Invalid file format. Please upload PDF, TXT, or DOCX files.')
    return redirect(url_for('main.blooms_checker'))
    """Handles the Bloom's level checking process."""
    logger.info("Received Bloom's level check request")
    
    if 'file' not in request.files:
        logger.error("No file part in request")
        flash('No file part in the request')
        return redirect(url_for('main.blooms_checker'))
    
    file = request.files['file']
    logger.debug(f"File received: {file.filename}")

    if file.filename == '':
        logger.error("No file selected")
        flash('No file selected')
        return redirect(url_for('main.blooms_checker'))
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(Config.UPLOAD_FOLDER, filename)
        
        logger.info(f"Processing file: {filename}")
        
        # Ensure upload directory exists
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        file.save(file_path)
        logger.debug(f"File saved to: {file_path}")

        # Extract text from the document
        document_text = extract_text_from_file(file_path)
        logger.debug(f"Extracted text length: {len(document_text) if document_text else 0}")

        if not document_text:
            logger.error("Could not extract text from file")
            flash('Could not extract text from the uploaded file')
            return redirect(url_for('main.blooms_checker'))

        # Get form data
        question_text = request.form.get('question', '').strip()
        blooms_level = request.form.get('blooms_level', '').strip()
        
        if not question_text:
            logger.error("No question provided")
            flash('Please provide a question to analyze')
            return redirect(url_for('main.blooms_checker'))
            
        if not blooms_level:
            logger.error("No Bloom's level selected")
            flash('Please select a Bloom\'s Taxonomy level')
            return redirect(url_for('main.blooms_checker'))
            
        # Map Bloom's level codes to full names
        blooms_mapping = {
            'remember': 'Remembering',
            'understand': 'Understanding', 
            'apply': 'Applying',
            'analyze': 'Analyzing',
            'evaluate': 'Evaluating',
            'create': 'Creating'
        }
        
        selected_blooms_name = blooms_mapping.get(blooms_level, blooms_level)
        
        # Analyze the question using AI - FIXED: Use the correct function
        logger.info("Analyzing question with AI model...")
        analysis_result = analyze_question_blooms_level_ai(  # CHANGED THIS LINE
            document_text, question_text, selected_blooms_name
        )
        
        logger.info(f"Analysis complete: {analysis_result}")
        
        # Render results
        return render_template('blooms_result.html', 
                             question=question_text,
                             selected_level=selected_blooms_name,
                             analysis=analysis_result,
                             filename=filename)
            
    logger.error(f"Invalid file format: {file.filename}")
    flash('Invalid file format. Please upload PDF, TXT, or DOCX files.')
    return redirect(url_for('main.blooms_checker'))
    """Handles the Bloom's level checking process."""
    logger.info("Received Bloom's level check request")
    
    if 'file' not in request.files:
        logger.error("No file part in request")
        flash('No file part in the request')
        return redirect(url_for('main.blooms_checker'))
    
    file = request.files['file']
    logger.debug(f"File received: {file.filename}")

    if file.filename == '':
        logger.error("No file selected")
        flash('No file selected')
        return redirect(url_for('main.blooms_checker'))
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(Config.UPLOAD_FOLDER, filename)
        
        logger.info(f"Processing file: {filename}")
        
        # Ensure upload directory exists
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        file.save(file_path)
        logger.debug(f"File saved to: {file_path}")

        # Extract text from the document
        document_text = extract_text_from_file(file_path)
        logger.debug(f"Extracted text length: {len(document_text) if document_text else 0}")

        if not document_text:
            logger.error("Could not extract text from file")
            flash('Could not extract text from the uploaded file')
            return redirect(url_for('main.blooms_checker'))

        # Get form data
        question_text = request.form.get('question', '').strip()
        blooms_level = request.form.get('blooms_level', '').strip()
        
        if not question_text:
            logger.error("No question provided")
            flash('Please provide a question to analyze')
            return redirect(url_for('main.blooms_checker'))
            
        if not blooms_level:
            logger.error("No Bloom's level selected")
            flash('Please select a Bloom\'s Taxonomy level')
            return redirect(url_for('main.blooms_checker'))
            
        # Map Bloom's level codes to full names
        blooms_mapping = {
            'remember': 'Remembering',
            'understand': 'Understanding', 
            'apply': 'Applying',
            'analyze': 'Analyzing',
            'evaluate': 'Evaluating',
            'create': 'Creating'
        }
        
        selected_blooms_name = blooms_mapping.get(blooms_level, blooms_level)
        
        # Analyze the question using AI
        logger.info("Analyzing question with AI model...")
        analysis_result = analyze_question_blooms_level(
            document_text, question_text, selected_blooms_name
        )
        
        logger.info(f"Analysis complete: {analysis_result}")
        
        # Render results
        return render_template('blooms_result.html', 
                             question=question_text,
                             selected_level=selected_blooms_name,
                             analysis=analysis_result,
                             filename=filename)
            
    logger.error(f"Invalid file format: {file.filename}")
    flash('Invalid file format. Please upload PDF, TXT, or DOCX files.')
    return redirect(url_for('main.blooms_checker'))




