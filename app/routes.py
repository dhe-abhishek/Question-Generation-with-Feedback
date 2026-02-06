import os
import logging
import re
import time
import qrcode
import io
import base64
import socket
from app.database import db, Question, Quiz, Student, QuizAttempt, StudentResponse, Context, McqOption, TextAnswer, QuestionEvaluation
from app.services.quiz_service import QuizService
from flask import render_template, request, send_file, Blueprint, flash, redirect, url_for, jsonify, session
from config import Config
from utils.file_utils import allowed_file, save_uploaded_file, save_mcq_results, extract_text_from_pdf, save_data_cloud_results, cleanup_file
from app.services.data_cloud_service import generate_data_cloud_from_text
from utils.validation import validate_num_questions, validate_file_and_params
from app.services.pdf_generation import create_pdf # Import the PDF creation utility
from app.services.question_generation import QuestionGenerator # Import the service class
from werkzeug.utils import secure_filename
from utils.question_evaluator_utils import read_questions_from_file, calculate_rouge_l, calculate_meteor, save_scores_to_excel, calculate_sentence_rouge_l
import pdfplumber
from app.services.answer_generation_service import AnswerGenerationService
from app.services.question_coverage_service import QuestionCoverageService
from utils.pdf_extraction_util import extract_text_from_pdf 
from utils.relevancy_utils import calculate_relevancy_score
from utils.faithfulness_utils import calculate_faithfulness_score
from utils.correctness_utils import calculate_correctness_score
from app.services.note_generation_service import NoteGenerationService
from app.services.question_evaluator import QuestionEvaluator




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
        
        # --- 3. Save File Content to Context Table ---
        try:
            new_context = Context(
                file_name=filename,
                file_content=text_content
            )
            db.session.add(new_context)
            db.session.commit()
            
            # CRITICAL FIX: Store the context ID in the session for the reframing service
            session['last_context_id'] = new_context.id
            
            logger.info(f"Successfully saved content for {filename} to Context table.")
        except Exception as db_err:
            db.session.rollback()
            logger.error(f"Database error while saving context: {str(db_err)}")
            # We don't necessarily stop the process if DB fails, but we log it.

        # --- 4. Generate Questions ---
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
        
        # CRITICAL FIX: Store questions and answers in session for the reframer
        session['current_questions'] = questions_list
        session['current_answers'] = answer_key_list
        session.modified = True
        
        # --- 5. Save and Render Results ---
        
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
        
        # =========================================================================
        # FIXED CODE: Trigger LLM Judge Evaluation on 7 Parameters
        # =========================================================================
        try:
            evaluator = QuestionEvaluator()
            saved_ids = []

            # 1. Flatten answer_key_list if it is accidentally nested
            if answer_key_list and isinstance(answer_key_list[0], list):
                answer_key_list = [item for sublist in answer_key_list for item in sublist]

            for i, q_data in enumerate(questions_list):
                # Ensure q_data is a dictionary
                if isinstance(q_data, list) and len(q_data) > 0:
                    q_data = q_data[0]
                
                if not isinstance(q_data, dict):
                    logger.error(f"Expected dict but got {type(q_data)} for question index {i}")
                    continue
                
                # Create Base Question record
                new_q = Question(
                    question_text=q_data.get('question'),
                    blooms_id=blooms_level_choice if blooms_level_choice != 'all' else None,
                    source_document=filename
                )
                db.session.add(new_q)
                db.session.flush() # Get ID before commit
                
                # 2. Retrieve the corresponding answer safely
                # Search for the dictionary in answer_key_list that matches the question number
                ans_text = "No answer provided"
                if isinstance(answer_key_list, list):
                    ans_text = next((a.get('correct_answer') for a in answer_key_list 
                                   if isinstance(a, dict) and a.get('question_number') == q_data.get('question_number')), 
                                   "No answer provided")
                
                # 3. Handle MCQ or Text Answer specific logic
                if question_type == '1':
                    opts = q_data.get('options', {})
                    # Ensure options is a dict if it came back as a list
                    if isinstance(opts, list):
                        # Map list [A, B, C, D] to dict {'A':..., 'B':...}
                        opts = {chr(65+idx): val for idx, val in enumerate(opts)}
                    
                    db.session.add(McqOption(
                        question_id=new_q.id,
                        option_a=opts.get('A'), 
                        option_b=opts.get('B'),
                        option_c=opts.get('C'), 
                        option_d=opts.get('D'),
                        correct_option=q_data.get('correct_option_letter')
                    ))
                else:
                    db.session.add(TextAnswer(question_id=new_q.id, answer_content=ans_text))
            
                # Use unified variable name 'ans_text'
                saved_ids.append((new_q.id, q_data.get('question'), ans_text))

            db.session.commit()
                
            # 4. Scientific Evaluation Trigger
            for q_id, q_text, a_text in saved_ids:
                evaluator.evaluate_and_save(q_id, text_content, q_text, a_text)
            logger.info("Scientific evaluation for QGEval dimensions completed.")

        except Exception as eval_err:
            db.session.rollback()
            logger.error(f"Evaluation trigger failed: {str(eval_err)}", exc_info=True)
        # =========================================================================
        
        logger.info(f"Successfully generated {len(questions_list)} questions.")
        
        # Store metadata in session for display_results and reframing
        session['last_pdf_filename'] = pdf_filename
        session['last_txt_filename'] = txt_filename
        session['last_question_type'] = question_type
        session.modified = True
        
        # Render the results page with download links
        
        return render_template(
            'results.html', 
            questions=questions_list, 
            answers=answer_key_list,
            question_type=question_type,
            text_content=text_content,
            pdf_filename=pdf_filename, 
            txt_filename=txt_filename 
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

@main_blueprint.route('/question-evaluator')
def question_evaluator():
    """Renders the question evaluation page for ROUGE and METEOR scoring."""
    logger.info("Rendering Question Evaluator page")
    return render_template('question_evaluator.html')

@main_blueprint.route('/evaluate_questions', methods=['POST'])
def evaluate_questions():
    """Handles the evaluation of reference and candidate questions."""
    logger.info("Received question evaluation request")
    
    ref_file_path = None
    cand_file_path = None
    
    try:
        # --- 1. File Validation ---
        ref_file = request.files.get('reference_file')
        cand_file = request.files.get('candidate_file')
        
        if not ref_file or not cand_file:
            flash('Both reference and candidate files are required.')
            return redirect(url_for('main.question_evaluator'))
        
        if ref_file.filename == '' or cand_file.filename == '':
            flash('Please select both files.')
            return redirect(url_for('main.question_evaluator'))
        
        # --- 2. Save Files ---
        # Create QUESTIONFILES directory if it doesn't exist
        question_files_dir = os.path.join(Config.UPLOAD_FOLDER, 'QUESTIONFILES')
        os.makedirs(question_files_dir, exist_ok=True)
        
        # Save reference file
        ref_filename = secure_filename(ref_file.filename)
        ref_file_path = os.path.join(question_files_dir, ref_filename)
        ref_file.save(ref_file_path)
        
        # Save candidate file
        cand_filename = secure_filename(cand_file.filename)
        cand_file_path = os.path.join(question_files_dir, cand_filename)
        cand_file.save(cand_file_path)
        
        # --- 3. Read and Process Files ---
        ref_questions = read_questions_from_file(ref_file_path)
        cand_questions = read_questions_from_file(cand_file_path)
        
        if not ref_questions:
            flash('Reference file is empty or could not be read.')
            cleanup_file(ref_file_path)
            cleanup_file(cand_file_path)
            return redirect(url_for('main.question_evaluator'))
        
        if not cand_questions:
            flash('Candidate file is empty or could not be read.')
            cleanup_file(ref_file_path)
            cleanup_file(cand_file_path)
            return redirect(url_for('main.question_evaluator'))
        
        # --- 4. Calculate Scores ---
        #rouge_l_score = calculate_rouge_l(cand_questions, ref_questions)
        #meteor_score = calculate_meteor(cand_questions, ref_questions)
        rouge_l_score = calculate_sentence_rouge_l(cand_questions, ref_questions)
        meteor_score = calculate_meteor(cand_questions, ref_questions)
        
        # --- 5. Save Results to Excel ---
        excel_path = save_scores_to_excel(ref_filename, cand_filename, rouge_l_score, meteor_score)
        
        logger.info(f"Evaluation completed - ROUGE-L: {rouge_l_score:.4f}, METEOR: {meteor_score:.4f}")
        
        # --- 6. Render Results ---
        return render_template('evaluation_results.html',
                             rouge_score=f"{rouge_l_score:.4f}",
                             meteor_score=f"{meteor_score:.4f}",
                             ref_filename=ref_filename,
                             cand_filename=cand_filename,
                             num_ref_questions=len(ref_questions),
                             num_cand_questions=len(cand_questions),
                             excel_filename=os.path.basename(excel_path))
        
    except Exception as e:
        logger.error(f"Error during question evaluation: {str(e)}", exc_info=True)
        # Cleanup files in case of error
        cleanup_file(ref_file_path)
        cleanup_file(cand_file_path)
        flash(f'An error occurred during evaluation: {str(e)}')
        return redirect(url_for('main.question_evaluator'))

@main_blueprint.route('/answer-generator', methods=['GET'])
def answer_generator_ui():
    return render_template("answer_generator.html")

@main_blueprint.route('/generate-answer', methods=['POST'])
def generate_answer():
    file = request.files.get('file')
    question = request.form.get('question')
    blooms_level = request.form.get('blooms_level')

    logger.info(f"Answer generation request - Question: {question}, Bloom's level: {blooms_level}")

    if not file or file.filename == '':
        flash("Please upload a PDF file")
        return redirect(url_for("main.answer_generator_ui"))

    file_path, filename, error = save_uploaded_file(file)
    if error:
        flash(error)
        return redirect(url_for("main.answer_generator_ui"))

    logger.info(f"File saved: {file_path}")

    answer, error = AnswerGenerationService.generate_answer_from_context(
        file_path, question, blooms_level
    )

    logger.info(f"Answer generation result - Answer: {answer}, Error: {error}")

    cleanup_file(file_path)

    if error:
        flash(error)
        return redirect(url_for("main.answer_generator_ui"))

    # Pass the form data back to template to preserve the selected Bloom's level
    return render_template("answer_generator.html", 
                         answer=answer,
                         question=question,
                         blooms_level=blooms_level)
    
    
    
@main_blueprint.route('/question-coverage-analysis', methods=['GET'])
def question_coverage_analysis_ui():
    """Renders the Question Coverage Analysis page."""
    logger.info("Rendering Question Coverage Analysis page")
    return render_template('question_coverage_analysis.html')

@main_blueprint.route('/calculate-coverage', methods=['POST'])
def calculate_coverage():
    """Calculates question relevance scores using TF-IDF."""
    logger.info("Starting question coverage calculation request.")
    
    context_file = request.files.get('context_file')
    questions_file = request.files.get('questions_file')

    temp_files = [] # To keep track of files for cleanup

    if not context_file or context_file.filename == '':
        flash("Please upload the Context Material PDF.", 'error')
        return redirect(url_for('main.question_coverage_analysis_ui'))

    if not questions_file or questions_file.filename == '':
        flash("Please upload the Generated Questions PDF.", 'error')
        return redirect(url_for('main.question_coverage_analysis_ui'))

    # Save Context File
    context_path, _, error = save_uploaded_file(context_file)
    if error:
        flash(f"Context File Upload Error: {error}", 'error')
        return redirect(url_for('main.question_coverage_analysis_ui'))
    temp_files.append(context_path)

    # Save Questions File
    questions_path, _, error = save_uploaded_file(questions_file)
    if error:
        flash(f"Questions File Upload Error: {error}", 'error')
        cleanup_file(context_path)
        return redirect(url_for('main.question_coverage_analysis_ui'))
    temp_files.append(questions_path)

    context_text = None
    questions_text = None

    try:
        # Extract text from Context PDF (Returns text, error)
        # CRITICAL FIX: Correctly unpack the (text, error) tuple
        context_text, context_error = extract_text_from_pdf(context_path)
        if context_error:
            raise ValueError(f"Context Text Extraction Error: {context_error}")
        if not context_text or len(context_text.strip()) < 50:
            raise ValueError("Context PDF is too short or empty after extraction. Please ensure the context file has sufficient content.")

        # Extract text from Questions PDF (Returns text, error)
        # CRITICAL FIX: Correctly unpack the (text, error) tuple
        questions_text, questions_error = extract_text_from_pdf(questions_path)
        if questions_error:
            raise ValueError(f"Questions Text Extraction Error: {questions_error}")
        if not questions_text or len(questions_text.strip()) < 20:
            raise ValueError("Questions PDF is too short or empty after extraction.")
            
        # Parse the questions text into a list of individual questions
        questions_list = QuestionCoverageService.extract_questions_from_text(questions_text)

        if not questions_list:
            raise ValueError("Could not extract individual questions from the Generated Questions PDF. Please check the file format or ensure questions are clearly numbered/separated.")

        # Calculate scores
        results = QuestionCoverageService.calculate_relevance_scores(context_text, questions_list)

        if results is None:
            raise Exception("Failed to calculate relevance scores. Check logs for details.")

        flash(f"Successfully calculated relevance scores for {len(results)} questions.", 'success')
        return render_template('question_coverage_analysis.html', results=results)

    except Exception as e:
        # Log the error with the traceback
        logger.error(f"Error during question coverage analysis: {e}", exc_info=True)
        flash(f"An error occurred during analysis: {str(e)}", 'error')
        return redirect(url_for('main.question_coverage_analysis_ui'))
    
    finally:
        # Cleanup all temporary files
        for f_path in temp_files:
            cleanup_file(f_path)
            
            
            
# In routes.py, locate and replace the two new routes you added:

# ====================================================================
# ANSWER RELEVANCY CHECKER ROUTES
# ====================================================================

@main_blueprint.route('/answer-relevancy-checker')
def answer_relevancy_checker_ui():
    """
    Renders the UI for the Answer Relevancy Analysis feature.
    Passes None for result variables to prevent Jinja2 UndefinedError on initial load.
    """
    logger.info("Rendering Answer Relevancy Checker UI")
    # FIX: Pass all expected template variables with None/empty values
    return render_template(
        'answer_relevancy_checker.html',
        question=None,
        answer=None,
        relevancy_score=None
    )

@main_blueprint.route('/calculate-relevancy-score', methods=['POST'])
def calculate_relevancy_score_route():
    """Handles the POST request, calculates the Ragas Response Relevancy score, and displays results."""
    # NOTE: Function name changed slightly to avoid conflict with the imported utility function.
    question = request.form.get('question')
    answer = request.form.get('answer')
    
    # ----------------------------------------------------
    # INPUT VALIDATION
    # ----------------------------------------------------
    # Pass back existing input in case of validation error
    def render_error_page(message, category='error', q=question, a=answer):
        flash(message, category)
        return render_template('answer_relevancy_checker.html', question=q, answer=a, relevancy_score=None)

    if not question or not answer:
        return render_error_page("Both the Question and the Answer fields are required for relevancy analysis.", 'error')
    
    if len(question.strip()) < 10 or len(answer.strip()) < 10:
        return render_error_page("Please provide meaningful text for both the Question and the Answer.", 'warning')

    logger.info(f"Relevancy calculation requested. Question: {question[:50]}..., Answer: {answer[:50]}...")

    # ----------------------------------------------------
    # CORE LOGIC EXECUTION (RAGAS CALL)
    # ----------------------------------------------------
    relevancy_score = None
    try:
        # Call the synchronous wrapper function from the utility file
        relevancy_score = calculate_relevancy_score(question, answer)
        
        flash(f"Relevancy score calculated successfully!", 'success')
        
        return render_template('answer_relevancy_checker.html',
                               question=question,
                               answer=answer,
                               relevancy_score=relevancy_score)
        
    except RuntimeError as e:
        # Catch the specific error raised by the utility file for Ragas issues
        logger.error(f"Ragas Calculation Failure: {e}", exc_info=True)
        return render_error_page(f"Response Relevancy calculation failed. Error: {str(e)}. Check your Groq API Key and network connection.", 'error')

    except Exception as e:
        # Catch any other unexpected errors
        logger.error(f"Unexpected error during Response Relevancy calculation: {e}", exc_info=True)
        return render_error_page(f"An unexpected error occurred during analysis: {str(e)}.", 'error')


@main_blueprint.route('/faithfulness-checker')
def faithfulness_checker_ui():
    return render_template('faithfulness_checker.html')
    
@main_blueprint.route('/calculate-faithfulness', methods=['POST'])
def calculate_faithfulness_route():
    data = request.json
    question = data.get('question')
    answer = data.get('answer')
    context = data.get('context')

    if not all([question, answer, context]):
        return jsonify({'error': 'Missing required fields'}), 400

    try:
        score = calculate_faithfulness_score(question, answer, context)
        return jsonify({'score': round(score, 4)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

@main_blueprint.route('/calculate-correctness', methods=['POST'])
def calculate_correctness_route():
    data = request.json
    score = calculate_correctness_score(
        data.get('question'), 
        data.get('answer'), 
        data.get('ground_truth')
    )
    return jsonify({'score': round(score, 4)})

@main_blueprint.route('/correctness-checker')
def correctness_checker_ui():
    return render_template('correctness_checker.html')


@main_blueprint.route('/generate-quiz')
def generate_quiz():
    """Renders the interactive quiz generation page."""
    logger.info("Rendering Generate Quiz page")
    return render_template('generate_quiz.html')


@main_blueprint.route('/start-quiz', methods=['POST'])
def start_quiz():
    if 'file' not in request.files:
        flash('No file uploaded')
        return redirect(url_for('main.generate_quiz'))
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected')
        return redirect(url_for('main.generate_quiz'))

    try:
        # 1. Save and Extract Text
        file_path, filename, error = save_uploaded_file(file)
        if error:
            flash(error)
            return redirect(url_for('main.generate_quiz'))

        text_content, extract_error = extract_text_from_pdf(file_path)
        if extract_error or not text_content:
            flash(f"Extraction Error: {extract_error or 'No text found'}")
            cleanup_file(file_path)
            return redirect(url_for('main.generate_quiz'))

        # 2. Process and Save to Database
        # Note: Your QuizService should handle deleting old questions for this filename
        success = QuizService.process_and_save_quiz(text_content, filename)
        cleanup_file(file_path)

        if success:
            # 3. Generate QR Code for Phone Access
            local_ip = get_local_ip()
            # Construct the URL using port 5000 and the local IP
            quiz_url = f"http://{local_ip}:5000/take-quiz/{filename}"
            
            # Create QR code image
            qr = qrcode.QRCode(version=1, box_size=10, border=5)
            qr.add_data(quiz_url)
            qr.make(fit=True)
            img = qr.make_image(fill_color="black", back_color="white")
            
            # Convert to base64 for embedding in HTML
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            qr_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

            # Render a bridge page showing the QR Code to the teacher
            return render_template('quiz_ready.html', 
                                   qr_code=qr_base64, 
                                   quiz_url=quiz_url, 
                                   filename=filename)
        else:
            flash("AI failed to structure the data for your SQL tables.")
            return redirect(url_for('main.generate_quiz'))

    except Exception as e:
        flash(f"Internal Error: {str(e)}")
        return redirect(url_for('main.generate_quiz'))
    
@main_blueprint.route('/take-quiz/<doc_name>')
def take_quiz(doc_name):
    # 1. Find only the LATEST quiz session for this file
    quiz = Quiz.query.filter_by(title=doc_name).order_by(Quiz.created_at.desc()).first()
    
    if not quiz:
        flash("Quiz not found.")
        return redirect(url_for('main.home'))

    # 2. Get questions ONLY belonging to this specific session
    questions = quiz.questions 
    
    return render_template('take_quiz.html', questions=questions, doc_name=doc_name)


""" @main_blueprint.route('/submit-quiz/<doc_name>', methods=['POST'])
def submit_quiz(doc_name):
    # 1. Capture Student Details
    student_info = {
        'name': request.form.get('student_name'),
        'class': request.form.get('student_class'),
        'roll_no': request.form.get('student_roll')
    }

    # 2. Existing Quiz Logic
    from app.database import Question
    questions = Question.query.filter_by(source_document=doc_name).all()
    
    score = 0
    results = []

    for q in questions:
        student_ans = request.form.get(f'q_{q.id}', '').strip()
        
        correct_ans = ""
        if q.mcq_data:
            correct_ans = q.mcq_data.correct_option
        elif q.text_answer:
            correct_ans = q.text_answer.answer_content
            
        is_correct = student_ans.lower() == correct_ans.lower()
        if is_correct:
            score += 1
            
        results.append({
            'text': q.question_text,
            'student': student_ans,
            'correct': correct_ans,
            'is_correct': is_correct
        })

    # 3. Pass student_info to the results page
    return render_template('generatedquiz.html', 
                           score=score, 
                           total=len(questions), 
                           results=results, 
                           student=student_info,
                           doc_name=doc_name)
    """ 

@main_blueprint.route('/submit-quiz/<doc_name>', methods=['POST'])
def submit_quiz(doc_name):
    # 1. Identify Student (Create if not exists)
    roll_no = request.form.get('student_roll')
    student = Student.query.filter_by(roll_no=roll_no).first()
    
    if not student:
        student = Student(
            roll_no=roll_no,
            full_name=request.form.get('student_name'),
            class_name=request.form.get('student_class')
        )
        db.session.add(student)
        db.session.flush()

    # 2. Identify Quiz
    quiz = Quiz.query.filter_by(title=doc_name).order_by(Quiz.created_at.desc()).first()
    if not quiz:
        flash("Quiz context not found.")
        return redirect(url_for('main.home'))

    # 3. Initialize Attempt
    attempt = QuizAttempt(student_id=student.id, quiz_id=quiz.id)
    db.session.add(attempt)
    db.session.flush()

    score = 0
    results = []
    questions = quiz.questions

    for q in questions:
        student_ans = request.form.get(f'q_{q.id}', '').strip()
        
        correct_ans = ""
        if q.mcq_data:
            correct_ans = q.mcq_data.correct_option
        elif q.text_answer:
            correct_ans = q.text_answer.answer_content
            
        is_correct = student_ans.lower() == correct_ans.lower()
        marks = 1.0 if is_correct else 0.0
        if is_correct:
            score += 1
            
        # 4. Save individual response
        response_record = StudentResponse(
            attempt_id=attempt.id,
            question_id=q.id,
            submitted_answer=student_ans,
            is_correct=is_correct,
            marks_obtained=marks
        )
        db.session.add(response_record)

        results.append({
            'text': q.question_text,
            'student': student_ans,
            'correct': correct_ans,
            'is_correct': is_correct
        })

    # 5. Finalize Attempt Score
    attempt.final_score = score
    db.session.commit()

    return render_template('generatedquiz.html', 
                           score=score, 
                           total=len(questions), 
                           results=results, 
                           student={'name': student.full_name, 'class': student.class_name, 'roll_no': student.roll_no},
                           doc_name=doc_name)

def get_local_ip():
    """Gets the local IP address of your computer (e.g., 192.168.1.x)"""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP


@main_blueprint.route('/note-generator', methods=['GET'])
def note_generator_ui():
    """Renders the note generation interface."""
    logger.info("Rendering Note Generator page")
    return render_template('note_generator.html')

@main_blueprint.route('/generate-notes', methods=['POST'])
def generate_notes():
    logger.info("Received request to generate tailored notes")
    file_path = None
    try:
        file = request.files.get('file')
        learner_type = request.form.get('learner_type', 'average')
        links = request.form.get('links', '')
        
        if not file or file.filename == '':
            flash("Main PDF file is required.")
            return redirect(url_for('main.note_generator_ui'))
            
        file_path, filename, error = save_uploaded_file(file)
        if error:
            flash(error)
            return redirect(url_for('main.note_generator_ui'))

        # Extract text using your utility
        text_content, extract_error = extract_text_from_pdf(file_path)
        
        if extract_error:
            flash(f"Error extracting text: {extract_error}")
            return redirect(url_for('main.note_generator_ui'))

        # Set the dynamic filename: original_name_notes
        base_name = os.path.splitext(filename)[0]
        generated_filename = f"{base_name}_notes"

        from app.services.note_generation_service import NoteGenerationService
        note_service = NoteGenerationService()
        
        generated_notes, service_error = note_service.generate_notes(
            text_content=text_content,
            learner_level=learner_type,
            additional_links=links
        )

        if service_error:
            flash(f"AI Generation Error: {service_error}")
            return redirect(url_for('main.note_generator_ui'))

        return render_template(
            'note_results.html',
            notes=generated_notes,
            learner_level=learner_type.capitalize(),
            doc_name=generated_filename # Pass the new filename
        )

    finally:
        if file_path:
            cleanup_file(file_path)
    """Handles the note generation process based on learner level."""
    logger.info("Received request to generate tailored notes")
    
    file_path = None
    try:
        # 1. Retrieve Parameters
        file = request.files.get('file')
        learner_type = request.form.get('learner_type', 'average')
        links = request.form.get('links', '')
        # additional_docs is optional; for simplicity, we focus on the primary PDF content
        
        # 2. Validate and Save Main File
        if not file or file.filename == '':
            flash("Main PDF file is required.")
            return redirect(url_for('main.note_generator_ui'))
            
        file_path, filename, error = save_uploaded_file(file)
        if error:
            flash(error)
            return redirect(url_for('main.note_generator_ui'))

        # 3. Extract Text from PDF (Using your existing utility)
        # Note: extract_text_from_pdf returns (text, error) in your file_utils.py
        text_content, extract_error = extract_text_from_pdf(file_path)
        
        if extract_error:
            logger.error(f"Extraction failed: {extract_error}")
            flash(f"Error extracting text: {extract_error}")
            return redirect(url_for('main.note_generator_ui'))

        # 4. Initialize Service and Generate Content
        from app.services.note_generation_service import NoteGenerationService
        note_service = NoteGenerationService()
        
        generated_notes, service_error = note_service.generate_notes(
            text_content=text_content,
            learner_level=learner_type,
            additional_links=links
        )

        if service_error:
            flash(f"AI Generation Error: {service_error}")
            return redirect(url_for('main.note_generator_ui'))

        # 5. Render Results (You'll need a results template)
        return render_template(
            'note_results.html',
            notes=generated_notes,
            learner_level=learner_type.capitalize(),
            doc_name=filename
        )

    except Exception as e:
        logger.error(f"Unexpected error in generate_notes: {str(e)}", exc_info=True)
        flash(f"An unexpected error occurred: {str(e)}")
        return redirect(url_for('main.note_generator_ui'))
        
    finally:
        # 6. Cleanup File from Uploads Folder
        if file_path:
            cleanup_file(file_path)
            
@main_blueprint.route('/reframe_question', methods=['POST'])
def reframe_question():
    data = request.get_json()
    idx = int(data.get('question_index'))
    reason = data.get('reason')
    
    # 1. Retrieve the question type from the request (sent by JS)
    # Default to '1' (MCQ) only if not provided
    q_type = data.get('question_type', '1')
    
    #print("question_type=",q_type)
    
    # 2. Retrieve existing data from session
    questions = session.get('current_questions', [])
    answers = session.get('current_answers', [])
    context_id = session.get('last_context_id')
    
    if not questions or idx >= len(questions):
        return jsonify({"error": "Question index out of bounds"}), 400

    context_record = Context.query.get(context_id)
    if not context_record:
        return jsonify({"error": "Source context not found"}), 404

    context_record = Context.query.get(context_id)
    generator = QuestionGenerator(Config.GEMINI_MODEL)
    
    # 3. Call the generator with the specific question and its type
    new_q, new_a, error = generator.reframe_question(
        text_content=context_record.file_content, # Using correct DB attribute
        original_question=questions[idx].get('question'),
        original_answer=answers[idx].get('correct_answer'),
        feedback=reason,
        question_type=q_type
    )

    
    if not error:
        # 1. Update the items at the specific index with the new generated data
        questions[idx] = new_q
        answers[idx] = new_a

        # 2. REARRANGE/RECONSTRUCT Questions List to enforce strict key order
        ordered_questions = []
        for i, q in enumerate(questions):
            ordered_questions.append({
                'question_number': i + 1, # Force correct sequential numbering
                'blooms_level': q.get('blooms_level'),
                'question': q.get('question'),
                'options': q.get('options'),
                'correct_option_letter': q.get('correct_option_letter')
            })
            
        ordered_answers = []
        for i, a in enumerate(answers):
            ordered_answers.append({
                'question_number': i + 1,
                'correct_answer': a.get('correct_answer')
            })

        # Print for verification as requested
  

        # 4. Save the ordered lists back to session
        session['current_questions'] = ordered_questions
        session['current_answers'] = ordered_answers
        session.modified = True 
        
        # Use the file_name from the database record
        target_filename = context_record.file_name 
        
        # Ensure target_filename doesn't accidentally get double .pdf
        if target_filename.endswith('.pdf'):
            # Pass the name without the extra extension if your create_pdf adds it
            base_filename = target_filename.rsplit('.', 1)[0]
        else:
            base_filename = target_filename
        
        # 5. Refresh the downloadable files
        from app.services.pdf_generation import create_pdf
    
        
        new_pdf, pdf_err = create_pdf(ordered_questions, ordered_answers, base_filename, q_type)
        new_txt, txt_err = save_mcq_results(ordered_questions, ordered_answers, base_filename)
        
        # CRITICAL FIX: Update session with new filenames
        session['last_pdf_filename'] = new_pdf
        session['last_txt_filename'] = new_txt
        session.modified = True
        
        return jsonify({"status": "success"})
    
    return jsonify({"error": error}), 500

@main_blueprint.route('/display_results')
def display_results():
    # Retrieve data from session
    logger.info(f"Display Results Called")
    questions = session.get('current_questions')
    answers = session.get('current_answers')
    context_id = session.get('last_context_id')
    q_type = session.get('last_question_type', '1')
    pdf_filename = session.get('last_pdf_filename')
    txt_filename = session.get('last_txt_filename')
    ordered_questions = []
    for q in questions:
        # We create a new dict for each item in the exact order requested
        item = {
                'question_number': q.get('question_number'),
                'blooms_level': q.get('blooms_level'),
                'question': q.get('question'),
                'options': q.get('options'),
                'correct_option_letter': q.get('correct_option_letter')
        }
        ordered_questions.append(item)
            
        # 3. REARRANGE/RECONSTRUCT Answers List to enforce strict key order
        ordered_answers = []
    for a in answers:
        item = {
                'question_number': a.get('question_number'),
                'correct_answer': a.get('correct_answer')
        }
        ordered_answers.append(item)
    

    
    if not questions:
        flash("No questions found in session.")
        return redirect(url_for('main.question_generator'))

    # We still need the filename for the download links
    from app.database import Context
    context_record = Context.query.get(context_id)
    filename = context_record.file_name if context_record else "results"
    

    return render_template(
        'results.html',
        questions=ordered_questions,
        answers=ordered_answers,
        question_type=q_type,
        pdf_filename=pdf_filename,
        txt_filename=txt_filename
    )