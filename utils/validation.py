import logging

logger = logging.getLogger('validation')

def validate_num_questions(num_questions_str, min_val=1, max_val=20):
    """Validates the number of questions parameter"""
    try:
        num_questions = int(num_questions_str)
        if num_questions < min_val or num_questions > max_val:
            logger.error(f"Invalid number of questions: {num_questions}")
            return None, f"Number of questions must be between {min_val} and {max_val}"
        return num_questions, None
    except ValueError:
        logger.error(f"Invalid number format: {num_questions_str}")
        return None, "Invalid number of questions"

def validate_model_choice(model_choice, available_models):
    """Validates the model choice parameter"""
    if not model_choice or model_choice not in available_models:
        logger.error(f"Invalid model selection: {model_choice}")
        return None, "Invalid model selection"
    return available_models[model_choice], None


def validate_file_and_params(uploaded_file, num_questions_str, max_questions, allowed_extensions):
    """
    Validates the uploaded file and question generation parameters.
    Returns error message string or None if valid.
    """
    # 1. File existence check
    if not uploaded_file or uploaded_file.filename == '':
        logger.warning("File upload missing or empty filename.")
        return "No file selected."

    # 2. File extension check
    filename = uploaded_file.filename
    # Simple check for the extension
    if '.' not in filename:
        logger.warning(f"File {filename} missing extension.")
        return "Invalid file: missing file extension."
        
    file_ext = filename.rsplit('.', 1)[1].lower()
    if file_ext not in allowed_extensions:
        error_msg = f"Invalid file type ({file_ext}). Allowed types are: {', '.join(allowed_extensions)}"
        logger.warning(f"File extension check failed: {error_msg}")
        return error_msg

    # 3. Question number check
    # We only need the error status, not the parsed number, since routes.py parses it later.
    _, error = validate_num_questions(num_questions_str, max_val=max_questions)
    if error:
        return error
        
    logger.info("File and parameters validated successfully.")
    return None # Validation passed