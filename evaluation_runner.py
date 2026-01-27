import sys
import os
# Dynamically add the project root to the path to resolve 'app' and 'utils' imports
# Assumes the script is run from the project root or its immediate subdirectory.
# If running from the project root:
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# If running from a subdirectory (e.g., 'scripts/'):
# project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
# sys.path.append(project_root)

import uuid
import logging.config
import time
import pandas as pd
from typing import Dict, Any, List

# --- Import from existing project structure ---
try:
    # NOTE: You will need a 'config.py' file in your root directory 
    # for this import to succeed in a real environment.
    from config import Config 
    from utils.file_utils import extract_text_from_pdf 
    from app.services.llm_evaluation_service import LLMEvaluationService
    from utils.evaluation_utils import calculate_derived_metrics, generate_visualizations, generate_html_report
    from utils.pydantic_schema import EvaluationData
    from app.services.llm_models import get_registered_models

    # Placeholder for configuration if Config is not available
    RUN_CONFIG = {
        'UPLOAD_FOLDER': 'uploads',
        'RESULTS_FOLDER': 'results',
        'CONTEXT_FILE': 'context_doc.pdf',
        'NUM_QUESTIONS': 3,
        'QUESTION_TYPE': '1', # '1': MCQ, '4': LA
        'BLOOMS_LEVEL': '3', # '3': Applying
        'LOG_LEVEL': 'DEBUG'
    }

    # Setup Logging (using a basic configuration if Flask/Config isn't present)
    logging.config.dictConfig({
        'version': 1,
        'formatters': {
            'default': {'format': '[%(asctime)s] %(levelname)s in %(name)s: %(message)s'},
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'formatter': 'default',
                'level': RUN_CONFIG['LOG_LEVEL']
            },
        },
        'root': {
            'handlers': ['console'],
            'level': RUN_CONFIG['LOG_LEVEL'],
        },
        'disable_existing_loggers': False,
    })
    logger = logging.getLogger('evaluation_runner')


    def main():
        """Main function to run the LLM evaluation workflow."""
        start_time = time.time()
        run_id = str(uuid.uuid4())
        
        # 1. Initialization and Setup
        logger.info(f"Starting LLM Evaluation Run: {run_id}")
        
        # Ensure result directories exist
        os.makedirs(RUN_CONFIG['RESULTS_FOLDER'], exist_ok=True)
        os.makedirs(RUN_CONFIG['UPLOAD_FOLDER'], exist_ok=True)
        
        # 2. Load Context Text
        context_file_path = os.path.join(RUN_CONFIG['UPLOAD_FOLDER'], RUN_CONFIG['CONTEXT_FILE'])
        
        try:
            context_text = extract_text_from_pdf(context_file_path)
        except Exception as e:
            logger.error(f"Failed to extract text from context file '{context_file_path}'. Check if the file exists and is readable.", exc_info=True)
            return

        if not context_text:
            logger.error(f"Context text is empty after parsing '{context_file_path}'. Exiting.")
            return

        # 3. Instantiate and Run Evaluation Service
        models_to_evaluate = get_registered_models()
        
        eval_service = LLMEvaluationService(
            run_id=run_id,
            context_text=context_text,
            num_questions=RUN_CONFIG['NUM_QUESTIONS'],
            question_type=RUN_CONFIG['QUESTION_TYPE'],
            blooms_level=RUN_CONFIG['BLOOMS_LEVEL'],
            models=models_to_evaluate
        )
        
        raw_results: List[Dict[str, Any]] = eval_service.run_evaluation()

        if not raw_results:
            logger.warning("No successful LLM responses received for comparison.")
            return

        # 4. Process and Analyze Results
        df = pd.DataFrame(raw_results)
        
        df = calculate_derived_metrics(df)
        
        # 5. Generate Visualizations and Report
        # Save charts (e.g., in results/run_id_latency.png)
        chart_paths = generate_visualizations(df, RUN_CONFIG['RESULTS_FOLDER'], run_id)
        
        # Generate the final HTML report
        report_filename = generate_html_report(df, RUN_CONFIG, chart_paths, RUN_CONFIG['RESULTS_FOLDER'], run_id)
        
        end_time = time.time()
        total_time = end_time - start_time

        logger.info(f"--- Evaluation Complete ---")
        logger.info(f"Total Runtime: {total_time:.2f} seconds")
        logger.info(f"Report saved to: {report_filename}")
        logger.info("-" * 25)


    if __name__ == "__main__":
        # NOTE: If running from the project root, this path fixing should work.
        # If you still get errors, ensure the project root is in your IDE/Terminal path.
        main()

except ImportError as e:
    # Fallback for running outside of the full Flask app context
    print(f"Warning: Could not import application modules (Config, file_utils, etc.). Error: {e}")
    print("Please ensure you have all files (config.py, app/services/, utils/) correctly placed in the project structure and necessary packages (pandas, matplotlib, seaborn) are installed.")
    print(f"The specific import error was: {e}")
# End of file generation section
