import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging
import sys
import os
from app import create_app
from app.database import db, QuestionEvaluation
from config import Config # Necessary for the create_app factory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def generate_pearson_matrix():
    logger.info("Step 1: Initializing Flask App context with Config...")
    # Fix: Passing Config to resolve the TypeError
    app = create_app(Config)
    
    with app.app_context():
        logger.info("Step 2: Connecting to the database...")
        try:
            evals = QuestionEvaluation.query.all()
            total_records = len(evals)
            logger.info(f"Step 3: Database connection successful. Found {total_records} records.")
            
            if total_records == 0:
                logger.warning("No evaluation records found. Task aborted.")
                return

            logger.info("Step 4: Extracting data and filtering null values...")
            data = []
            for e in evals:
                # Only append if at least one value is not None
                scores = [e.fluency, e.clarity, e.conciseness, e.relevance, 
                          e.consistency, e.answerability, e.answer_consistency]
                
                if any(v is not None for v in scores):
                    data.append({
                        'Flu.': e.fluency,
                        'Clar.': e.clarity,
                        'Conc.': e.conciseness,
                        'Rel.': e.relevance,
                        'Cons.': e.consistency,
                        'Ans.': e.answerability,
                        'AnsC.': e.answer_consistency
                    })

            df = pd.DataFrame(data)
            
            if df.empty:
                logger.error("Step 5: No valid records found after ignoring nulls.")
                return

            logger.info(f"Step 6: Loading completed with {len(df)} valid records.")

            # Calculate Pearson Correlation
            logger.info("Step 7: Calculating Pearson Correlation Matrix...")
            corr_matrix = df.corr(method='pearson')

            # --- Visualization Logic ---
            logger.info("Step 8: Generating High-Resolution Plot...")
            plt.figure(figsize=(10, 8))
            
            # Using specific professional 'Blues' colormap as per your reference image
            # 'annot=True' writes the correlation numbers inside the boxes
            sns.heatmap(
                corr_matrix, 
                annot=True, 
                fmt=".2f", 
                cmap='Blues', 
                square=True, 
                linewidths=1.5,
                cbar_kws={"shrink": .8},
                annot_kws={"size": 12, "weight": "bold"}
            )

            plt.title('Pearson Correlation: Seven Dimensions', fontsize=16, pad=20)
            
            # Auto-save the file
            output_filename = 'pearson_correlation_results.png'
            plt.savefig(output_filename, bbox_inches='tight', dpi=300)
            plt.close()
            
            logger.info(f"Step 9: Image generated successfully.")
            logger.info(f"Step 10: File saved as '{os.path.abspath(output_filename)}'")
            
            print("\n" + "="*30)
            print(f"PROCESS COMPLETE: Image is ready.")
            print("="*30 + "\n")

        except Exception as e:
            logger.error(f"A critical error occurred: {str(e)}", exc_info=True)

if __name__ == "__main__":
    generate_pearson_matrix()