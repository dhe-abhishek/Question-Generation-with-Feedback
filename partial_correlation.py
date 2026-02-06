import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging
import pingouin as pg
from app import create_app
from app.database import QuestionEvaluation
from config import Config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_partial():
    logger.info("Initializing Partial Correlation script...")
    app = create_app(Config)
    with app.app_context():
        evals = QuestionEvaluation.query.all()
        data = [{'Flu': e.fluency, 'Clar': e.clarity, 'Conc': e.conciseness, 'Rel': e.relevance, 
                 'Cons': e.consistency, 'Ans': e.answerability, 'AnsC': e.answer_consistency} for e in evals]
        
        df = pd.DataFrame(data).dropna() # Partial correlation requires no missing values
        logger.info(f"Calculating Partial Correlation (controlling for all other variables)...")
        
        # Use pingouin to get partial correlation matrix
        pcorr_matrix = df.pcorr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(pcorr_matrix, annot=True, fmt=".2f", cmap='RdBu_r', center=0, square=True)
        plt.title("Partial Correlation (Direct Relationships Only)")
        
        output = 'partial_correlation.png'
        plt.savefig(output, bbox_inches='tight', dpi=300)
        logger.info(f"Saved: {output}")

if __name__ == "__main__":
    generate_partial()