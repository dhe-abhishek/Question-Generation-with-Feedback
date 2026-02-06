import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging
from app import create_app
from app.database import QuestionEvaluation
from config import Config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_kendall():
    logger.info("Initializing Kendall's Tau script...")
    app = create_app(Config)
    with app.app_context():
        evals = QuestionEvaluation.query.all()
        data = [{'Flu.': e.fluency, 'Clar.': e.clarity, 'Conc.': e.conciseness, 'Rel.': e.relevance, 
                 'Cons.': e.consistency, 'Ans.': e.answerability, 'AnsC.': e.answer_consistency} for e in evals]
        
        df = pd.DataFrame(data).dropna(how='all')
        logger.info(f"Calculating Kendall's Tau for {len(df)} records...")
        
        corr_matrix = df.corr(method='kendall')
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='YlGnBu', square=True)
        plt.title("Kendall's Tau Correlation (Robust for Tied Ranks)")
        
        output = 'kendall_correlation.png'
        plt.savefig(output, bbox_inches='tight', dpi=300)
        logger.info(f"Saved: {output}")

if __name__ == "__main__":
    generate_kendall()