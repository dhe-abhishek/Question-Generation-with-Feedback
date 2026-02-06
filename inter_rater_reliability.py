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

def generate_icc():
    logger.info("Initializing ICC Reliability script...")
    app = create_app(Config)
    with app.app_context():
        evals = QuestionEvaluation.query.all()
        # To calculate ICC, we need: [Target (QuestionID), Rater (Model), Rating (Score)]
        # This script treats the 7 parameters as a 'battery' to see if the AI is consistent across them
        data = []
        for e in evals:
            # We unpivot the data into long format for ICC
            metrics = {'Fluency': e.fluency, 'Clarity': e.clarity, 'Consistency': e.consistency}
            for metric, score in metrics.items():
                if score is not None:
                    data.append({'question': e.question_id, 'metric': metric, 'score': score})
        
        df = pd.DataFrame(data)
        logger.info("Calculating Intraclass Correlation Coefficient...")
        
        # Calculate ICC
        icc = pg.intraclass_corr(data=df, targets='question', raters='metric', ratings='score')
        
        # We visualize the ICC results table as a heatmap/dataframe plot
        plt.figure(figsize=(10, 4))
        sns.heatmap(icc.set_index('Type')[['ICC', 'lower 95%', 'upper 95%']], annot=True, cmap='Greens')
        plt.title("Inter-Rater Reliability (ICC) of AI Dimensions")
        
        output = 'irr_icc_results.png'
        plt.savefig(output, bbox_inches='tight', dpi=300)
        logger.info(f"Saved: {output}")

if __name__ == "__main__":
    generate_icc()