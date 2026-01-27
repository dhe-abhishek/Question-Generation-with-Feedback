import re
import string
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
# Assuming the necessary NLTK data (punkt, stopwords, wordnet, omw-1.4) has been downloaded via a dedicated setup function like ensure_nltk_data()
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from flask import current_app

def generate_data_cloud_from_text(text_content):
    """
    Processes text content to clean it, calculate word frequencies, and generate
    data (frequency map and WordCloud object) for the Data Cloud visualization.

    Args:
        text_content (str): The raw text extracted from the uploaded file.

    Returns:
        dict: A dictionary containing 'frequency_data' (list of dicts) and
              'wordcloud_object' (WordCloud instance) or None on failure.
    """
    if not text_content:
        # Use current_app.logger for logging within the Flask context
        current_app.logger.warning("generate_data_cloud_from_text called with empty text.")
        return None
        
    try:
        # --- 1. Text Preprocessing and Cleaning ---
        
        # Remove extra whitespace and newline characters
        cleaned_text = re.sub(r'\s+', ' ', text_content).strip()

        # Optional: Basic header/footer removal (based on line length heuristic)
        # Filters out lines shorter than 50 characters, often headers/footers.
        lines = cleaned_text.split('\n')
        cleaned_lines = []
        for line in lines:
            if len(line) > 50:
                cleaned_lines.append(line)

        cleaned_text = ' '.join(cleaned_lines)
        
        # Convert text to lowercase
        text = cleaned_text.lower()

        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))

        # --- 2. Tokenization and Normalization ---
        
        # Tokenize text
        tokens = nltk.word_tokenize(text)

        # Remove stop words and filter non-alphabetic tokens (e.g., numbers, single characters)
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [word for word in tokens if word not in stop_words and word.isalpha()] 

        # Apply lemmatization
        lemmatizer = WordNetLemmatizer()
        lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]

        # --- 3. Calculate Frequency Distribution ---
        
        # Calculate frequency distribution
        fdist = Counter(lemmatized_tokens)
        
        # Optional: Refine frequency data (e.g., remove single characters or filter by length)
        # Only keep words longer than 1 character or 'a'/'i' (proper articles/pronouns often missed)
        fdist = Counter({
            word: count 
            for word, count in fdist.items() 
            if len(word) > 1 or word in ['a', 'i']
        })
        
        # Sort and take top 100 words for structured data output and visualization efficiency
        top_words = fdist.most_common(100) 

        # Format frequency data for potential table/list display on results page
        frequency_data = [{'word': word, 'count': count} for word, count in top_words]

        # --- 4. Generate Word Cloud Object ---
        
        # Generate word cloud from the top frequency distribution
        wordcloud_object = WordCloud(width=800, 
                                     height=400, 
                                     background_color='white',
                                     max_words=100,
                                     colormap='viridis',
                                     contour_color='steelblue').generate_from_frequencies(dict(top_words)) 

        # Return the generated data
        return {
            'frequency_data': frequency_data,
            'wordcloud_object': wordcloud_object
        }

    except Exception as e:
        current_app.logger.error(f"Error during Data Cloud processing: {str(e)}", exc_info=True)
        return None