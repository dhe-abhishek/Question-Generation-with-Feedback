import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key')
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
    RESULTS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL', 'postgresql://postgres:admin1234@localhost:5432/BloomsAI')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    
    # Allowed file extensions
    ALLOWED_EXTENSIONS = {'txt', 'pdf', 'doc', 'docx', 'ppt', 'pptx'}
    
    # Model paths
    MODEL_DIR = os.getenv('MODEL_DIR', 'models')
    AVAILABLE_MODELS = {
        'Gemma 2B': 'google/gemma-2b-it',
        'Gemma 7B': 'google/gemma-7b-it',
        'Gemma 7B (Q4_K_M GGUF)': 'ggml-org/gemma-1.1-7b-it-GGUF/gemma-1.1-7b-it.Q4_K_M.gguf',
        #'DeepSeek Coder': 'deepseek-coder',
        #'Gemma 7B': 'gemma-7b',
        #'Llama 2': 'llama-2-7b'
    }
    
    # The default model for question generation using the google-genai SDK
    GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-3-flash-preview') 
    # The API key is loaded from the .env file (GOOGLE_API_KEY) and used by the SDK automatically
    GEMINI_API_KEY = os.getenv('GOOGLE_API_KEY') 
    MAX_QUESTIONS = 20 # Maximum number of questions allowed
    
    # New: Define Question Types
    QUESTION_TYPES = {
        '1': 'Multiple Choice Question (MCQ)',
        '2': 'Fill-in-the-Blank (FIB)',
        '3': 'Short Answer (SA)',
        '4': 'Long Answer (LA)'
    }
    
    # Question Evaluation Settings
    MAX_EVALUATION_QUESTIONS = 1000
    EVALUATION_ALLOWED_EXTENSIONS = {'txt', 'csv'}

class DevelopmentConfig(Config):
    DEBUG = True
    TESTING = False

class ProductionConfig(Config):
    DEBUG = False
    TESTING = False

class TestingConfig(Config):
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'