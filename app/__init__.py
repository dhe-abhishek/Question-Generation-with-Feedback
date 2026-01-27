from flask import Flask
import os  # Add this import
from app.database import db  # Import the db object


def create_app(config_class):
    """
    Creates and configures the Flask application.
    This factory function is a best practice for larger Flask apps.
    """
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # Initialize SQLAlchemy with the app
    db.init_app(app)
    
    # Ensure secret key is set
    if not app.config['SECRET_KEY']:
        app.config['SECRET_KEY'] = os.urandom(24).hex()
        
    # 🌟 NEW: Check and download NLTK data on application startup
    # This runs once when the app object is created, preventing checks per request.
    #ensure_nltk_data(app) 

    from .routes import main_blueprint
    app.register_blueprint(main_blueprint)

    return app