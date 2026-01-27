# run.py
import os
from app import create_app
from config import Config
from utils.logger import setup_logging
#from utils.logger import setup_logger


# Setup logging first
log_file = setup_logging()
print(f"Logging to: {log_file}")

# Create necessary directories
if not os.path.exists(Config.UPLOAD_FOLDER):
    os.makedirs(Config.UPLOAD_FOLDER)
if not os.path.exists(Config.RESULTS_FOLDER):
    os.makedirs(Config.RESULTS_FOLDER)
    
app = create_app(Config)

if __name__ == '__main__':
    print("Starting MCQ Generator Application...")
    #app.run(debug=True)
    # host='0.0.0.0' makes the server accessible to other devices on the same Wi-Fi
    app.run(host='0.0.0.0', port=5000, debug=True)