import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys and Credentials
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


# LLM Settings
DEFAULT_MODEL = "gpt-4-turbo"
DEFAULT_TEMPERATURE = 0

# File Processing
TEMP_DIR = "temp_files"
SUPPORTED_FILE_FORMATS = {
    'text': ['.csv', '.txt'],
    'spreadsheet': ['.xlsx', '.xls'],
    'document': ['.pdf', '.docx', '.doc'],
    'image': ['.png', '.jpg', '.jpeg']
}

# Create temp directory if it doesn't exist
os.makedirs(TEMP_DIR, exist_ok=True) 