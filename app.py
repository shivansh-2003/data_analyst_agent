import os
import logging
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import json

from data_agent.agent import DataAnalystAgent
from data_agent.settings import TEMP_DIR, SUPPORTED_FILE_FORMATS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize the agent
agent = DataAnalystAgent()

# Initialize Flask app
app = Flask(__name__)

# Ensure temp directory exists
os.makedirs(TEMP_DIR, exist_ok=True)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "ok", 
        "message": "Data Analyst Agent API is running",
        "version": "0.1.0"
    })

@app.route('/upload', methods=['POST'])
def upload_file():
    """Upload file endpoint"""
    try:
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({"status": "error", "message": "No file in request"}), 400
            
        file = request.files['file']
        
        # Check if file has a name
        if file.filename == '':
            return jsonify({"status": "error", "message": "No file selected"}), 400
            
        # Check if file extension is supported
        filename = secure_filename(file.filename)
        file_extension = os.path.splitext(filename)[1].lower()
        
        supported_extensions = []
        for category in SUPPORTED_FILE_FORMATS.values():
            supported_extensions.extend(category)
            
        if file_extension not in supported_extensions:
            return jsonify({
                "status": "error", 
                "message": f"Unsupported file format: {file_extension}. Supported formats: {supported_extensions}"
            }), 400
            
        # Save file
        file_path = os.path.join(TEMP_DIR, filename)
        file.save(file_path)
        
        # Process document
        df = agent.process_document(file_path)
        
        # Return dataframe info
        info = agent.get_dataframe_info()
        info["status"] = "success"
        info["message"] = f"File {filename} processed successfully"
        
        return jsonify(info)
        
    except Exception as e:
        logger.error(f"Error processing file upload: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/query', methods=['POST'])
def process_query():
    """Process query endpoint"""
    try:
        data = request.json
        
        if not data or 'query' not in data:
            return jsonify({"status": "error", "message": "No query provided"}), 400
            
        query = data.get('query')
        
        # Process query
        result = agent.process_query(query)
        result["status"] = "success"
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/info', methods=['GET'])
def get_info():
    """Get dataframe info endpoint"""
    try:
        info = agent.get_dataframe_info()
        info["status"] = "success"
        
        return jsonify(info)
        
    except Exception as e:
        logger.error(f"Error getting dataframe info: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/history', methods=['GET'])
def get_history():
    """Get conversation history endpoint"""
    try:
        history = agent.list_conversation_history()
        
        return jsonify({
            "status": "success",
            "history": history
        })
        
    except Exception as e:
        logger.error(f"Error getting conversation history: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

# Enable CORS for API
@app.after_request
def add_cors_headers(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return response

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 