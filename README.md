---
noteId: "4ee10160054011f0ade429bcb721d3fd"
tags: []

---

# Data Analyst Agent

A conversational data analyst agent that can process various document types, answer questions, and create visualizations.

## Features

- Process various document types (.csv, .txt, .xlsx, .pdf, .docx, image files)
- Extract tables from documents using OCR
- Answer questions about the data using natural language
- Create visualizations based on queries
- Maintain conversation history
- Beautiful Streamlit UI for interactive data analysis

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/data_agent.git
cd data_agent
```

2. Install the package:
```bash
pip install -e .
```

3. Create a `.env` file with your API keys:
```
OPENAI_API_KEY=your_openai_api_key
```

4. Install system dependencies:
- For PDF processing: `poppler-utils`
- For OCR: `tesseract-ocr`

On Ubuntu/Debian:
```bash
sudo apt-get install poppler-utils tesseract-ocr
```

On macOS:
```bash
brew install poppler tesseract
```

## Usage

### Streamlit Web Interface

The easiest way to use the Data Analyst Agent is through its interactive Streamlit interface:

```bash
data-agent streamlit
```

This will start the Streamlit app on port 8501 by default. You can then access it at http://localhost:8501.

The Streamlit interface provides:
- File upload for various document types
- Data preview and statistics
- Interactive query interface
- Beautiful visualizations
- Conversation history
- Data exploration tools

### Command-line Interface

Process a document:
```bash
data-agent process path/to/document.csv
```

Query the data:
```bash
data-agent query "What are the top 5 categories by sales?"
```

Process a document and query in one command:
```bash
data-agent query -f path/to/document.csv "What are the top 5 categories by sales?"
```

Get information about the loaded data:
```bash
data-agent info
```

View conversation history:
```bash
data-agent history
```

Start the API server:
```bash
data-agent server
```

Start the Streamlit web interface:
```bash
data-agent streamlit
```

### API

The agent also provides a Flask API with the following endpoints:

- `GET /health`: Health check
- `POST /upload`: Upload a document
- `POST /query`: Process a query
- `GET /info`: Get dataframe info
- `GET /history`: Get conversation history

#### Example API Usage

Upload a document:
```bash
curl -X POST -F "file=@path/to/document.csv" http://localhost:5000/upload
```

Query the data:
```bash
curl -X POST -H "Content-Type: application/json" -d '{"query": "What are the top 5 categories by sales?"}' http://localhost:5000/query
```

## Project Structure

- `data_agent/`
  - `processors/`: Document processing modules
  - `visualizers/`: Visualization modules
  - `utils/`: Utility functions
  - `config/`: Configuration settings
  - `tests/`: Test files
  - `agent.py`: Main agent class
  - `app.py`: Flask API
  - `streamlit_app.py`: Streamlit web interface
  - `cli.py`: Command-line interface

## License

[MIT License](LICENSE)