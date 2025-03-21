import argparse
import sys
import os
import logging
import json
from pathlib import Path
import subprocess

from agent import DataAnalystAgent

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(description="Data Analyst Agent CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Process document command
    process_parser = subparsers.add_parser("process", help="Process a document")
    process_parser.add_argument("file_path", help="Path to the document to process")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Process a query")
    query_parser.add_argument("--file", "-f", help="Path to the document to process (if not already processed)")
    query_parser.add_argument("query", help="Query to process")
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Get dataframe info")
    
    # History command
    history_parser = subparsers.add_parser("history", help="Get conversation history")
    
    # Run server command
    server_parser = subparsers.add_parser("server", help="Run the API server")
    server_parser.add_argument("--host", default="0.0.0.0", help="Host to run the server on")
    server_parser.add_argument("--port", "-p", type=int, default=5000, help="Port to run the server on")
    server_parser.add_argument("--debug", "-d", action="store_true", help="Run in debug mode")
    
    # Run Streamlit app command
    streamlit_parser = subparsers.add_parser("streamlit", help="Run the Streamlit app")
    streamlit_parser.add_argument("--port", "-p", type=int, default=8501, help="Port to run Streamlit on")
    
    args = parser.parse_args()
    
    # Create agent
    agent = DataAnalystAgent()
    
    # Process command
    if args.command == "process":
        file_path = args.file_path
        
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            sys.exit(1)
            
        try:
            df = agent.process_document(file_path)
            info = agent.get_dataframe_info()
            
            print(f"File processed successfully: {file_path}")
            print(f"Dataframe shape: {info['shape']}")
            print(f"Columns: {info['columns']}")
            print("\nSample data:")
            for col, values in info['sample'].items():
                print(f"{col}: {list(values.values())[:5]}")
                
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            sys.exit(1)
            
    # Query command
    elif args.command == "query":
        # Process document if provided
        if args.file:
            file_path = args.file
            
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                sys.exit(1)
                
            try:
                agent.process_document(file_path)
            except Exception as e:
                logger.error(f"Error processing document: {e}")
                sys.exit(1)
                
        # Process query
        result = agent.process_query(args.query)
        
        print(f"\nQuery: {args.query}")
        print(f"Response: {result['answer']}")
        
        if 'visualization' in result:
            viz_path = f"visualization_{Path(agent.file_path).stem}.html"
            
            # Convert JSON to figure and save as HTML
            import plotly.io as pio
            import plotly.graph_objects as go
            
            fig = go.Figure(json.loads(result['visualization']))
            pio.write_html(fig, viz_path)
            
            print(f"\nVisualization saved to: {viz_path}")
            
    # Info command
    elif args.command == "info":
        info = agent.get_dataframe_info()
        
        if "message" in info and info["message"] == "No data loaded":
            print("No data loaded. Please process a document first.")
            sys.exit(1)
            
        print(f"Dataframe shape: {info['shape']}")
        print(f"Columns: {info['columns']}")
        print("\nData types:")
        for col, dtype in info['dtypes'].items():
            print(f"  {col}: {dtype}")
            
        print("\nMissing values:")
        for col, count in info['missing_values'].items():
            print(f"  {col}: {count}")
            
        print("\nSample data:")
        for col, values in info['sample'].items():
            print(f"  {col}: {list(values.values())[:5]}")
            
    # History command
    elif args.command == "history":
        history = agent.list_conversation_history()
        
        if not history:
            print("No conversation history.")
            sys.exit(0)
            
        print("Conversation history:")
        for i, message in enumerate(history):
            print(f"{i+1}. [{message['type']}] {message['content']}")
            
    # Server command
    elif args.command == "server":
        from data_agent.app import app
        
        app.run(
            host=args.host,
            port=args.port,
            debug=args.debug
        )
    
    # Streamlit command
    elif args.command == "streamlit":
        streamlit_path = os.path.join(os.path.dirname(__file__), "streamlit_app.py")
        
        if not os.path.exists(streamlit_path):
            logger.error(f"Streamlit app not found at: {streamlit_path}")
            sys.exit(1)
            
        print(f"Starting Streamlit app on port {args.port}...")
        subprocess.run([
            "streamlit", "run", 
            streamlit_path,
            "--server.port", str(args.port)
        ])
        
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main() 