import pandas as pd
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

from data_agent.document_processor import DocumentProcessorFactory
from data_agent.plotter import Visualizer
from data_agent.llm_utils import LLMManager
from langchain.memory import ConversationBufferMemory

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataAnalystAgent:
    """Main class for the Data Analyst Agent"""
    
    def __init__(self):
        """Initialize the Data Analyst Agent"""
        self.df = None
        self.file_path = None
        self.file_type = None
        self.llm_manager = LLMManager()
        self.visualizer = Visualizer()
        self.conversation_memory = ConversationBufferMemory(return_messages=True)
        self.agent = None
        
    def process_document(self, file_path: str) -> pd.DataFrame:
        """Process document and create dataframe"""
        try:
            logger.info(f"Processing document: {file_path}")
            self.file_path = file_path
            self.file_type = Path(file_path).suffix.lower()
            
            # Get appropriate processor using factory
            processor = DocumentProcessorFactory.get_processor(file_path)
            
            # Process document
            df = processor.process(file_path)
            
            if df.empty:
                logger.warning(f"No data extracted from {file_path}")
                return df
                
            self.df = df
            self.visualizer.set_dataframe(df)
            
            # Initialize the pandas agent
            self.agent = self.llm_manager.create_pandas_agent(
                df, 
                memory=self.conversation_memory
            )
            
            logger.info(f"Document processed successfully: {file_path}")
            logger.info(f"DataFrame shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            raise
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process user query and return response with optional visualization"""
        if self.df is None or self.agent is None:
            return {"answer": "No data loaded. Please upload a document first."}
        
        logger.info(f"Processing query: {query}")
        
        # Add query to conversation memory
        self.conversation_memory.chat_memory.add_user_message(query)
        
        try:
            # Process query using the pandas agent
            result = self.agent.run(query)
            self.conversation_memory.chat_memory.add_ai_message(result)
            
            # Check if visualization is needed
            visualization_keywords = [
                "plot", "graph", "chart", "visualize", "visualization", 
                "show", "display", "histogram", "bar", "line", "scatter", "pie"
            ]
            needs_viz = any(keyword in query.lower() for keyword in visualization_keywords)
            
            if needs_viz:
                # Use LLM to determine visualization parameters
                viz_params = self.llm_manager.analyze_for_visualization(query, result)
                
                # Create visualization
                fig = self.visualizer.create_visualization(**viz_params)
                
                return {
                    "answer": result,
                    "visualization": fig.to_json(),
                    "viz_type": viz_params.get("plot_type")
                }
            
            return {"answer": result}
            
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            logger.error(error_msg)
            self.conversation_memory.chat_memory.add_ai_message(error_msg)
            return {"answer": error_msg}
    
    def get_dataframe_info(self) -> Dict[str, Any]:
        """Get information about the loaded dataframe"""
        if self.df is None:
            return {"message": "No data loaded"}
        
        info = {
            "columns": list(self.df.columns),
            "shape": self.df.shape,
            "dtypes": {col: str(dtype) for col, dtype in self.df.dtypes.items()},
            "missing_values": self.df.isna().sum().to_dict(),
            "sample": self.df.head(5).to_dict(),
            "file_type": self.file_type
        }
        
        return info
    
    def list_conversation_history(self) -> List[Dict[str, Any]]:
        """Get conversation history"""
        messages = self.conversation_memory.chat_memory.messages
        history = []
        
        for message in messages:
            history.append({
                "type": message.type,
                "content": message.content
            })
            
        return history 