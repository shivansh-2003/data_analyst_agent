import os
import logging
from typing import Dict, Any, List, Optional
from langchain.memory import ConversationBufferMemory
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents import AgentType
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.tools import BaseTool
import pandas as pd
from settings import OPENAI_API_KEY, DEFAULT_MODEL, DEFAULT_TEMPERATURE

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LLMManager:
    """Class for managing LLM interactions"""
    
    def __init__(self, model: str = DEFAULT_MODEL, temperature: float = DEFAULT_TEMPERATURE):
        """Initialize LLM Manager"""
        self.model = model
        self.temperature = temperature
        self._llm = None
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize LLM client"""
        try:
            if not OPENAI_API_KEY:
                logger.warning("OpenAI API key not found in environment variables")
                return
                
            self._llm = ChatOpenAI(
                model=self.model,
                temperature=self.temperature,
                openai_api_key=OPENAI_API_KEY
            )
            logger.info(f"LLM initialized successfully: {self.model}")
        except Exception as e:
            logger.error(f"Error initializing LLM: {e}")
    
    @property
    def llm(self) -> ChatOpenAI:
        """Get LLM instance"""
        if not self._llm:
            self._initialize_llm()
            if not self._llm:
                raise ValueError("LLM could not be initialized")
        return self._llm
    
    def create_pandas_agent(self, df, memory: Optional[ConversationBufferMemory] = None):
        """Create a pandas DataFrame agent"""
        try:
            agent = create_pandas_dataframe_agent(
                self.llm,
                df,
                verbose=True,
                agent_type="tool-calling",
                handle_parsing_errors=True,
                memory=memory,
                allow_dangerous_code=True
            )
            return agent
        except Exception as e:
            logger.error(f"Error creating pandas agent: {e}")
            raise
    
    def analyze_for_visualization(self, query: str, result: str) -> Dict[str, Any]:
        """
        Analyze query and result to determine visualization parameters
        """
        try:
            viz_prompt = f"""
            Based on the query "{query}" and the result "{result}", determine the appropriate visualization parameters:
            1. Plot type (bar, line, scatter, pie, histogram, box, heatmap)
            2. X-axis column
            3. Y-axis column (if applicable)
            4. Title
            5. Any additional parameters like color or size column (if applicable)
            
            Return in JSON format:
            {{
                "plot_type": "...",
                "x": "...",
                "y": "..." (if applicable),
                "title": "...",
                "color": "..." (optional),
                "size": "..." (optional)
            }}
            
            Choose the visualization that best represents the data analysis requested.
            """
            
            response = self.llm.invoke(viz_prompt)
            # Parse the JSON response
            import json
            import re
            
            # Extract JSON content from the response
            content = response.content
            # Find JSON pattern using regex
            json_match = re.search(r'({[\s\S]*})', content)
            
            if json_match:
                json_str = json_match.group(1)
                viz_params = json.loads(json_str)
                return viz_params
            else:
                logger.warning("Could not parse JSON response from LLM")
                return {
                    "plot_type": "bar",
                    "x": "category",
                    "y": "value",
                    "title": "Data Analysis"
                }
                
        except Exception as e:
            logger.error(f"Error analyzing for visualization: {e}")
            return {
                "plot_type": "bar",
                "x": "category",
                "y": "value",
                "title": "Data Analysis"
            } 