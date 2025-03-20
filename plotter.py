import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Visualizer:
    """Class for creating visualizations using Plotly"""
    
    def __init__(self, df: pd.DataFrame = None):
        self.df = df
    
    def set_dataframe(self, df: pd.DataFrame):
        """Set the dataframe for visualization"""
        self.df = df
    
    def create_visualization(self, plot_type: str, x: str, y: str = None, 
                           title: str = None, color: str = None, size: str = None,
                           **kwargs) -> go.Figure:
        """Create visualization using Plotly"""
        if self.df is None:
            raise ValueError("No dataframe loaded. Please set a dataframe first.")
        
        # Validate column names
        all_cols = set(self.df.columns)
        required_cols = {x}
        if y:
            required_cols.add(y)
        if color:
            required_cols.add(color)
        if size:
            required_cols.add(size)
            
        missing_cols = required_cols - all_cols
        if missing_cols:
            raise ValueError(f"Columns not found in dataframe: {missing_cols}")
        
        fig = None
        
        try:
            if plot_type == 'bar':
                fig = px.bar(self.df, x=x, y=y, title=title, color=color, **kwargs)
                
            elif plot_type == 'line':
                fig = px.line(self.df, x=x, y=y, title=title, color=color, **kwargs)
                
            elif plot_type == 'scatter':
                fig = px.scatter(self.df, x=x, y=y, title=title, color=color, size=size, **kwargs)
                
            elif plot_type == 'pie':
                fig = px.pie(self.df, names=x, values=y, title=title, **kwargs)
                
            elif plot_type == 'histogram':
                fig = px.histogram(self.df, x=x, color=color, title=title, **kwargs)
                
            elif plot_type == 'box':
                fig = px.box(self.df, x=x, y=y, color=color, title=title, **kwargs)
                
            elif plot_type == 'heatmap':
                # For heatmap, we need to pivot the data
                pivot_data = self.df.pivot(index=x, columns=color, values=y)
                fig = px.imshow(pivot_data, title=title, **kwargs)
                
            else:
                raise ValueError(f"Unsupported plot type: {plot_type}")
                
            # Update layout for better appearance
            fig.update_layout(
                template='plotly_white',
                title={
                    'y':0.95,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'
                }
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating {plot_type} visualization: {e}")
            raise
    
    def determine_visualization_params(self, query: str, llm_analysis: str) -> Dict[str, Any]:
        """
        Parse LLM response to determine appropriate visualization parameters
        This would typically call the LLM but is simplified here
        """
        # In a real implementation, this would involve LLM parsing
        # For now, we'll return a placeholder
        try:
            # This would be replaced with actual LLM parsing code
            viz_params = {
                "plot_type": "bar",
                "x": self.df.columns[0],
                "y": self.df.columns[1] if len(self.df.columns) > 1 else None,
                "title": f"Analysis of {query}"
            }
            return viz_params
        except Exception as e:
            logger.error(f"Error determining visualization parameters: {e}")
            return {
                "plot_type": "bar",
                "x": self.df.columns[0],
                "title": "Data Analysis"
            } 