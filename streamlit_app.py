import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import json
from pathlib import Path
import time
import base64
from io import BytesIO
import tempfile

from agent import DataAnalystAgent
from settings import SUPPORTED_FILE_FORMATS

# Set page configuration
st.set_page_config(
    page_title="Data Analyst Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize the agent
@st.cache_resource
def get_agent():
    return DataAnalystAgent()

agent = get_agent()

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: #1E88E5;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    .info-box {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #1E88E5;
        margin-bottom: 1rem;
    }
    
    .stButton button {
        background-color: #1E88E5;
        color: white;
        border-radius: 0.5rem;
        border: none;
        padding: 0.5rem 1rem;
        font-size: 1rem;
    }
    
    .query-box {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
    }
    
    .response-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 5px solid #1E88E5;
    }
    
    .upload-section {
        border: 2px dashed #1E88E5;
        border-radius: 0.5rem;
        padding: 2rem;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .stats-card {
        background-color: #f0f8ff;
        border-radius: 0.5rem;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .stats-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1E88E5;
    }
    
    .stats-label {
        font-size: 0.9rem;
        color: #455A64;
    }
    
    .divider {
        height: 1px;
        background-color: #e0e0e0;
        margin: 1.5rem 0;
    }
    
</style>
""", unsafe_allow_html=True)

# Helper functions
def get_supported_file_extensions():
    """Get all supported file extensions"""
    extensions = []
    for category in SUPPORTED_FILE_FORMATS.values():
        extensions.extend(category)
    return extensions

def get_file_extension_category(extension):
    """Get category of file extension"""
    for category, exts in SUPPORTED_FILE_FORMATS.items():
        if extension in exts:
            return category
    return "unknown"

def get_file_icon(extension):
    """Get icon for file extension"""
    category = get_file_extension_category(extension)
    icons = {
        "text": "📄",
        "spreadsheet": "📊",
        "document": "📝",
        "image": "🖼️",
        "unknown": "❓"
    }
    return icons.get(category, "❓")

def display_dataframe_stats(df):
    """Display dataframe statistics in a row of cards"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="stats-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="stats-value">{df.shape[0]:,}</div>', unsafe_allow_html=True)
        st.markdown('<div class="stats-label">Rows</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="stats-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="stats-value">{df.shape[1]}</div>', unsafe_allow_html=True)
        st.markdown('<div class="stats-label">Columns</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="stats-card">', unsafe_allow_html=True)
        missing = df.isna().sum().sum()
        st.markdown(f'<div class="stats-value">{missing:,}</div>', unsafe_allow_html=True)
        st.markdown('<div class="stats-label">Missing Values</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="stats-card">', unsafe_allow_html=True)
        memory_usage = df.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
        st.markdown(f'<div class="stats-value">{memory_usage:.2f} MB</div>', unsafe_allow_html=True)
        st.markdown('<div class="stats-label">Memory Usage</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

def create_download_link(df, filename="data.csv", text="Download CSV"):
    """Create a download link for dataframe"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" class="download-button">{text}</a>'
    return href

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/data-analytics.png", width=80)
    st.markdown("## Data Analyst Agent")
    st.markdown("---")
    
    st.markdown("### 🔍 About")
    st.markdown(
        "This data analyst agent can process various document types, "
        "extract data, answer questions, and create visualizations."
    )
    
    st.markdown("### 📁 Supported Files")
    for category, extensions in SUPPORTED_FILE_FORMATS.items():
        st.markdown(f"**{category.capitalize()}**: {', '.join(extensions)}")
    
    st.markdown("### 🛠️ Tools Used")
    tools_cols = st.columns(2)
    with tools_cols[0]:
        st.markdown("- GPT-4 Turbo")
        st.markdown("- Pandas")
        st.markdown("- Plotly")
    with tools_cols[1]:
        st.markdown("- LangChain")
        st.markdown("- OCR (Tesseract)")
    
    st.markdown("---")
    st.markdown("### 📝 Session Info")
    
    if 'conversation_history' in st.session_state and st.session_state.conversation_history:
        st.success(f"✅ {len(st.session_state.conversation_history)} messages in conversation")
    else:
        st.info("No conversation yet")
    
    if 'data_loaded' in st.session_state and st.session_state.data_loaded:
        st.success("✅ Data loaded")
    else:
        st.warning("No data loaded")
    
    st.markdown("---")
    st.markdown("### ⚙️ Options")
    clear_conversation = st.button("Clear Conversation")
    if clear_conversation:
        if 'conversation_history' in st.session_state:
            st.session_state.conversation_history = []
        st.success("Conversation cleared!")

# Initialize session state
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

# Main content
st.markdown('<h1 class="main-header">📊 Data Analyst Agent</h1>', unsafe_allow_html=True)

# Tabs for different functions
tab1, tab2, tab3 = st.tabs(["📁 Upload & Analyze", "📊 Data Explorer", "💬 Conversation History"])

with tab1:
    # File upload section
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.markdown('<h2 class="sub-header">Upload Your Data</h2>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a file to analyze",
        type=get_supported_file_extensions(),
        help="Upload a file to analyze. Supported formats: CSV, Excel, PDF, Images, etc."
    )
    
    file_placeholder = st.empty()
    
    if uploaded_file is not None:
        # Create a temp file to save the uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        
        with st.spinner("Processing document..."):
            # Attempt to process the document
            try:
                df = agent.process_document(tmp_path)
                st.session_state.data_loaded = True
                
                # If dataframe is empty, show error
                if df.empty:
                    st.error("Could not extract data from the uploaded file. Please try another file.")
                    st.session_state.data_loaded = False
                else:
                    file_placeholder.success(f"File processed successfully: {uploaded_file.name}")
                    
                    # Display dataframe stats
                    st.markdown('<h3 class="sub-header">Data Overview</h3>', unsafe_allow_html=True)
                    display_dataframe_stats(df)
                    
                    # Display dataframe preview
                    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                    st.markdown('<h3 class="sub-header">Data Preview</h3>', unsafe_allow_html=True)
                    
                    st.dataframe(df.head(10), use_container_width=True)
                    
                    # Display download link
                    st.markdown(create_download_link(df, f"processed_{uploaded_file.name}.csv", "💾 Download Processed Data"), unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"Error processing document: {e}")
                st.session_state.data_loaded = False
            
            # Remove the temp file
            os.unlink(tmp_path)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Query section
    st.markdown('<h2 class="sub-header">Ask Questions About Your Data</h2>', unsafe_allow_html=True)
    
    # Disable text input if no data is loaded
    is_disabled = not st.session_state.data_loaded
    if is_disabled:
        st.warning("Please upload a file first before asking questions.")
    
    with st.form(key='query_form'):
        query = st.text_area(
            "Ask a question about your data:",
            placeholder="e.g., 'What are the top 5 categories by sales?' or 'Show me a bar chart of sales by region'",
            disabled=is_disabled,
            height=100
        )
        
        col1, col2 = st.columns([1, 5])
        with col1:
            submit_button = st.form_submit_button(label="Ask", disabled=is_disabled)
        with col2:
            st.markdown(
                '<div style="padding-top: 1rem;">Examples: "Summarize the data", "Show me a chart of X vs Y", "Find correlations"</div>',
                unsafe_allow_html=True
            )
    
    # Process query
    if submit_button and query:
        with st.spinner("Processing query..."):
            try:
                result = agent.process_query(query, allow_dangerous_code=True)
                
                # Add to conversation history
                st.session_state.conversation_history.append({"role": "user", "content": query})
                st.session_state.conversation_history.append({"role": "assistant", "content": result})
                
                # Display answer
                st.markdown('<div class="query-box">', unsafe_allow_html=True)
                st.markdown(f"**Your question:** {query}", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="response-box">', unsafe_allow_html=True)
                st.markdown(f"**Response:** {result['answer']}", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Display visualization if available
                if 'visualization' in result:
                    st.markdown('<h3 class="sub-header">Visualization</h3>', unsafe_allow_html=True)
                    
                    # Convert JSON to figure
                    fig = go.Figure(json.loads(result['visualization']))
                    
                    # Update layout for better appearance
                    fig.update_layout(
                        template='plotly_white',
                        height=500,
                        margin=dict(t=50, l=20, r=20, b=50),
                    )
                    
                    # Display the figure
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Option to download the visualization
                    viz_html = BytesIO()
                    fig.write_html(viz_html, include_plotlyjs="cdn")
                    viz_html = viz_html.getvalue().decode()
                    b64 = base64.b64encode(viz_html.encode()).decode()
                    
                    st.markdown(
                        f'<a href="data:text/html;base64,{b64}" download="visualization.html" class="download-button">💾 Download Visualization</a>',
                        unsafe_allow_html=True
                    )
                
            except Exception as e:
                st.error(f"Error processing query: {e}")

with tab2:
    if not st.session_state.data_loaded:
        st.info("Please upload a file in the 'Upload & Analyze' tab to explore the data.")
    else:
        st.markdown('<h2 class="sub-header">Data Explorer</h2>', unsafe_allow_html=True)
        
        # Get dataframe info
        info = agent.get_dataframe_info()
        
        # Display statistics
        display_dataframe_stats(agent.df)
        
        # Full dataframe view
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown('<h3 class="sub-header">Full Dataset</h3>', unsafe_allow_html=True)
        
        # Filter options
        with st.expander("Filter and Sort Options"):
            # Select columns to display
            selected_columns = st.multiselect(
                "Select columns to display",
                options=info['columns'],
                default=info['columns']
            )
            
            # Number of rows to display
            num_rows = st.slider("Number of rows to display", min_value=5, max_value=100, value=50)
            
            # Sort options
            sort_column = st.selectbox("Sort by column", options=["None"] + info['columns'])
            sort_ascending = st.checkbox("Sort ascending", value=True)
        
        # Filter and sort the dataframe
        filtered_df = agent.df[selected_columns] if selected_columns else agent.df
        
        if sort_column != "None":
            filtered_df = filtered_df.sort_values(by=sort_column, ascending=sort_ascending)
        
        # Display the filtered dataframe
        st.dataframe(filtered_df.head(num_rows), use_container_width=True)
        
        # Download link
        st.markdown(
            create_download_link(filtered_df, "filtered_data.csv", "💾 Download Filtered Data"),
            unsafe_allow_html=True
        )
        
        # Column statistics
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown('<h3 class="sub-header">Column Statistics</h3>', unsafe_allow_html=True)
        
        # Select a column for detailed statistics
        selected_col = st.selectbox("Select a column for detailed statistics", options=info['columns'])
        
        if selected_col:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="stats-card">', unsafe_allow_html=True)
                st.markdown("<b>Basic Statistics</b>", unsafe_allow_html=True)
                
                # Check if column is numeric
                if pd.api.types.is_numeric_dtype(agent.df[selected_col]):
                    # Display numeric statistics
                    stats = agent.df[selected_col].describe()
                    for stat_name, value in stats.items():
                        if stat_name == 'count':
                            continue  # Skip count as we show it elsewhere
                        st.markdown(f"**{stat_name}:** {value:.4f}")
                else:
                    # Display categorical statistics
                    value_counts = agent.df[selected_col].value_counts().head(5)
                    st.markdown("**Top 5 values:**")
                    for value, count in value_counts.items():
                        st.markdown(f"- {value}: {count} occurrences")
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="stats-card">', unsafe_allow_html=True)
                st.markdown("<b>Missing Values & Uniqueness</b>", unsafe_allow_html=True)
                
                # Missing values
                missing = agent.df[selected_col].isna().sum()
                missing_pct = (missing / len(agent.df)) * 100
                st.markdown(f"**Missing values:** {missing} ({missing_pct:.2f}%)")
                
                # Uniqueness
                unique = agent.df[selected_col].nunique()
                unique_pct = (unique / len(agent.df)) * 100
                st.markdown(f"**Unique values:** {unique} ({unique_pct:.2f}%)")
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Add visualization for the selected column
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown("<b>Column Visualization</b>", unsafe_allow_html=True)
            
            if pd.api.types.is_numeric_dtype(agent.df[selected_col]):
                # Histogram for numeric columns
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=agent.df[selected_col].dropna(),
                    histnorm='probability density',
                    name=selected_col,
                    marker=dict(color='#1E88E5')
                ))
                
                fig.update_layout(
                    title=f'Distribution of {selected_col}',
                    xaxis_title=selected_col,
                    yaxis_title='Frequency',
                    template='plotly_white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Bar chart for categorical columns
                value_counts = agent.df[selected_col].value_counts().head(10)
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=value_counts.index,
                    y=value_counts.values,
                    marker=dict(color='#1E88E5')
                ))
                
                fig.update_layout(
                    title=f'Top 10 values in {selected_col}',
                    xaxis_title=selected_col,
                    yaxis_title='Count',
                    template='plotly_white'
                )
                
                st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown('<h2 class="sub-header">Conversation History</h2>', unsafe_allow_html=True)
    
    if not st.session_state.conversation_history:
        st.info("No conversation history yet. Ask questions about your data to start a conversation.")
    else:
        # Display conversation history
        for i, message in enumerate(st.session_state.conversation_history):
            if message["role"] == "user":
                st.markdown(
                    f'<div class="query-box"><b>You:</b> {message["content"]}</div>',
                    unsafe_allow_html=True
                )
            else:
                content = message["content"]
                if isinstance(content, dict) and "answer" in content:
                    content = content["answer"]
                
                st.markdown(
                    f'<div class="response-box"><b>Assistant:</b> {content}</div>',
                    unsafe_allow_html=True
                )
        
        # Option to clear conversation
        if st.button("Clear History"):
            st.session_state.conversation_history = []
            st.experimental_rerun()

# Footer
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown(
    '<p style="text-align: center; color: #666;">Data Analyst Agent | Built with Streamlit, LangChain, and ❤️</p>',
    unsafe_allow_html=True
) 