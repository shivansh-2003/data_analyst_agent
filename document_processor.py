import os
import pandas as pd
import numpy as np
from pathlib import Path
from abc import ABC, abstractmethod
import pytesseract
from pdf2image import convert_from_path
import docx2txt
import cv2
from PIL import Image
import io
import logging

from settings import TEMP_DIR, SUPPORTED_FILE_FORMATS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BaseDocumentProcessor(ABC):
    """Base abstract class for document processors"""
    
    @abstractmethod
    def process(self, file_path: str) -> pd.DataFrame:
        """Process the document and return a DataFrame"""
        pass
    
    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean dataframe by handling missing values and duplicate columns"""
        if df is None or df.empty:
            return pd.DataFrame()
            
        # Handle missing values
        df = df.fillna(value=np.nan)  # Convert all missing values to NaN
        
        # Identify duplicate columns
        duplicate_cols = []
        for i, col1 in enumerate(df.columns):
            for col2 in df.columns[i+1:]:
                if df[col1].equals(df[col2]):
                    duplicate_cols.append(col2)
        
        # Drop duplicate columns
        if duplicate_cols:
            logger.info(f"Removing {len(duplicate_cols)} duplicate columns")
            df = df.drop(columns=duplicate_cols)
        
        return df


class CSVProcessor(BaseDocumentProcessor):
    """Processor for CSV and TXT files"""
    
    def process(self, file_path: str) -> pd.DataFrame:
        logger.info(f"Processing CSV/TXT file: {file_path}")
        try:
            df = pd.read_csv(file_path)
            return self.clean_dataframe(df)
        except Exception as e:
            logger.error(f"Error processing CSV/TXT file: {e}")
            return pd.DataFrame()


class ExcelProcessor(BaseDocumentProcessor):
    """Processor for Excel files"""
    
    def process(self, file_path: str) -> pd.DataFrame:
        logger.info(f"Processing Excel file: {file_path}")
        try:
            df = pd.read_excel(file_path)
            return self.clean_dataframe(df)
        except Exception as e:
            logger.error(f"Error processing Excel file: {e}")
            return pd.DataFrame()


class PDFProcessor(BaseDocumentProcessor):
    """Processor for PDF files using OCR"""
    
    def process(self, file_path: str) -> pd.DataFrame:
        logger.info(f"Processing PDF file: {file_path}")
        try:
            images = convert_from_path(file_path)
            dfs = []
            
            for i, image in enumerate(images):
                logger.info(f"Processing page {i+1}/{len(images)}")
                # Convert PIL image to OpenCV format
                open_cv_image = np.array(image)
                open_cv_image = open_cv_image[:, :, ::-1].copy()  # RGB to BGR
                
                # Convert to grayscale
                gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
                
                # Use pytesseract to extract text
                text = pytesseract.image_to_string(gray)
                
                # Try to parse text into a dataframe
                try:
                    # Create a text file and then read it
                    temp_txt = os.path.join(TEMP_DIR, f"temp_pdf_page_{i}.txt")
                    with open(temp_txt, 'w') as f:
                        f.write(text)
                    page_df = pd.read_csv(temp_txt, sep='\t')
                    dfs.append(page_df)
                    os.remove(temp_txt)
                except Exception as e:
                    logger.warning(f"Error processing page {i}: {e}")
                    continue
            
            # Combine all dataframes
            if dfs:
                combined_df = pd.concat(dfs, ignore_index=True)
                return self.clean_dataframe(combined_df)
            else:
                logger.warning("No tables extracted from PDF")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error processing PDF file: {e}")
            return pd.DataFrame()


class ImageProcessor(BaseDocumentProcessor):
    """Processor for image files using OCR"""
    
    def process(self, file_path: str) -> pd.DataFrame:
        logger.info(f"Processing image file: {file_path}")
        try:
            # Load image
            image = cv2.imread(file_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Use pytesseract to extract text
            text = pytesseract.image_to_string(gray)
            
            # Try to parse text into a dataframe
            try:
                # Create a text file and then read it
                temp_txt = os.path.join(TEMP_DIR, "temp_image.txt")
                with open(temp_txt, 'w') as f:
                    f.write(text)
                df = pd.read_csv(temp_txt, sep='\t')
                os.remove(temp_txt)
                return self.clean_dataframe(df)
            except Exception as e:
                logger.warning(f"Error parsing image text to dataframe: {e}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error processing image file: {e}")
            return pd.DataFrame()


class DocxProcessor(BaseDocumentProcessor):
    """Processor for DOCX files"""
    
    def process(self, file_path: str) -> pd.DataFrame:
        logger.info(f"Processing DOCX file: {file_path}")
        try:
            text = docx2txt.process(file_path)
            # Create a text file and then read it
            temp_txt = os.path.join(TEMP_DIR, "temp_doc.txt")
            with open(temp_txt, 'w') as f:
                f.write(text)
            df = pd.read_csv(temp_txt, sep='\t')
            os.remove(temp_txt)
            return self.clean_dataframe(df)
        except Exception as e:
            logger.error(f"Error processing DOCX file: {e}")
            return pd.DataFrame()


class DocumentProcessorFactory:
    """Factory for creating document processors based on file type"""
    
    @staticmethod
    def get_processor(file_path: str) -> BaseDocumentProcessor:
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension in SUPPORTED_FILE_FORMATS['text']:
            return CSVProcessor()
        elif file_extension in SUPPORTED_FILE_FORMATS['spreadsheet']:
            return ExcelProcessor()
        elif file_extension in SUPPORTED_FILE_FORMATS['document']:
            if file_extension == '.pdf':
                return PDFProcessor()
            elif file_extension == '.docx' or file_extension == '.doc':
                return DocxProcessor()
        elif file_extension in SUPPORTED_FILE_FORMATS['image']:
            return ImageProcessor()
        else:
            raise ValueError(f"Unsupported file format: {file_extension}") 