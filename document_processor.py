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
import tabula
import google.generativeai as genai
from dotenv import load_dotenv

from settings import TEMP_DIR, SUPPORTED_FILE_FORMATS

# Create temp directory if it doesn't exist
os.makedirs(TEMP_DIR, exist_ok=True)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure Gemini API (will use GOOGLE_API_KEY from environment variables if available)
try:
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        genai.configure(api_key=api_key)
        HAS_GEMINI = True
    else:
        HAS_GEMINI = False
        logger.warning("GOOGLE_API_KEY not found in environment variables. Gemini Vision features will be disabled.")
except Exception as e:
    HAS_GEMINI = False
    logger.warning(f"Error configuring Gemini API: {e}. Gemini Vision features will be disabled.")

class BaseDocumentProcessor(ABC):
    """Base abstract class for document processors"""
    
    def validate_file(self, file_path: str) -> bool:
        """Validate if file exists and is not empty"""
        if not os.path.exists(file_path):
            logger.error(f"File does not exist: {file_path}")
            return False
            
        if os.path.getsize(file_path) == 0:
            logger.error(f"File is empty: {file_path}")
            return False
            
        return True
    
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
        
        if not self.validate_file(file_path):
            return pd.DataFrame()
            
        try:
            df = pd.read_csv(file_path)
            return self.clean_dataframe(df)
        except Exception as e:
            logger.error(f"Error processing CSV/TXT file: {e}")
            # Try with different separators
            try:
                logger.info("Trying with different separator (tab)")
                df = pd.read_csv(file_path, sep='\t')
                return self.clean_dataframe(df)
            except Exception as e2:
                logger.error(f"Error processing CSV/TXT with tab separator: {e2}")
                try:
                    logger.info("Trying with different separator (semicolon)")
                    df = pd.read_csv(file_path, sep=';')
                    return self.clean_dataframe(df)
                except Exception as e3:
                    logger.error(f"Error processing CSV/TXT with semicolon separator: {e3}")
                    return pd.DataFrame()


class ExcelProcessor(BaseDocumentProcessor):
    """Processor for Excel files"""
    
    def process(self, file_path: str) -> pd.DataFrame:
        logger.info(f"Processing Excel file: {file_path}")
        
        if not self.validate_file(file_path):
            return pd.DataFrame()
            
        try:
            df = pd.read_excel(file_path)
            return self.clean_dataframe(df)
        except Exception as e:
            logger.error(f"Error processing Excel file: {e}")
            return pd.DataFrame()


class PDFProcessor(BaseDocumentProcessor):
    """Processor for PDF files using tabula-py"""
    
    def process(self, file_path: str) -> pd.DataFrame:
        logger.info(f"Processing PDF file: {file_path}")
        
        if not self.validate_file(file_path):
            return pd.DataFrame()
            
        try:
            # Check for Java installation (required by tabula)
            try:
                import subprocess
                java_version = subprocess.check_output(['java', '-version'], stderr=subprocess.STDOUT)
                logger.info(f"Java found: {java_version}")
            except:
                logger.warning("Java not found. Tabula requires Java to be installed.")
                return self._process_with_ocr(file_path)
                
            # Extract tables using tabula
            tables = tabula.read_pdf(file_path, pages="all", pandas_options={"header": None})
            logger.info(f"Total tables extracted: {len(tables)}")
            
            if not tables:
                logger.warning("No tables detected by tabula-py, falling back to OCR")
                return self._process_with_ocr(file_path)
                
            # Save tables to temp CSV files for debugging
            for i, df in enumerate(tables):
                if not df.empty:
                    temp_csv = os.path.join(TEMP_DIR, f"table_{i}.csv")
                    df.to_csv(temp_csv, index=False)
                    logger.info(f"Table {i} saved as {temp_csv}")
            
            # Filter out empty dataframes
            tables = [df for df in tables if not df.empty]
            
            # Combine all tables
            if len(tables) > 1:
                combined_df = pd.concat(tables, ignore_index=True)
            elif len(tables) == 1:
                combined_df = tables[0]
            else:
                logger.warning("No non-empty tables extracted by tabula")
                return self._process_with_ocr(file_path)
                
            return self.clean_dataframe(combined_df)
                
        except Exception as e:
            logger.error(f"Error processing PDF with tabula: {e}")
            logger.info("Falling back to OCR method")
            return self._process_with_ocr(file_path)
    
    def _process_with_ocr(self, file_path: str) -> pd.DataFrame:
        """Fallback method using OCR for PDFs that tabula can't handle"""
        try:
            # Check for poppler installation
            try:
                import shutil
                poppler_path = shutil.which('pdftoppm')
                if not poppler_path:
                    logger.warning("Poppler not found in PATH. Required for PDF to image conversion.")
                    return pd.DataFrame()
            except Exception as e:
                logger.warning(f"Error checking for poppler: {e}")
                
            # Convert PDF to images
            try:
                images = convert_from_path(file_path)
            except Exception as e:
                logger.error(f"Error converting PDF to images: {e}")
                return pd.DataFrame()
                
            dfs = []
            
            for i, image in enumerate(images):
                logger.info(f"Processing page {i+1}/{len(images)} with OCR")
                
                try:
                    # Save image to temp file to ensure it's properly loaded
                    temp_img_path = os.path.join(TEMP_DIR, f"temp_pdf_page_{i}.png")
                    image.save(temp_img_path)
                    
                    # Load with OpenCV
                    img = cv2.imread(temp_img_path)
                    if img is None:
                        logger.error(f"Failed to load image for page {i+1}")
                        continue
                        
                    # Convert to grayscale
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    
                    # Check if tesseract is installed
                    try:
                        import shutil
                        tesseract_path = shutil.which('tesseract')
                        if not tesseract_path:
                            logger.warning("Tesseract not found in PATH. Required for OCR.")
                            continue
                    except Exception as e:
                        logger.warning(f"Error checking for tesseract: {e}")
                    
                    # Use pytesseract to extract text
                    text = pytesseract.image_to_string(gray)
                    
                    # Try to parse text into a dataframe
                    try:
                        # Create a text file and then read it
                        temp_txt = os.path.join(TEMP_DIR, f"temp_pdf_page_{i}.txt")
                        with open(temp_txt, 'w') as f:
                            f.write(text)
                            
                        # Try different separators
                        try:
                            page_df = pd.read_csv(temp_txt, sep='\t')
                        except:
                            try:
                                page_df = pd.read_csv(temp_txt, sep=',')
                            except:
                                try:
                                    page_df = pd.read_csv(temp_txt, sep=';')
                                except:
                                    logger.warning(f"Could not parse text to dataframe for page {i+1}")
                                    continue
                                    
                        dfs.append(page_df)
                        os.remove(temp_txt)
                    except Exception as e:
                        logger.warning(f"Error processing page {i} with OCR: {e}")
                        continue
                        
                    # Clean up temp image
                    try:
                        os.remove(temp_img_path)
                    except:
                        pass
                        
                except Exception as e:
                    logger.error(f"Error processing page {i+1} image: {e}")
                    continue
            
            # Combine all dataframes
            if dfs:
                combined_df = pd.concat(dfs, ignore_index=True)
                return self.clean_dataframe(combined_df)
            else:
                logger.warning("No tables extracted from PDF with OCR")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error processing PDF with OCR: {e}")
            return pd.DataFrame()


class ImageProcessor(BaseDocumentProcessor):
    """Processor for image files using Gemini Vision API or fallback to OCR"""
    
    def process(self, file_path: str) -> pd.DataFrame:
        logger.info(f"Processing image file: {file_path}")
        
        if not self.validate_file(file_path):
            return pd.DataFrame()
            
        if HAS_GEMINI:
            try:
                return self._process_with_gemini(file_path)
            except Exception as e:
                logger.error(f"Error processing image with Gemini: {e}")
                logger.info("Falling back to OCR method")
                return self._process_with_ocr(file_path)
        else:
            logger.info("Gemini Vision not available, using OCR")
            return self._process_with_ocr(file_path)
    
    def _process_with_gemini(self, file_path: str) -> pd.DataFrame:
        """Process image using Gemini Vision API"""
        try:
            # Verify image can be opened
            try:
                pil_image = Image.open(file_path)
                # Convert PIL image to RGB mode if it's not
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
            except Exception as e:
                logger.error(f"Could not open image file: {e}")
                raise
                
            # Initialize Gemini Vision model
            model = genai.GenerativeModel("gemini-1.5-flash")
            
            # Generate a response from the model
            response = model.generate_content([pil_image, "Extract the table data from this image in structured format. Format as CSV."])
            
            # Get the extracted text
            extracted_text = response.text
            
            if not extracted_text or len(extracted_text.strip()) == 0:
                logger.warning("No text extracted by Gemini")
                return pd.DataFrame()
                
            # Save to temp file and try to read as CSV
            temp_txt = os.path.join(TEMP_DIR, "temp_gemini_extraction.txt")
            with open(temp_txt, 'w') as f:
                f.write(extracted_text)
                
            # Try to parse with different separators
            try:
                df = pd.read_csv(temp_txt)
                os.remove(temp_txt)
                return self.clean_dataframe(df)
            except Exception as csv_err:
                logger.warning(f"Error parsing Gemini output as CSV: {csv_err}")
                try:
                    # Try tab separator
                    df = pd.read_csv(temp_txt, sep='\t')
                    os.remove(temp_txt)
                    return self.clean_dataframe(df)
                except Exception as tab_err:
                    logger.warning(f"Error parsing Gemini output with tab separator: {tab_err}")
                    
                    # Fall back to manual parsing
                    try:
                        # Convert text to structured format
                        lines = extracted_text.strip().split("\n")
                        if not lines:
                            logger.warning("No lines detected in Gemini output")
                            os.remove(temp_txt)
                            return pd.DataFrame()
                            
                        # Try to detect separator
                        first_line = lines[0]
                        if ',' in first_line:
                            sep = ','
                        elif '\t' in first_line:
                            sep = '\t'
                        elif ';' in first_line:
                            sep = ';'
                        else:
                            sep = ' '
                            
                        columns = lines[0].split(sep)  # Extract column headers
                        if not columns:
                            logger.warning("No columns detected in Gemini output")
                            os.remove(temp_txt)
                            return pd.DataFrame()
                            
                        rows = [line.split(sep) for line in lines[1:]]  # Extract rows
                        
                        # Check if the number of columns in rows matches the number of columns
                        adjusted_rows = []
                        for i, row in enumerate(rows):
                            if len(row) != len(columns):
                                logger.info(f"Row {i} has {len(row)} columns, expected {len(columns)} columns. Adjusting...")
                                # Pad or truncate the row to match the number of columns
                                if len(row) < len(columns):
                                    row += [""] * (len(columns) - len(row))  # Pad with empty strings
                                else:
                                    row = row[:len(columns)]  # Truncate
                            adjusted_rows.append(row)
                        
                        # Create a DataFrame
                        df = pd.DataFrame(adjusted_rows, columns=columns)
                        os.remove(temp_txt)
                        
                        # Clean and return the dataframe
                        return self.clean_dataframe(df)
                    except Exception as e:
                        logger.error(f"Error manually parsing Gemini output: {e}")
                        os.remove(temp_txt)
                        raise
            
        except Exception as e:
            logger.error(f"Error processing image with Gemini: {e}")
            raise
    
    def _process_with_ocr(self, file_path: str) -> pd.DataFrame:
        """Fallback method using OCR for images"""
        try:
            # Check if tesseract is installed
            try:
                import shutil
                tesseract_path = shutil.which('tesseract')
                if not tesseract_path:
                    logger.warning("Tesseract not found in PATH. Required for OCR.")
                    return pd.DataFrame()
            except Exception as e:
                logger.warning(f"Error checking for tesseract: {e}")
            
            # Load image safely
            try:
                # First try with PIL to validate
                pil_image = Image.open(file_path)
                # Save to a temp file to ensure format compatibility
                temp_img_path = os.path.join(TEMP_DIR, "temp_ocr_image.png")
                pil_image.save(temp_img_path)
                
                # Now load with OpenCV
                image = cv2.imread(temp_img_path)
                if image is None:
                    logger.error("Failed to load image with OpenCV")
                    return pd.DataFrame()
                    
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            except Exception as e:
                logger.error(f"Error loading image: {e}")
                return pd.DataFrame()
            
            # Use pytesseract to extract text
            try:
                text = pytesseract.image_to_string(gray)
            except Exception as e:
                logger.error(f"Error extracting text with pytesseract: {e}")
                return pd.DataFrame()
            
            # Try to parse text into a dataframe
            try:
                # Create a text file and then read it
                temp_txt = os.path.join(TEMP_DIR, "temp_image.txt")
                with open(temp_txt, 'w') as f:
                    f.write(text)
                    
                # Try with different separators
                try:
                    df = pd.read_csv(temp_txt, sep='\t')
                except:
                    try:
                        df = pd.read_csv(temp_txt, sep=',')
                    except:
                        try:
                            df = pd.read_csv(temp_txt, sep=';')
                        except:
                            logger.warning("Could not parse text to dataframe")
                            os.remove(temp_txt)
                            return pd.DataFrame()
                
                os.remove(temp_txt)
                
                # Clean up temp files
                try:
                    os.remove(temp_img_path)
                except:
                    pass
                    
                return self.clean_dataframe(df)
            except Exception as e:
                logger.warning(f"Error parsing image text to dataframe: {e}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error processing image with OCR: {e}")
            return pd.DataFrame()


class DocxProcessor(BaseDocumentProcessor):
    """Processor for DOCX files"""
    
    def process(self, file_path: str) -> pd.DataFrame:
        logger.info(f"Processing DOCX file: {file_path}")
        
        if not self.validate_file(file_path):
            return pd.DataFrame()
            
        try:
            # Check file magic bytes to confirm it's really a DOCX file
            with open(file_path, 'rb') as f:
                magic = f.read(4)
                if magic != b'PK\x03\x04':  # ZIP file signature (DOCX is a ZIP)
                    logger.error(f"File is not a valid DOCX (ZIP) file: {file_path}")
                    return pd.DataFrame()
            
            text = docx2txt.process(file_path)
            # Create a text file and then read it
            temp_txt = os.path.join(TEMP_DIR, "temp_doc.txt")
            with open(temp_txt, 'w') as f:
                f.write(text)
                
            # Try with different separators
            try:
                df = pd.read_csv(temp_txt, sep='\t')
            except:
                try:
                    df = pd.read_csv(temp_txt, sep=',')
                except:
                    try:
                        df = pd.read_csv(temp_txt, sep=';')
                    except:
                        logger.warning("Could not parse DOCX text to dataframe")
                        os.remove(temp_txt)
                        return pd.DataFrame()
                        
            os.remove(temp_txt)
            return self.clean_dataframe(df)
        except Exception as e:
            logger.error(f"Error processing DOCX file: {e}")
            return pd.DataFrame()


class DocumentProcessorFactory:
    """Factory for creating document processors based on file type"""
    
    @staticmethod
    def get_processor(file_path: str) -> BaseDocumentProcessor:
        if not os.path.exists(file_path):
            raise ValueError(f"File does not exist: {file_path}")
            
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