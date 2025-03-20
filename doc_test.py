# doc_test.py
import unittest
import os
import pandas as pd
from document_processor import CSVProcessor, ExcelProcessor, PDFProcessor, ImageProcessor, DocxProcessor

class TestDocumentProcessors(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory for test files
        self.test_dir = 'data_agent/test_files'
        os.makedirs(self.test_dir, exist_ok=True)

        # Sample test data
        self.csv_file = os.path.join(self.test_dir, 'test.csv')
        self.excel_file = os.path.join(self.test_dir, 'test.xlsx')
        self.pdf_file = os.path.join(self.test_dir, 'test.pdf')
        self.image_file = os.path.join(self.test_dir, 'test_image.png')
        self.docx_file = os.path.join(self.test_dir, 'test.docx')

        # Create sample files for testing
        pd.DataFrame({'A': [1, 2], 'B': [3, 4]}).to_csv(self.csv_file, index=False)
        pd.DataFrame({'C': [5, 6], 'D': [7, 8]}).to_excel(self.excel_file, index=False)

        # Create empty files for PDF, image, and DOCX for testing
        open(self.pdf_file, 'a').close()
        open(self.image_file, 'a').close()
        open(self.docx_file, 'a').close()

    def tearDown(self):
        # Remove the test directory and all its contents
        for file in os.listdir(self.test_dir):
            os.remove(os.path.join(self.test_dir, file))
        os.rmdir(self.test_dir)

    def test_csv_processor(self):
        processor = CSVProcessor()
        df = processor.process(self.csv_file)
        self.assertEqual(df.shape[0], 2)  # Check if 2 rows are processed

    def test_excel_processor(self):
        processor = ExcelProcessor()
        df = processor.process(self.excel_file)
        self.assertEqual(df.shape[0], 2)  # Check if 2 rows are processed

    def test_pdf_processor(self):
        processor = PDFProcessor()
        df = processor.process(self.pdf_file)
        self.assertTrue(df.empty)  # Check if no data is processed from an empty PDF

    def test_image_processor(self):
        processor = ImageProcessor()
        df = processor.process(self.image_file)
        self.assertTrue(df.empty)  # Check if no data is processed from an empty image

    def test_docx_processor(self):
        processor = DocxProcessor()
        df = processor.process(self.docx_file)
        self.assertTrue(df.empty)  # Check if no data is processed from an empty DOCX

if __name__ == '__main__':
    unittest.main()