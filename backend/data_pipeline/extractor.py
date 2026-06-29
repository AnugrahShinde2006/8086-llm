import os
import glob
import re

try:
    import fitz  # PyMuPDF
except ImportError:
    print("Please install PyMuPDF: pip install PyMuPDF")
    exit(1)

def clean_text(text):
    # Remove excessive newlines and whitespace
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r' +', ' ', text)
    # Remove weird OCR artifacts or non-ascii if necessary
    # text = text.encode('ascii', 'ignore').decode()
    return text.strip()

def extract_pdfs_to_dataset(pdf_dir, output_file):
    pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {pdf_dir}")
        return

    print(f"Found {len(pdf_files)} PDFs. Starting extraction...")
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for pdf_path in pdf_files:
            print(f"Processing {os.path.basename(pdf_path)}...")
            try:
                doc = fitz.open(pdf_path)
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    text = page.get_text("text")
                    
                    cleaned = clean_text(text)
                    if len(cleaned) > 50: # Skip mostly empty pages
                        # We format it roughly as assistant knowledge text
                        out_f.write(cleaned + "\n<|endoftext|>\n")
                        
                doc.close()
            except Exception as e:
                print(f"Failed to read {pdf_path}: {e}")
                
    print(f"Extraction complete! Dataset saved to {output_file}")

if __name__ == "__main__":
    # Put your downloaded 8086 textbooks in this folder:
    PDF_DIRECTORY = "../data/raw/pdfs"
    OUTPUT_DATASET = "../data/raw/textbook_data.txt"
    
    os.makedirs(PDF_DIRECTORY, exist_ok=True)
    print(f"Please place your 8086 PDF textbooks in: {os.path.abspath(PDF_DIRECTORY)}")
    
    # Uncomment to run extraction once PDFs are in the folder:
    # extract_pdfs_to_dataset(PDF_DIRECTORY, OUTPUT_DATASET)
