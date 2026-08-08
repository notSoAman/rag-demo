from pypdf import PdfReader


#we load a pdf from the /knowledge/raw folder and convert it to a .txt file inside a /knowledge/processed folder 
def load_pdf_to_txt(pdf_path: str, txt_path: str):
    # Load the PDF file
    reader = PdfReader(pdf_path)
    
    # Extract text from each page and write to a .txt file
    with open(txt_path, 'w', encoding='utf-8') as txt_file:
        for page in reader.pages:
            text = page.extract_text()
            if text:  # Check if text is not None
                txt_file.write(text)
                txt_file.write('\n')  # Add a newline after each page's text

# for all three folders in /knowledge/raw, we convert to .txt files in /knowledge/processed
def convert_all_pdfs_to_txt(raw_folder: str, processed_folder: str):
    import os

    # Ensure the processed folder exists
    os.makedirs(processed_folder, exist_ok=True)

    # Iterate through all files in the raw folder
    for filename in os.listdir(raw_folder):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(raw_folder, filename)
            txt_filename = os.path.splitext(filename)[0] + '.txt'
            txt_path = os.path.join(processed_folder, txt_filename)
            
            # Convert PDF to TXT
            load_pdf_to_txt(pdf_path, txt_path)
            print(f'Converted {pdf_path} to {txt_path}')
            
convert_all_pdfs_to_txt('chat/knowledge/raw/hindu', 'chat/knowledge/processed/hindu')
convert_all_pdfs_to_txt('chat/knowledge/raw/greek', 'chat/knowledge/processed/greek')
convert_all_pdfs_to_txt('chat/knowledge/raw/norse', 'chat/knowledge/processed/norse')
