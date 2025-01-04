import pandas as pd
import os

def validate_text(csv_path, txt_files_dir):
    # Load the CSV file
    df = pd.read_csv(csv_path, header=None)  # Adjust if your CSV has a header

    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        # Extract the relevant fields from the row
        file_name, expected_text, start_idx, end_idx, _, _ = row
        
        # Construct the full path to the text file
        txt_file_path = os.path.join(txt_files_dir, file_name)
        
        # Read the content of the file
        with open(txt_file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Extract the text from the specified range
        extracted_text = content[int(start_idx):int(end_idx+1)]
        
        # Print the file name and extracted text
        print(f"File: {file_name}")
        print(f"Extracted Text: '{extracted_text}'")
        
        # Check if the extracted text matches the expected text
        if extracted_text != expected_text:
            raise ValueError(f"Mismatch in {file_name}: Expected '{expected_text}', but got '{extracted_text}'")
        
        print("Match confirmed.\n")

# Example usage
csv_path = '/home/mohamed/repos/nlp_proj/val.csv'  # Replace with your CSV file path
txt_files_dir = '/home/mohamed/repos/nlp_proj/split/EN+PT_txt_files'     # Replace with the directory containing the .txt files
validate_text(csv_path, txt_files_dir)