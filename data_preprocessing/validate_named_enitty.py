import pandas as pd
import os

def validate_text(csv_path, txt_files_dir):
    # Open and read the CSV file line by line
    with open(csv_path, 'r', encoding='utf-8') as csv_file:
        for line in csv_file:
            # Split the line by the delimiter ',' and take the first 4 columns
            try:
                columns = line.strip().split(',')[:4]
                if len(columns) != 4:
                    print(columns)
                    raise ValueError("Invalid number of columns")
                    break
            except:
                print(f"Error: Unable to parse line '{line}'")
                continue
            # Unpack the columns into variables
            file_name, expected_text, start_idx, end_idx = columns
            
            # Construct the full path to the text file
            txt_file_path = os.path.join(txt_files_dir, file_name)
            
            # Read the content of the file
            with open(txt_file_path, 'r', encoding='utf-8') as txt_file:
                content = txt_file.read()
            
            # Extract the text from the specified range
            extracted_text = content[int(start_idx):int(end_idx)+1]
            
            # Print the file name and extracted text
            print(f"File: {file_name}")
            print(f"Extracted Text: '{extracted_text}'")
            
            # Check if the extracted text matches the expected text
            if extracted_text != expected_text:
                print(f"Mismatch in {file_name}: Expected '{expected_text}', but got '{extracted_text}'")
            
            print("Match confirmed.\n")
# Example usage
csv_path = '/home/mohamed/repos/nlp_proj/data_preprocessing/combined_file.csv'  # Replace with your CSV file path
txt_files_dir = '/home/mohamed/repos/nlp_proj/split/EN+PT_txt_files'     # Replace with the directory containing the .txt files
validate_text(csv_path, txt_files_dir)