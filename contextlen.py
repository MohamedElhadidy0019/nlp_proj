folder_path = "/home/mohamed/repos/nlp_proj/EN/raw-documents"
import os
from transformers import GPT2Tokenizer

# Initialize the GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Path to the folder containing your text files
# folder_path = "path/to/your/folder"

# Variables to store token counts
token_counts = []
files_with_more_than_1000_tokens = 0
total_files = 0

# Iterate through each file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        total_files += 1
        file_path = os.path.join(folder_path, filename)
        
        # Read the content of the file
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
        # Tokenize the text
        tokens = tokenizer.tokenize(text)
        
        # Count the number of tokens
        token_count = len(tokens)
        token_counts.append(token_count)
        
        # Count files with more than 1000 tokens
        if token_count > 1000:
            files_with_more_than_1000_tokens += 1
        
        # Print the number of tokens for each file
        print(f"{filename}: {token_count} tokens")

# Calculate the minimum, maximum, and average number of tokens
min_tokens = min(token_counts)
max_tokens = max(token_counts)
avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0

# Calculate the percentage of files with more than 1000 tokens
percentage_more_than_1000 = (files_with_more_than_1000_tokens / total_files) * 100 if total_files > 0 else 0

print("\nStatistics:")
print(f"Minimum number of tokens: {min_tokens}")
print(f"Maximum number of tokens: {max_tokens}")
print(f"Average number of tokens: {avg_tokens:.2f}")
print(f"Percentage of files with more than 1000 tokens: {percentage_more_than_1000:.2f}%")
