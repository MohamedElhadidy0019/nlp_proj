import random

# Set random seed for reproducibility
random.seed(42)

# Function to process rows with variable-length subclasses
def process_row(row):
    fields = row.strip().split(",")
    # if len(fields) < 6:
    #     return None
    try:
        text_file = fields[0]
        entity = fields[1]
        start_pos = int(fields[2])
        end_pos = int(fields[3])
        main_class = fields[4]
        subclasses = fields[5:]  # Capture all remaining fields as subclasses
    
        return {
            'text_file': text_file,
            'entity': entity,
            'start_pos': start_pos,
            'end_pos': end_pos,
            'main_class': main_class,
            'subclasses': subclasses
        }
    except:
        print(f'Error processing row: {row}')
        return None

# Load the dataset
file_path = '/home/mohamed/repos/nlp_proj/all_data_cleaned.csv'  # Replace with your dataset file path
with open(file_path, 'r') as file:
    data = [process_row(line) for line in file]

# Filter out rows with missing values
data = [row for row in data if row is not None]
print(f"Loaded {len(data)} rows.")

# Get unique text files
unique_files = list(set(row['text_file'] for row in data))

# Shuffle the unique files
random.shuffle(unique_files)

# Define train, val, and test split ratios
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Compute split indices
n = len(unique_files)
train_end = int(train_ratio * n)
val_end = train_end + int(val_ratio * n)

# Split the unique files
train_files = set(unique_files[:train_end])
val_files = set(unique_files[train_end:val_end])
test_files = set(unique_files[val_end:])

# Assign rows to splits
train_data = [row for row in data if row['text_file'] in train_files]
val_data = [row for row in data if row['text_file'] in val_files]
test_data = [row for row in data if row['text_file'] in test_files]

# Function to save data to a CSV file
def save_to_csv(filename, data):
    with open(filename, 'w') as file:
        for row in data:
            subclasses = ",".join(row['subclasses'])
            line = f"{row['text_file']},{row['entity']},{row['start_pos']},{row['end_pos']},{row['main_class']},{subclasses}\n"
            file.write(line)

# Save the splits to separate files
save_to_csv('train.csv', train_data)
save_to_csv('val.csv', val_data)
save_to_csv('test.csv', test_data)

print("Splitting completed! Files saved as train.csv, val.csv, and test.csv.")
