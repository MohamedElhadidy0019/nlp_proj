import pandas as pd
from ast import literal_eval
import matplotlib.pyplot as plt
from tqdm import tqdm  # For progress bar

# Define the main classes and subclasses
main_classes = ['Antagonist', 'Protagonist', 'Innocent']
subclasses = {
    'Antagonist': ['Instigator', 'Conspirator', 'Tyrant', 'Foreign Adversary', 
                   'Traitor', 'Spy', 'Saboteur', 'Corrupt', 'Incompetent', 
                   'Terrorist', 'Deceiver', 'Bigot'],
    'Protagonist': ['Guardian', 'Martyr', 'Peacemaker', 'Rebel', 'Underdog', 'Virtuous'],
    'Innocent': ['Forgotten', 'Exploited', 'Victim', 'Scapegoat']
}

# Load the dataset (replace 'data_file' with your file path)
data_file = '/home/mohamed/repos/nlp_proj/split/train.csv'  # Update with your file path
df = pd.read_csv(data_file)

# Initialize a dictionary to store subclass frequencies
subclass_frequencies = {main_class: {subclass: 0 for subclass in subclasses[main_class]} for main_class in main_classes}

# Process each row in the dataset
for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows", unit="row", colour="green"):
    # Extract relevant fields
    text_file = row['article_id']
    entity = row['entity_mention']
    start_pos = row['start_offset']
    end_pos = row['end_offset']
    main_class = row['main_role']
    subclasses_list = literal_eval(row['fine_grained_roles'])  # Convert string to list

    # Update subclass frequencies
    for subclass in subclasses_list:
        if subclass in subclass_frequencies[main_class]:
            subclass_frequencies[main_class][subclass] += 1

# Plot the frequencies for each main class
for main_class in main_classes:
    # Extract subclass names and their frequencies
    subclass_names = list(subclass_frequencies[main_class].keys())
    frequencies = list(subclass_frequencies[main_class].values())
    
    # Create a bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(subclass_names, frequencies, color='skyblue')
    plt.title(f'Subclass Frequencies for {main_class}')
    plt.xlabel('Subclass')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'/home/mohamed/repos/nlp_proj/data_analysis/{main_class}_subclass_frequencies.png')
    plt.close()  # Close the plot to avoid overlapping plots