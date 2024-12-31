import csv

# Read the input text file
input_file = "/home/mohamed/repos/nlp_proj/PT/subtask-1-annotations.txt"
output_file = "output.csv"

with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
    # Create a CSV writer object with default comma delimiter
    csv_writer = csv.writer(outfile)
    
    for line in infile:
        # Split the line by tabs and write it to the CSV file
        row = line.strip().split('\t')
        csv_writer.writerow(row)

print("Conversion completed!")
