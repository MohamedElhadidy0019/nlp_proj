def concatenate_files(file1_path, file2_path, output_path):
    # Open the first file and read its content
    with open(file1_path, 'r') as file1:
        content1 = file1.read()

    # Open the second file and read its content
    with open(file2_path, 'r') as file2:
        content2 = file2.read()

    # Open the output file and write the combined content
    with open(output_path, 'w') as output_file:
        output_file.write(content1)
        output_file.write("\n")  # Add a newline between files
        output_file.write(content2)

# Example usage
file1_path = '/home/mohamed/repos/nlp_proj/english.csv'
file2_path = '/home/mohamed/repos/nlp_proj/portugese.csv'
output_path = 'combined_file.txt'

concatenate_files(file1_path, file2_path, output_path)
