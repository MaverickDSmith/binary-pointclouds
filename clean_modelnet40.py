import os
from tqdm import tqdm

# def fix_off_files(root_directory):
#     # Collect all .off files with their full paths
#     off_files = []
#     for root, _, files in os.walk(root_directory):
#         for filename in files:
#             if filename.endswith(".off"):
#                 off_files.append(os.path.join(root, filename))

#     # Use tqdm to display a progress bar
#     for file_path in tqdm(off_files, desc="Processing .off files"):
#         with open(file_path, 'r') as file:
#             lines = file.readlines()
        
#         # Check if the file starts with the incorrect "OFF" format
#         if lines[0].startswith('OFF') and len(lines[0].split()) > 1:
#             print(f"Incorrect file format found at {file_path}")
#             # Split the first line into "OFF" and the numbers
#             parts = lines[0].split()
#             off_line = parts[0] + '\n'  # Just "OFF" and a newline
#             numbers_line = ' '.join(parts[1:]) + '\n'  # The rest of the numbers
            
#             # Replace the incorrect lines
#             lines[0] = off_line
#             lines.insert(1, numbers_line)
            
#             # Write the corrected content back to the file
#             with open(file_path, 'w') as file:
#                 file.writelines(lines)

def fix_off_files(root_directory):
    # Collect all .off files with their full paths
    off_files = []
    for root, _, files in os.walk(root_directory):
        for filename in files:
            if filename.endswith(".off"):
                off_files.append(os.path.join(root, filename))

    # Use tqdm to display a progress bar
    for file_path in tqdm(off_files, desc="Processing .off files"):
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        # Check if the first line starts with "OFF" and also contains numbers right after
        if lines[0].startswith('OFF') and len(lines[0].strip()) > 3:
            print(f"Incorrect file format found at {file_path}")

            # Separate "OFF" from the numbers
            off_line = 'OFF\n'
            numbers_line = lines[0][3:].strip() + '\n'
            
            # Update the lines list
            lines[0] = off_line
            lines.insert(1, numbers_line)
            
            # Write the corrected content back to the file
            with open(file_path, 'w') as file:
                file.writelines(lines)

directory = "/home/hi5lab/github/github_ander/Fall 2024/data/ModelNet40"  # Replace this with the path to your directory
fix_off_files(directory)