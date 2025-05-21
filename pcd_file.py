import os
import numpy as np
import open3d as o3d
from bitarray import bitarray
from tqdm import tqdm

# Used to convert .off files to .pcd files in compressed format
# Intended to compare file size of datasets to our technique

def save_pcd(file_path):
    mesh = o3d.io.read_triangle_mesh(file_path)
    points = np.asarray(mesh.vertices)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


def iterate_modelnet40(dataset_dir, output_dir):
    # Get a list of all .npy files under train and test directories for each class
    off_files = []
    for class_dir in os.listdir(dataset_dir):
        class_path = os.path.join(dataset_dir, class_dir)
        if os.path.isdir(class_path):  # Check for the class directory
            for subset in ['train', 'test']:  # Handle both 'train' and 'test' subdirectories
                subset_path = os.path.join(class_path, subset)
                if os.path.isdir(subset_path):
                    for file in os.listdir(subset_path):
                        if file.endswith(".npy") or (file.endswith(".off")):
                            off_files.append(os.path.join(subset_path, file))

    # Initialize the main output directory
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over the files with progress bar
    for off_file_path in tqdm(off_files, desc="Processing files", unit="file"):
        # Extract the label (class) and object name
        label = os.path.basename(os.path.dirname(os.path.dirname(off_file_path)))  # Get class name
        subset = os.path.basename(os.path.dirname(off_file_path))  # 'train' or 'test'
        object_name = os.path.splitext(os.path.basename(off_file_path))[0]

        # Print the file being processed (for debugging)
        print(f"Processing: {off_file_path} (Class: {label}, Subset: {subset})")
   
        # Process the file (assumes process_file() returns a list of processed data)
        result = save_pcd(off_file_path)



        # Ensure the class and train/test directories exist within the flagged directory
        class_dir = os.path.join(output_dir, label)
        os.makedirs(class_dir, exist_ok=True)
        # Create the train/test subdirectories if they don't exist
        subset_dir = os.path.join(class_dir, subset)
        os.makedirs(subset_dir, exist_ok=True)

        out_dir = os.path.join(output_dir, subset_dir)
        

        # Save the data to the correct file path in the train/test subdirectory
        output_file_path = os.path.join(out_dir, f"{object_name}.pcd")
        o3d.io.write_point_cloud(output_file_path, result, write_ascii=False, compressed=True)


# Set input and output directories
input_dir = "/home/hi5lab/pointcloud_data/ModelNet40"
output_dir = "/home/hi5lab/pointcloud_data/ModelNet40_compressed"
os.makedirs(output_dir, exist_ok=True)

# Run script
iterate_modelnet40(input_dir, output_dir)