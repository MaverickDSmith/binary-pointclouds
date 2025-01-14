import os
import logging
import open3d as o3d
import numpy as np
from tqdm import tqdm

from utils.utils import normalize
from binary_encoder import binary_your_pointcloud, rle_encode_variable_length, binary_your_pointcloud_voxels, rle_encode_variable_length_voxels, sc_encode_variable_length_with_bounds


def setup_logger(log_file, to_console):
    """
    Set up the logger to write to a file and console.
    
    Parameters:
        log_file (str): Path to the log file.
    """
    # Determine if output should go to console or not
    if to_console:
        handlers = [
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]

    else: 
         handlers = [
            logging.FileHandler(log_file)
        ]       

    # Initialize Logger Config
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=handlers
    )


def analyze_storage_sizes(output_dir, log_file="storage_analysis.log", to_console=False):
    """
    Analyze and compare the storage sizes of datasets created by different compression techniques.

    Parameters:
        output_dir (str): The root directory containing the datasets for each compression technique.
        log_file (str): Path to the log file where results will be saved.

    Returns:
        None
    """
    # Set up logger
    setup_logger(log_file, to_console)

    # Datasets directories
    datasets = {}
    for dir_name in os.listdir(output_dir):
        dir_path = os.path.join(output_dir, dir_name)
        if os.path.isdir(dir_path):
            datasets[dir_name] = dir_path

    # Initialize storage size dictionaries
    total_sizes = {key: 0 for key in datasets}
    class_sizes = {key: {} for key in datasets}

    # Calculate sizes
    for key, path in datasets.items():
        for root, dirs, files in os.walk(path):
            # Extract the class name (immediate subdirectory under dataset root)
            if root == path:
                continue  # Skip the root directory

            class_name = os.path.basename(root)

            if class_name not in class_sizes[key]:
                class_sizes[key][class_name] = 0

            # Accumulate file sizes for this class
            for file in files:
                if file.endswith(".bin"):
                    file_path = os.path.join(root, file)
                    file_size = os.path.getsize(file_path)
                    total_sizes[key] += file_size
                    class_sizes[key][class_name] += file_size

    # Log total size for each dataset
    logging.info("Total Dataset Sizes:")
    for key, size in total_sizes.items():
        logging.info(f"  {key}: {size / 1e6:.2f} MB")
    logging.info("\n")

    # Compare sizes class by class
    logging.info("Class-wise Comparisons:")
    comparisons = {
        f"{list(datasets.keys())[0]}_vs_{list(datasets.keys())[1]}": {"larger": 0, "smaller": 0, "equal": 0},
    }
    
    # For comparing the classes between the datasets
    for class_name in class_sizes[list(datasets.keys())[0]]:
        size1 = class_sizes[list(datasets.keys())[0]].get(class_name, 0)
        size2 = class_sizes[list(datasets.keys())[1]].get(class_name, 0)

        equal_files_count = 0
        if size1 == size2:
            comparisons[f"{list(datasets.keys())[0]}_vs_{list(datasets.keys())[1]}"]["equal"] += 1

            # Count equal file sizes per class
            equal_files_count = sum(1 for file in os.listdir(os.path.join(datasets[list(datasets.keys())[0]], class_name))
                                    if file.endswith(".bin") and os.path.getsize(os.path.join(datasets[list(datasets.keys())[0]], class_name, file)) == 
                                    os.path.getsize(os.path.join(datasets[list(datasets.keys())[1]], class_name, file)))

        if size1 > size2:
            comparisons[f"{list(datasets.keys())[0]}_vs_{list(datasets.keys())[1]}"]["larger"] += 1
        elif size1 < size2:
            comparisons[f"{list(datasets.keys())[0]}_vs_{list(datasets.keys())[1]}"]["smaller"] += 1

        # Log individual class comparison
        logging.info(f"  {class_name}:")
        logging.info(f"    {list(datasets.keys())[0]}: {size1 / 1e3:.2f} KB")
        logging.info(f"    {list(datasets.keys())[1]}: {size2 / 1e3:.2f} KB")
        logging.info(f"    Equal Files: {equal_files_count}")
        logging.info("\n")

    # Log summary of size comparisons
    logging.info("Comparison Summary:")
    for key, comp in comparisons.items():
        logging.info(f"  {key}:")
        logging.info(f"    Larger: {comp['larger']}")
        logging.info(f"    Smaller: {comp['smaller']}")
        logging.info(f"    Equal: {comp['equal']}")



def process_off_file(off_file_path, kdtree_rle_flag, voxel_rle_flag, voxel_sc_flag):
    # Load and Normalize
    mesh = o3d.io.read_triangle_mesh(off_file_path)
    points = np.asarray(mesh.vertices)
    points_normalized = normalize(points)
    min_bound = np.min(points_normalized, axis=0)
    max_bound = np.max(points_normalized, axis=0)

    point_cloud_normalized = o3d.geometry.PointCloud()
    point_cloud_normalized.points = o3d.utility.Vector3dVector(points_normalized)

    package = [None, None, None]

    # Binary Stuff
    if kdtree_rle_flag:
        kdtree_rle_data, kdtree_rle_points_64, min_bin1_bound, max_bin1_bound = binary_your_pointcloud(point_cloud_normalized, 128, max_bound, min_bound)
        kdtree_rle_data = rle_encode_variable_length(kdtree_rle_data, min_bin1_bound, max_bin1_bound)
        package[0] = kdtree_rle_data

    if voxel_rle_flag:
        voxel_rle_data, voxel_rle_points_64, min_bin2_bound, max_bin2_bound = binary_your_pointcloud_voxels(point_cloud_normalized, 128, max_bound, min_bound)
        voxel_rle_data = rle_encode_variable_length_voxels(voxel_rle_data, min_bin2_bound, max_bin2_bound)
        package[1] = voxel_rle_data


    if voxel_sc_flag:
        voxel_sc_data, voxel_sc_points_64, min_bin3_bound, max_bin3_bound = binary_your_pointcloud_voxels(point_cloud_normalized, 128, max_bound, min_bound)
        voxel_sc_data = sc_encode_variable_length_with_bounds(voxel_sc_data, min_bin3_bound, max_bin3_bound)
        package[2] = voxel_sc_data


    return package


def save_bin_data(output_dir, label, object_name, data, suffix):
    # Create the directory for the class (label) if it doesn't exist
    class_dir = os.path.join(output_dir, label)
    os.makedirs(class_dir, exist_ok=True)
    
    # Create the output file path
    output_file_path = os.path.join(class_dir, f"{object_name}_{suffix}.bin")
    
    # Save the data to the .bin file
    with open(output_file_path, 'wb') as f:
        f.write(data)  
        f.close()


def iterate_modelnet40(dataset_dir, output_dir, kdflag, voxel_rle_flag, voxel_sc_flag):
    # Get a list of all .off files to initialize the tqdm progress bar
    off_files = []
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(".off"):
                off_files.append(os.path.join(root, file))


    # Initialize output directories
    os.makedirs(output_dir, exist_ok=True)
    if kdflag:
        kd_out = os.path.join(output_dir, "slice128")
        os.makedirs(kd_out, exist_ok=True)
    if voxel_rle_flag:
        voxel_rle_out = os.path.join(output_dir, "slice_128_voxel_rle")
        os.makedirs(voxel_rle_out, exist_ok=True)
    if voxel_sc_flag:
        voxel_sc_out = os.path.join(output_dir, "slice_128_voxel_sc")
        os.makedirs(voxel_sc_out, exist_ok=True)

    # Iterate over .off files with a progress bar
    with tqdm(total=len(off_files), desc="Processing files") as pbar:
        for off_file_path in off_files:
            # Extract label and object_name from the path
            label = os.path.basename(os.path.dirname(os.path.dirname(off_file_path)))
            object_name = os.path.splitext(os.path.basename(off_file_path))[0]
            pbar.set_postfix({'Label': label})

            # Process the .off file to generate voxelized and custom data
            data = process_off_file(off_file_path, kdflag, voxel_rle_flag, voxel_sc_flag)
            
            # Save slice64 data
            if kdflag:
                suffix = "slice128"
                save_bin_data(kd_out, label, object_name, data[0], suffix)

            # Save slice64_rle data
            if voxel_rle_flag:
                suffix = "slice128_rle"
                save_bin_data(voxel_rle_out, label, object_name, data[1], suffix)

            # Save slice64_sc data
            if voxel_sc_flag:
                suffix = "slice128_sc"
                save_bin_data(voxel_sc_out, label, object_name, data[2], suffix)
            pbar.update(1)


if __name__ == "__main__":
    # Variables
    root_dir    = "/home/hi5lab/pointcloud_data/ModelNet40"             # Root Directory of Data to compress
    output_dir  = "/home/hi5lab/pointcloud_data/dataset_128"                # Root Output Directory for Compressed Data to go to
    log_file    = "/home/hi5lab/pointcloud_data/storage_analysis_128.log"   # Output Log

    kdtree      = False                                                 # If True, uses Open3D's KDTreeFlann method. Utilizes RLE by default.
    voxel_rle   = True                                                  # If True, uses modified Voxelization technique. Performs custom RLE on the bitarray.
    voxel_sc    = True                                                  # If True, uses modified Voxelization technique. Performs SC Encoding on the bitarray.
    to_console  = False                                                 # If True, prints storage analysis to the console.

    # Performs compression
    iterate_modelnet40(root_dir, output_dir, kdtree, voxel_rle, voxel_sc)

    # Performs storage analysis between two datasets. Recommended to only have two dataset directories in the first argument's file path.
    analyze_storage_sizes(output_dir, log_file)

    
