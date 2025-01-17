import os
import logging
import open3d as o3d
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
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


def process_pointcloud_file(off_file_path, kdtree_rle_flag, voxel_rle_flag, voxel_sc_flag, slices):
    # Load and Normalize
    points = np.load(off_file_path)
    points_normalized = normalize(points)
    min_bound = np.min(points_normalized, axis=0)
    max_bound = np.max(points_normalized, axis=0)

    point_cloud_normalized = o3d.geometry.PointCloud()
    point_cloud_normalized.points = o3d.utility.Vector3dVector(points_normalized)

    package = [None, None, None]

    # Binary Stuff
    if kdtree_rle_flag:
        kdtree_rle_data, kdtree_rle_points_64, min_bin1_bound, max_bin1_bound = binary_your_pointcloud(point_cloud_normalized, slices, max_bound, min_bound)
        kdtree_rle_data = rle_encode_variable_length(kdtree_rle_data, min_bin1_bound, max_bin1_bound)
        package[0] = kdtree_rle_data

    if voxel_rle_flag:
        voxel_rle_data, voxel_rle_points_64, min_bin2_bound, max_bin2_bound = binary_your_pointcloud_voxels(point_cloud_normalized, slices, max_bound, min_bound)
        voxel_rle_data = rle_encode_variable_length_voxels(voxel_rle_data, min_bin2_bound, max_bin2_bound)
        package[1] = voxel_rle_data


    if voxel_sc_flag:
        voxel_sc_data, voxel_sc_points_64, min_bin3_bound, max_bin3_bound = binary_your_pointcloud_voxels(point_cloud_normalized, slices, max_bound, min_bound)
        voxel_sc_data = sc_encode_variable_length_with_bounds(voxel_sc_data, min_bin3_bound, max_bin3_bound)
        package[2] = voxel_sc_data


    return package


def process_off_file(off_file_path, kdtree_rle_flag, voxel_rle_flag, voxel_sc_flag, slices):
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
        kdtree_rle_data, kdtree_rle_points_64, min_bin1_bound, max_bin1_bound = binary_your_pointcloud(point_cloud_normalized, slices, max_bound, min_bound)
        kdtree_rle_data = rle_encode_variable_length(kdtree_rle_data, min_bin1_bound, max_bin1_bound)
        package[0] = kdtree_rle_data

    if voxel_rle_flag:
        voxel_rle_data, voxel_rle_points_64, min_bin2_bound, max_bin2_bound = binary_your_pointcloud_voxels(point_cloud_normalized, slices, max_bound, min_bound)
        voxel_rle_data = rle_encode_variable_length_voxels(voxel_rle_data, min_bin2_bound, max_bin2_bound)
        package[1] = voxel_rle_data


    if voxel_sc_flag:
        voxel_sc_data, voxel_sc_points_64, min_bin3_bound, max_bin3_bound = binary_your_pointcloud_voxels(point_cloud_normalized, 128, max_bound, min_bound)
        voxel_sc_data = sc_encode_variable_length_with_bounds(voxel_sc_data, min_bin3_bound, max_bin3_bound)
        package[2] = voxel_sc_data


    return package


def save_bin_data(output_dir, label, object_name, data, suffix):
    # Create the output file path
    output_file_path = os.path.join(output_dir, f"{object_name}_{suffix}.bin")
    # print(f"Saving to: {output_file_path}")
    
    # Save the data to the .bin file
    with open(output_file_path, 'wb') as f:
        f.write(data)
        f.close()


def process_file(off_file_path, kdflag, voxel_rle_flag, voxel_sc_flag, pcd_flag, slices):
    if pcd_flag:
        return process_pointcloud_file(off_file_path, kdflag, voxel_rle_flag, voxel_sc_flag, slices)
    else:
        return process_off_file(off_file_path, kdflag, voxel_rle_flag, voxel_sc_flag, slices)


# def iterate_modelnet40(dataset_dir, output_dir, kdflag, voxel_rle_flag, voxel_sc_flag, pcd_flag, slices):
#     # Get a list of all .npy files under train and test directories for each class
#     off_files = []
#     for class_dir in os.listdir(dataset_dir):
#         class_path = os.path.join(dataset_dir, class_dir)
#         if os.path.isdir(class_path):  # Check for the class directory
#             for subset in ['train', 'test']:  # Handle both 'train' and 'test' subdirectories
#                 subset_path = os.path.join(class_path, subset)
#                 if os.path.isdir(subset_path):
#                     for file in os.listdir(subset_path):
#                         if file.endswith(".npy") or (pcd_flag and file.endswith(".off")):
#                             off_files.append(os.path.join(subset_path, file))

#     # Initialize output directories
#     os.makedirs(output_dir, exist_ok=True)
#     if kdflag:
#         kd_out = os.path.join(output_dir, "slice128")
#         os.makedirs(kd_out, exist_ok=True)
#     if voxel_rle_flag:
#         voxel_rle_out = os.path.join(output_dir, "slice_128_voxel_rle")
#         os.makedirs(voxel_rle_out, exist_ok=True)
#     if voxel_sc_flag:
#         voxel_sc_out = os.path.join(output_dir, "slice_128_voxel_sc")
#         os.makedirs(voxel_sc_out, exist_ok=True)

#     # Initialize parallel processing
#     with ProcessPoolExecutor() as executor:
#         futures = []
#         for off_file_path in off_files:
#             # Extract the label from the subdirectory structure
#             label = os.path.basename(os.path.dirname(os.path.dirname(off_file_path)))  # Get the class name
#             subset = os.path.basename(os.path.dirname(off_file_path))  # 'train' or 'test'
#             object_name = os.path.splitext(os.path.basename(off_file_path))[0]
            
#             # Log file path for debugging
#             print(f"Processing: {off_file_path} (Class: {label}, Subset: {subset})")
            
#             futures.append(executor.submit(process_file, off_file_path, kdflag, voxel_rle_flag, voxel_sc_flag, pcd_flag, slices))

#         for future in tqdm(futures, desc="Processing files"):
#             result = future.result()
#             # Extract result and save the .bin data as appropriate
#             for idx, data in enumerate(result):
#                 if data is not None:
#                     suffix = ["kd", "voxel_rle", "voxel_sc"][idx]
#                     # Modify the save path to reflect both train/test and class names
#                     save_bin_data(output_dir, f"{label}_{subset}", object_name, data, suffix)


def iterate_modelnet40(dataset_dir, output_dir, kdflag, voxel_rle_flag, voxel_sc_flag, pcd_flag, slices):
    # Get a list of all .npy files under train and test directories for each class
    off_files = []
    for class_dir in os.listdir(dataset_dir):
        class_path = os.path.join(dataset_dir, class_dir)
        if os.path.isdir(class_path):  # Check for the class directory
            for subset in ['train', 'test']:  # Handle both 'train' and 'test' subdirectories
                subset_path = os.path.join(class_path, subset)
                if os.path.isdir(subset_path):
                    for file in os.listdir(subset_path):
                        if file.endswith(".npy") or (pcd_flag and file.endswith(".off")):
                            off_files.append(os.path.join(subset_path, file))

    # Initialize the main output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize the flagged directories based on the flags
    if kdflag:
        kd_out = os.path.join(output_dir, "slice128")
        os.makedirs(kd_out, exist_ok=True)
    if voxel_rle_flag:
        voxel_rle_out = os.path.join(output_dir, "slice_128_voxel_rle")
        os.makedirs(voxel_rle_out, exist_ok=True)
    if voxel_sc_flag:
        voxel_sc_out = os.path.join(output_dir, "slice_128_voxel_sc")
        os.makedirs(voxel_sc_out, exist_ok=True)

    # Iterate over the files with progress bar
    for off_file_path in tqdm(off_files, desc="Processing files", unit="file"):
        # Extract the label (class) and object name
        label = os.path.basename(os.path.dirname(os.path.dirname(off_file_path)))  # Get class name
        subset = os.path.basename(os.path.dirname(off_file_path))  # 'train' or 'test'
        object_name = os.path.splitext(os.path.basename(off_file_path))[0]

        # Print the file being processed (for debugging)
        # print(f"Processing: {off_file_path} (Class: {label}, Subset: {subset})")

        # Process the file (assumes process_file() returns a list of processed data)
        result = process_file(off_file_path, kdflag, voxel_rle_flag, voxel_sc_flag, pcd_flag, slices)

        # Save the processed data in the appropriate directory
        for idx, data in enumerate(result):
            if data is not None:
                suffix = ["kd", "voxel_rle", "voxel_sc"][idx]

                # Determine the output directory based on the flag and file type
                if suffix == "kd":
                    output_dir_suffix = kd_out
                elif suffix == "voxel_rle":
                    output_dir_suffix = voxel_rle_out
                elif suffix == "voxel_sc":
                    output_dir_suffix = voxel_sc_out

                # Ensure the class and train/test directories exist within the flagged directory
                class_dir = os.path.join(output_dir_suffix, label)
                os.makedirs(class_dir, exist_ok=True)

                # Create the train/test subdirectories if they don't exist
                subset_dir = os.path.join(class_dir, subset)
                os.makedirs(subset_dir, exist_ok=True)

                # Save the data to the correct file path in the train/test subdirectory
                save_bin_data(subset_dir, label, object_name, data, suffix)

if __name__ == "__main__":
    # Variables
    root_dir    = "/home/hi5lab/pointcloud_data/ModelNet40_Pointclouds_1024"             # Root Directory of Data to compress
    output_dir  = "/home/hi5lab/pointcloud_data/uniformsampled_ModelNet40_Dataset"                # Root Output Directory for Compressed Data to go to
    log_file    = "/home/hi5lab/pointcloud_data/storage_analysis_64_pointcloud.log"   # Output Log

    kdtree      = False                                                 # If True, uses Open3D's KDTreeFlann method. Utilizes RLE by default.
    voxel_rle   = True                                                  # If True, uses modified Voxelization technique. Performs custom RLE on the bitarray.
    voxel_sc    = True                                                  # If True, uses modified Voxelization technique. Performs SC Encoding on the bitarray.
    to_console  = False                                                 # If True, prints storage analysis to the console.
    pc_file     = True                                                  # If True, processes point cloud files instead of mesh files.
    slices      = 64                                                    # Number of slices in point cloud grid

    # Performs compression
    iterate_modelnet40(root_dir, output_dir, kdtree, voxel_rle, voxel_sc, pc_file, slices)

    # Performs storage analysis between two datasets. Recommended to only have two dataset directories in the first argument's file path.
    analyze_storage_sizes(output_dir, log_file, to_console)


