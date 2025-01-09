import sys
import os

# Ensure the project root is the first entry in the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import time, operator, collections, functools
from tqdm import tqdm

import open3d as o3d
import numpy as np

# from utils.helper import normalize
from binary_encoder import binary_your_pointcloud, rle_encode_variable_length
from binary_encoder_test import binary_your_pointcloud_test, rle_encode_variable_length_test, sc_encode_variable_length_with_bounds

# from utils.voxel_downsample import density_aware_downsampling

def normalize(points):
    # Compute the bounding box
    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)
    center = (min_bound + max_bound) / 2
    scale = max(max_bound - min_bound)
    size = max_bound - min_bound
    pointcloud = (points - center) / scale


    return pointcloud

# Comparing speed of binary compression using different tactics
def process_off_file(off_file_path):
    timings = {}
    start_time = time.time()

    # Load and Normalize
    mesh = o3d.io.read_triangle_mesh(off_file_path)
    points = np.asarray(mesh.vertices)
    points_normalized = normalize(points)
    min_bound = np.min(points_normalized, axis=0)
    max_bound = np.max(points_normalized, axis=0)

    point_cloud_normalized = o3d.geometry.PointCloud()
    point_cloud_normalized.points = o3d.utility.Vector3dVector(points_normalized)

    timings['loading_and_normalize'] = time.time() - start_time


    # Binary Stuff First
    start_time = time.time()
    slice64_data, points_64, min_bin_bound, max_bin_bound = binary_your_pointcloud(point_cloud_normalized, 64, max_bound, min_bound)
    slice64_data = rle_encode_variable_length(slice64_data, min_bin_bound, max_bin_bound)
    timings['slice64'] = time.time() - start_time

    start_time = time.time()
    slice64_data_rle_test, points_64_rle_test, min_bin_bound, max_bin_bound = binary_your_pointcloud_test(point_cloud_normalized, 64, max_bound, min_bound)
    slice64_data_rle_test = rle_encode_variable_length_test(slice64_data_rle_test, min_bin_bound, max_bin_bound)
    timings['slice64_rle_test'] = time.time() - start_time

    start_time = time.time()
    slice64_data_sc_test, points_64_sc_test, min_bin_bound, max_bin_bound = binary_your_pointcloud_test(point_cloud_normalized, 64, max_bound, min_bound)
    slice64_data_sc_test = sc_encode_variable_length_with_bounds(slice64_data_sc_test, min_bin_bound, max_bin_bound)
    timings['slice64_sc_test'] = time.time() - start_time



    return slice64_data, slice64_data_rle_test, slice64_data_sc_test, timings

def get_directory_stats(directory_path, size_threshold_kb):
    total_size_bytes = 0
    file_count = 0
    larger_than_threshold = 0
    smaller_than_threshold = 0
    equal_to_threshold = 0

    # Traverse through all files in the directory and subdirectories
    for root, _, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                # Get the size of each file in bytes
                file_size_bytes = os.path.getsize(file_path)
                total_size_bytes += file_size_bytes
                file_count += 1

                # Convert file size to kilobytes
                file_size_kb = file_size_bytes / 1024

                # Count files based on the size threshold
                if file_size_kb > size_threshold_kb:
                    larger_than_threshold += 1
                elif file_size_kb < size_threshold_kb:
                    smaller_than_threshold += 1
                else:
                    equal_to_threshold += 1
            except FileNotFoundError:
                print(f"File not found: {file_path}")
            except PermissionError:
                print(f"Permission denied: {file_path}")
            except OSError as e:
                print(f"Error processing file {file_path}: {e}")

    # Calculate the average file size in kilobytes
    average_size_kb = (total_size_bytes / 1024) / file_count if file_count > 0 else 0

    return {
        'total_size_kb': total_size_bytes / 1024,
        'average_size_kb': average_size_kb,
        'file_count': file_count,
        'larger_than_threshold': larger_than_threshold,
        'smaller_than_threshold': smaller_than_threshold,
        'equal_to_threshold': equal_to_threshold
    }

# def save_voxel_data(output_dir, label, object_name, data, suffix):
#     # Create the directory for the class (label) if it doesn't exist
#     class_dir = os.path.join(output_dir, label)
#     os.makedirs(class_dir, exist_ok=True)
    
#     # Create the output file path
#     output_file_path = os.path.join(class_dir, f"{object_name}_{suffix}.pcd")
    
#     o3d.io.write_point_cloud(output_file_path, data, write_ascii=False)


def save_bin_data(output_dir, label, object_name, data, suffix):
    # Create the directory for the class (label) if it doesn't exist
    class_dir = os.path.join(output_dir, label)
    os.makedirs(class_dir, exist_ok=True)
    
    # Create the output file path
    output_file_path = os.path.join(class_dir, f"{object_name}_{suffix}.bin")
    
    # # Save the data to the .bin file
    # with open(output_file_path, 'wb') as f:
    #     data.tofile(f) 
    # Save the data to the .bin file
    with open(output_file_path, 'wb') as f:
        f.write(data)  

def iterate_modelnet40(dataset_dir, slice64_output_dir, slice64_rle_output_dir, slice64_sc_output_dir):
    # Initialize timers and log file
    start_time = time.time()
    log_file_path = "processing_log.txt"
    final_log_path = "final_metrics.txt"
    total_timings = {}

    with open(log_file_path, 'w') as log_file:
        log_file.write("Processing Log\n")
        log_file.write("=================\n\n")

    # Get a list of all .off files to initialize the tqdm progress bar
    off_files = []
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(".off"):
                off_files.append(os.path.join(root, file))

    # Iterate over .off files with a progress bar
    with tqdm(total=len(off_files), desc="Processing files") as pbar:
        for off_file_path in off_files:
            # Extract label and object_name from the path
            label = os.path.basename(os.path.dirname(os.path.dirname(off_file_path)))
            object_name = os.path.splitext(os.path.basename(off_file_path))[0]
            pbar.set_postfix({'Label': label})

            # Process the .off file to generate voxelized and custom data
            slice64_data, slice64_rle_data, slice64_sc_data, timings  = process_off_file(off_file_path)
            
            # Save slice64 data
            suffix = "slice64"
            save_bin_data(slice64_output_dir, label, object_name, slice64_data, suffix)

            # Save slice64_rle data
            suffix = "slice64_rle"
            save_bin_data(slice64_rle_output_dir, label, object_name, slice64_rle_data, suffix)

            # Save slice64_sc data
            suffix = "slice64_sc"
            save_bin_data(slice64_sc_output_dir, label, object_name, slice64_sc_data, suffix)

            with open(log_file_path, 'a') as log_file:
                log_file.write(f"Processed {object_name} in {label}:\n")
                log_file.write(f" - Load and Normalize: {timings['loading_and_normalize']:.6f} seconds\n")
                log_file.write(f" - Binary Encoding - Open3D KDTreeFlann (64 Slices): {timings['slice64']:.6f} seconds\n")
                log_file.write(f" - Binary Encoding - Modified Voxelization with RLE (64 Slices): {timings['slice64_rle_test']:.6f} seconds\n")
                log_file.write(f" - Binary Encoding - Modified Voxelization with SC Encoding (64 Slices): {timings['slice64_sc_test']:.6f} seconds\n")
                log_file.write("\n")
            total_timings.update(timings)
            pbar.update(1)
            
    # Calculate total processing time and log it
    end_time = time.time()
    total_time = end_time - start_time
    with open(log_file_path, 'a') as log_file:
        log_file.write(f"Finished Processing. Total Time: {total_time:.2f} seconds\n")

    print("Time ended, computing metrics...\n")


    # sum the values with same keys
    result = dict(functools.reduce(operator.add,
            map(collections.Counter, total_timings)))
    
    # get sizes
    slice64_stats = get_directory_stats(slice64_output_dir, 32)
    slice64_rle_stats = get_directory_stats(slice64_rle_output_dir, 32)
    slice64_sc_stats = get_directory_stats(slice64_sc_output_dir, 32)

    with open(final_log_path, 'w') as log_file:
        # log_file.write("Time Stats\n")
        # log_file.write("=================\n\n")

        # log_file.write(f"\n\nSlice64 processing took {result['slice64']:.2f} seconds to complete.\n")
        # log_file.write(f"Slice128 processing took {result['slice128']:.2f} seconds to complete.\n")
        # log_file.write(f"Total time of all operations: {total_time:.2f}\n")

        log_file.write("Size Stats\n")
        log_file.write("=================\n\n")

        log_file.write(f"Open3D KDTreeFlann (64 Slices) Total Size: {slice64_stats['total_size_kb']}\n")
        log_file.write(f"Open3D KDTreeFlann (64 Slices) Average Size: {slice64_stats['average_size_kb']}\n")
        log_file.write(f"Open3D KDTreeFlann (64 Slices) File Count: {slice64_stats['file_count']}\n\n")

        log_file.write(f"Modified Voxelization with RLE (64 Slices) Test Total Size: {slice64_rle_stats['total_size_kb']}\n")
        log_file.write(f"Modified Voxelization with RLE (64 Slices) Test Average Size: {slice64_rle_stats['average_size_kb']}\n")
        log_file.write(f"Modified Voxelization with RLE (64 Slices) Test File Count: {slice64_rle_stats['file_count']}\n\n")

        log_file.write(f"Modified Voxelization with SC Encoding (64 Slices) Test Total Size: {slice64_sc_stats['total_size_kb']}\n")
        log_file.write(f"Modified Voxelization with SC Encoding (64 Slices) Test Average Size: {slice64_sc_stats['average_size_kb']}\n")
        log_file.write(f"Modified Voxelization with SC Encoding (64 Slices) Test File Count: {slice64_sc_stats['file_count']}\n\n")

        # log_file.write(f"Slice 128 Total Size: {slice128_stats['total_size_kb']}\n")
        # log_file.write(f"Slice 128 Average Size: {slice128_stats['average_size_kb']}\n")
        # log_file.write(f"Slice 128 File Count: {slice128_stats['file_count']}\n\n")


    print("All Done!\n")



if __name__ == "__main__":
    # Define your dataset and output directories here
    root_dir          = "/home/hi5lab/pointcloud_data/ModelNet40"
    slice64_dir       = "/home/hi5lab/pointcloud_data/storage_test_three/slice64"
    slice64_rle_dir   = "/home/hi5lab/pointcloud_data/storage_test_three/slice64_rle"
    slice64_sc_dir    = "/home/hi5lab/pointcloud_data/storage_test_three/slice64_sc"

    iterate_modelnet40(root_dir, slice64_dir, slice64_rle_dir, slice64_sc_dir)