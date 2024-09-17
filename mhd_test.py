from metrics import modified_hausdorff_mean
from implementation_three import rle_decode_variable_length, decode_binary
from utils import normalize

import time, operator, collections, functools, os

import numpy as np
import open3d as o3d
from tqdm import tqdm

from bitarray import bitarray

def process_off_file(off_file_path, slice64_base_path, slice128_base_path, voxel64_base_path, voxel128_base_path):

    # Load and Normalize
    mesh = o3d.io.read_triangle_mesh(off_file_path)
    points = np.asarray(mesh.vertices)
    points_normalized = normalize(points)
    point_cloud_normalized = o3d.geometry.PointCloud()
    point_cloud_normalized.points = o3d.utility.Vector3dVector(points_normalized)

    # Extract the label and file name from the input .off file path
    label = os.path.basename(os.path.dirname(os.path.dirname(off_file_path)))
    object_name = os.path.splitext(os.path.basename(off_file_path))[0]

    # Construct the paths for the slice64 and slice128 files based on the label and object_name
    slice64_path = os.path.join(slice64_base_path, label, f"{object_name}_slice64.bin")
    slice128_path = os.path.join(slice128_base_path, label, f"{object_name}_slice128.bin")
    voxel64_path = os.path.join(voxel64_base_path, label, f"{object_name}_voxel64.pcd")
    voxel128_path = os.path.join(voxel128_base_path, label, f"{object_name}_voxel128.pcd")

    # Check if the slice64 and slice128 files exist
    slice64_data, slice128_data, voxel64_data, voxel128_data = None, None, None, None
    if os.path.exists(slice64_path):
        with open(slice64_path, 'rb') as f:
            ba = bitarray()
            ba.fromfile(f)
            ba, min_bound, max_bound = rle_decode_variable_length(ba)
            numpy_array_loaded = np.array(ba.tolist(), dtype=np.uint8)
            size = max_bound - min_bound
            slice64_data = decode_binary(numpy_array_loaded, 64, size, min_bound)
    if os.path.exists(slice128_path):
        with open(slice128_path, 'rb') as f:
            ba = bitarray()
            ba.fromfile(f)
            ba, min_bound, max_bound = rle_decode_variable_length(ba)
            numpy_array_loaded = np.array(ba.tolist(), dtype=np.uint8)
            size = max_bound - min_bound
            slice128_data = decode_binary(numpy_array_loaded, 128, size, min_bound)
    if os.path.exists(voxel64_path):
        voxel64_data = o3d.io.read_point_cloud(voxel64_path)

    if os.path.exists(voxel128_path):
        voxel128_data = o3d.io.read_point_cloud(voxel128_path)



    # Return the data and timings
    return slice64_data, slice128_data, voxel64_data, voxel128_data, points_normalized

def iterate_modelnet40(dataset_dir, slice64_dir, slice128_dir, voxel64_dir, voxel128_dir):
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
            slice64_data, slice128_data, voxel64_data, voxel128_data, input_pointcloud  = process_off_file(off_file_path, slice64_dir, slice128_dir, voxel64_dir, voxel128_dir)
            mhd_slice64 = modified_hausdorff_mean(input_pointcloud, slice64_data)
            mhd_slice128 = modified_hausdorff_mean(input_pointcloud, slice128_data)
            mhd_voxel64 = modified_hausdorff_mean(input_pointcloud, voxel64_data)
            mhd_voxel128 = modified_hausdorff_mean(input_pointcloud, voxel128_data)

            print(f"Slice64: {mhd_slice64}")
            print(f"Slice128: {mhd_slice128}")
            print(f"Voxel64: {mhd_voxel64}")
            print(f"Voxel128: {mhd_voxel128}")

            pbar.update(1)
            
    # Calculate total processing time and log it
    end_time = time.time()
    total_time = end_time - start_time

    print("Time ended, computing metrics...\n")

    # sum the values with same keys
    result = dict(functools.reduce(operator.add,
            map(collections.Counter, total_timings)))
    

    # with open(final_log_path, 'w') as log_file:
    #     # log_file.write("Time Stats\n")
    #     # log_file.write("=================\n\n")

    #     # log_file.write(f"\n\nSlice64 processing took {result['slice64']:.2f} seconds to complete.\n")
    #     # log_file.write(f"Slice128 processing took {result['slice128']:.2f} seconds to complete.\n")
    #     # log_file.write(f"Total time of all operations: {total_time:.2f}\n")

    #     log_file.write("Size Stats\n")
    #     log_file.write("=================\n\n")

    #     log_file.write(f"Slice 64 Total Size: {slice64_stats['total_size_kb']}\n")
    #     log_file.write(f"Slice 64 Average Size: {slice64_stats['average_size_kb']}\n")
    #     log_file.write(f"Slice 64 File Count: {slice64_stats['file_count']}\n\n")

    #     log_file.write(f"Slice 128 Total Size: {slice128_stats['total_size_kb']}\n")
    #     log_file.write(f"Slice 128 Average Size: {slice128_stats['average_size_kb']}\n")
    #     log_file.write(f"Slice 128 File Count: {slice128_stats['file_count']}\n\n")


    print("All Done!\n")

if __name__ == "__main__":
    # Define your dataset and output directories here
    root_dir = "/home/hi5lab/github/github_ander/Fall 2024/data/ModelNet40"
    slice64_dir = "/home/hi5lab/github/github_ander/Fall 2024/data/storage_test_two/slice64"
    slice128_dir = "/home/hi5lab/github/github_ander/Fall 2024/data/storage_test_two/slice128"
    voxel64_dir = "/home/hi5lab/github/github_ander/Fall 2024/data/storage_test_one/voxel64"
    voxel128_dir = "/home/hi5lab/github/github_ander/Fall 2024/data/storage_test_tone/voxel128"

    iterate_modelnet40(root_dir, slice64_dir, slice128_dir, voxel64_dir, voxel128_dir)