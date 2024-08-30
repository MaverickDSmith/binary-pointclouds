import os
import time, operator, collections, functools
from tqdm import tqdm

import open3d as o3d
import numpy as np

from utils import normalize
from implementation_two import binary_your_pointcloud
from voxel_downsample import density_aware_downsampling



def process_off_file(off_file_path):
    # Placeholder function for your conversion methods
    # Replace with your actual conversion code

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
    slice64_data, points_64 = binary_your_pointcloud(point_cloud_normalized, 64, max_bound, min_bound)
    timings['slice64'] = time.time() - start_time
    start_time = time.time()
    slice128_data, points_128 = binary_your_pointcloud(point_cloud_normalized, 128, max_bound, min_bound)
    timings['slice128'] = time.time() - start_time

    # Voxelize
    # Note about this: we don't want to unfairly judge voxelization versus binary
    # so what we do is apply a density aware downsampling if the voxelization technique has
    # more points than the binary technique, equalizing their total number of points
    # If there are less points, then we call that a win on voxelization's part
    # assuming of course the structure is intact
    start_time = time.time()
    voxel64_data = point_cloud_normalized.voxel_down_sample(0.001)
    if (np.shape(voxel64_data.points)[0] > points_64):
        voxel64_data = density_aware_downsampling(voxel64_data, target_size=points_64, voxel_size=0.01)
    timings['voxel64'] = time.time() - start_time

    start_time = time.time()
    voxel128_data = point_cloud_normalized.voxel_down_sample(0.001)
    if (np.shape(voxel128_data.points)[0] > points_128):
        voxel128_data = density_aware_downsampling(voxel128_data, target_size=points_128, voxel_size=0.01)
    timings['voxel128'] = time.time() - start_time

    return slice64_data, slice128_data, voxel64_data, voxel128_data, timings

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

def save_voxel_data(output_dir, label, object_name, data, suffix):
    # Create the directory for the class (label) if it doesn't exist
    class_dir = os.path.join(output_dir, label)
    os.makedirs(class_dir, exist_ok=True)
    
    # Create the output file path
    output_file_path = os.path.join(class_dir, f"{object_name}_{suffix}.pcd")
    
    o3d.io.write_point_cloud(output_file_path, data, write_ascii=False)

def save_bin_data(output_dir, label, object_name, data, suffix):
    # Create the directory for the class (label) if it doesn't exist
    class_dir = os.path.join(output_dir, label)
    os.makedirs(class_dir, exist_ok=True)
    
    # Create the output file path
    output_file_path = os.path.join(class_dir, f"{object_name}_{suffix}.bin")
    
    # Save the data to the .bin file
    with open(output_file_path, 'wb') as f:
        data.tofile(f)  # Replace with actual binary data write logic

def iterate_modelnet40(dataset_dir, voxel64_output_dir, voxel128_output_dir, slice64_output_dir, slice128_output_dir):
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
            slice64_data, slice128_data, voxelized64_data, voxelized128_data, timings  = process_off_file(off_file_path)

            # Save voxelized data
            suffix = "voxelized64"
            save_voxel_data(voxel64_output_dir, label, object_name, voxelized64_data, suffix)

            suffix = "voxelized128"
            save_voxel_data(voxel128_output_dir, label, object_name, voxelized128_data, suffix)
            
            # Save slice64 data
            suffix = "slice64"
            save_bin_data(slice64_output_dir, label, object_name, slice64_data, suffix)

            # Save slice128 data
            suffix = "slice128"
            save_bin_data(slice128_output_dir, label, object_name, slice128_data, suffix)

            with open(log_file_path, 'a') as log_file:
                log_file.write(f"Processed {object_name} in {label}:\n")
                log_file.write(f" - Load and Normalize: {timings['loading_and_normalize']:.2f} seconds\n")
                log_file.write(f" - Binary Encoding (64 Slices): {timings['slice64']:.2f} seconds\n")
                log_file.write(f" - Voxelization (Equal to 64 Slices): {timings['voxel64']:.2f} seconds\n")
                log_file.write(f" - Binary Encoding (128 Slices): {timings['slice128']:.2f} seconds\n")
                log_file.write(f" - Voxelization (Equal to 128 Slices): {timings['voxel128']:.2f} seconds\n")
                log_file.write("\n")
            total_timings.update(timings)
            pbar.update(1)
            
    # Calculate total processing time and log it
    end_time = time.time()
    total_time = end_time - start_time

    print("Time ended, computing metrics...\n")


    # sum the values with same keys
    result = dict(functools.reduce(operator.add,
            map(collections.Counter, total_timings)))
    
    # get sizes
    slice64_stats = get_directory_stats(slice64_output_dir, 32)
    slice128_stats = get_directory_stats(slice128_output_dir, 256)
    voxel64_stats = get_directory_stats(voxel64_output_dir, 32)
    voxel128_stats = get_directory_stats(voxel128_output_dir, 256)

    with open(final_log_path, 'w') as log_file:
        log_file.write("Time Stats\n")
        log_file.write("=================\n\n")

        log_file.write(f"\n\nSlice64 processing took {result['slice64']:.2f} seconds to complete.\n")
        log_file.write(f"Slice128 processing took {result['slice128']:.2f} seconds to complete.\n")
        log_file.write(f"Voxel 64 took {result['voxel64']:.2f} seconds to complete.\n")
        log_file.write(f"Voxel 128 took {result['voxel128']:.2f} seconds to complete.\n")
        log_file.write(f"Total time of all operations: {total_time:.2f}\n")

        log_file.write("Size Stats\n")
        log_file.write("=================\n\n")

        log_file.write(f"Slice 64 Total Size: {slice64_stats['total_size_kb']}\n")
        log_file.write(f"Slice 64 Average Size: {slice64_stats['average_size_kb']}\n")
        log_file.write(f"Slice 64 File Count: {slice64_stats['file_count']}\n\n")

        log_file.write(f"Slice 128 Total Size: {slice128_stats['total_size_kb']}\n")
        log_file.write(f"Slice 128 Average Size: {slice128_stats['average_size_kb']}\n")
        log_file.write(f"Slice 128 File Count: {slice128_stats['file_count']}\n\n")

        log_file.write(f"Voxel 64 Total Size: {voxel64_stats['total_size_kb']}\n")
        log_file.write(f"Voxel 64 Average Size: {voxel64_stats['average_size_kb']}\n")
        log_file.write(f"Voxel 64 File Count: {voxel64_stats['file_count']}\n")
        log_file.write(f"Voxel 64 files larger than 32KB: {voxel64_stats['larger_than_threshold']}\n")
        log_file.write(f"Voxel 64 files smaller than 32KB: {voxel64_stats['smaller_than_threshold']}\n")
        log_file.write(f"Voxel 64 files equal to 32KB: {voxel64_stats['equal_to_threshold']}\n\n")

        log_file.write(f"Voxel 128 Total Size: {voxel128_stats['total_size_kb']}\n")
        log_file.write(f"Voxel 128 Average Size: {voxel128_stats['average_size_kb']}\n")
        log_file.write(f"Voxel 128 File Count: {voxel128_stats['file_count']}\n")
        log_file.write(f"Voxel 128 files larger than 256KB: {voxel128_stats['larger_than_threshold']}\n")
        log_file.write(f"Voxel 128 files smaller than 256KB: {voxel128_stats['smaller_than_threshold']}\n")
        log_file.write(f"Voxel 128 files equal to 256KB: {voxel128_stats['equal_to_threshold']}\n\n")

    print("All Done!\n")

if __name__ == "__main__":
    # Define your dataset and output directories here
    root_dir = "/home/hi5lab/github/github_ander/Fall 2024/data/ModelNet40"
    slice64_dir = "/home/hi5lab/github/github_ander/Fall 2024/data/storage_test/slice64"
    slice128_dir = "/home/hi5lab/github/github_ander/Fall 2024/data/storage_test/slice128"
    voxel64_dir = "/home/hi5lab/github/github_ander/Fall 2024/data/storage_test/voxel64"
    voxel128_dir = "/home/hi5lab/github/github_ander/Fall 2024/data/storage_test/voxel128"

    iterate_modelnet40(root_dir, voxel64_dir, voxel128_dir, slice64_dir, slice128_dir)