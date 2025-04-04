# from metrics.metrics import modified_hausdorff_mean
# from binary_encoder import rle_decode_variable_length, decode_binary
# from utils.utils import normalize

# import os

# import numpy as np
# import open3d as o3d
# from tqdm import tqdm

# from bitarray import bitarray

# def process_off_file(off_file_path, slice64_base_path, slice128_base_path, voxel64_base_path, voxel128_base_path):

#     # Load and Normalize
#     mesh = o3d.io.read_triangle_mesh(off_file_path)
#     points = np.asarray(mesh.vertices)
#     points_normalized = normalize(points)
#     point_cloud_normalized = o3d.geometry.PointCloud()
#     point_cloud_normalized.points = o3d.utility.Vector3dVector(points_normalized)

#     # Extract the label and file name from the input .off file path
#     label = os.path.basename(os.path.dirname(os.path.dirname(off_file_path)))
#     object_name = os.path.splitext(os.path.basename(off_file_path))[0]

#     # Construct the paths for the slice64 and slice128 files based on the label and object_name
#     slice64_path = os.path.join(slice64_base_path, label, f"{object_name}_slice64.bin")
#     slice128_path = os.path.join(slice128_base_path, label, f"{object_name}_slice128.bin")
#     voxel64_path = os.path.join(voxel64_base_path, label, f"{object_name}_voxel64.pcd")
#     voxel128_path = os.path.join(voxel128_base_path, label, f"{object_name}_voxel128.pcd")

#     # Check if the slice64 and slice128 files exist
#     slice64_data, slice128_data, voxel64_data, voxel128_data = None, None, None, None
#     if os.path.exists(slice64_path):
#         with open(slice64_path, 'rb') as f:
#             ba = bitarray()
#             ba.fromfile(f)
#             ba, min_bound, max_bound = rle_decode_variable_length(ba)
#             numpy_array_loaded = np.array(ba.tolist(), dtype=np.uint8)
#             size = max_bound - min_bound
#             slice64_data = decode_binary(numpy_array_loaded, 64, size, min_bound)
#     if os.path.exists(slice128_path):
#         with open(slice128_path, 'rb') as f:
#             ba = bitarray()
#             ba.fromfile(f)
#             ba, min_bound, max_bound = rle_decode_variable_length(ba)
#             numpy_array_loaded = np.array(ba.tolist(), dtype=np.uint8)
#             size = max_bound - min_bound
#             slice128_data = decode_binary(numpy_array_loaded, 128, size, min_bound)
#     if os.path.exists(voxel64_path):
#         voxel64_data = o3d.io.read_point_cloud(voxel64_path)
#         voxel64_data = np.asarray(voxel64_data.points)

#     if os.path.exists(voxel128_path):
#         voxel128_data = o3d.io.read_point_cloud(voxel128_path)
#         voxel128_data = np.asarray(voxel128_data.points)



#     # Return the data and timings
#     return slice64_data, slice128_data, voxel64_data, voxel128_data, points_normalized

# def iterate_modelnet40(dataset_dir, slice64_dir, slice128_dir, voxel64_dir, voxel128_dir):
#     # Initialize timers and log file
#     log_file_path = "processing_log_mhdtest.txt"
#     final_log_path = "final_metrics_mhdtest.txt"
#     total_mhd_slice64 = {}
#     total_mhd_slice128 = {}
#     total_mhd_voxel64 = {}
#     total_mhd_voxel128 = {}

#     with open(log_file_path, 'w') as log_file:
#         log_file.write("Processing Log\n")
#         log_file.write("=================\n\n")

#     # Get a list of all .off files to initialize the tqdm progress bar
#     off_files = []
#     for root, dirs, files in os.walk(dataset_dir):
#         for file in files:
#             if file.endswith(".off"):
#                 off_files.append(os.path.join(root, file))

#     # Iterate over .off files with a progress bar
#     with tqdm(total=len(off_files), desc="Processing files") as pbar:
#         for off_file_path in off_files:
#             # Extract label and object_name from the path
#             label = os.path.basename(os.path.dirname(os.path.dirname(off_file_path)))
#             object_name = os.path.splitext(os.path.basename(off_file_path))[0]
#             pbar.set_postfix({'Label': label})

#             # Process the .off file to generate voxelized and custom data
#             slice64_data, slice128_data, voxel64_data, voxel128_data, input_pointcloud  = process_off_file(off_file_path, slice64_dir, slice128_dir, voxel64_dir, voxel128_dir)
            
#             # Compute MHD values
#             mhd_slice64 = modified_hausdorff_mean(input_pointcloud, slice64_data)
#             mhd_slice128 = modified_hausdorff_mean(input_pointcloud, slice128_data)
#             mhd_voxel64 = modified_hausdorff_mean(input_pointcloud, voxel64_data)
#             mhd_voxel128 = modified_hausdorff_mean(input_pointcloud, voxel128_data)

#             # Append MHD values to corresponding dictionaries
#             total_mhd_slice64.setdefault(label, []).append(mhd_slice64)
#             total_mhd_slice128.setdefault(label, []).append(mhd_slice128)
#             total_mhd_voxel64.setdefault(label, []).append(mhd_voxel64)
#             total_mhd_voxel128.setdefault(label, []).append(mhd_voxel128)

#             # Log MHD values for this object
#             with open(log_file_path, 'a') as log_file:
#                 log_file.write(f"{label}: {object_name}\n")
#                 log_file.write(f"Slice64: {mhd_slice64}\n")
#                 log_file.write(f"Slice128: {mhd_slice128}\n")
#                 log_file.write(f"Voxel64: {mhd_voxel64}\n")
#                 log_file.write(f"Voxel128: {mhd_voxel128}\n\n")

#             pbar.update(1)


#     # Calculate and log final average MHD per category and overall
#     with open(final_log_path, 'w') as final_log:
#         final_log.write("Final MHD Averages\n")
#         final_log.write("=================\n\n")
        
#         def calculate_and_log_average(total_mhd, name):
#             final_log.write(f"{name} Averages:\n")
#             category_averages = []
#             for label, values in total_mhd.items():
#                 category_average = sum(values) / len(values)
#                 category_averages.append((label, category_average))
#                 final_log.write(f"{label}: {category_average}\n")
            
#             overall_average = sum([avg for _, avg in category_averages]) / len(category_averages)
#             final_log.write(f"Overall {name} Average: {overall_average}\n\n")

#         # Calculate and log averages for each MHD dictionary
#         calculate_and_log_average(total_mhd_slice64, "Slice64")
#         calculate_and_log_average(total_mhd_slice128, "Slice128")
#         calculate_and_log_average(total_mhd_voxel64, "Voxel64")
#         calculate_and_log_average(total_mhd_voxel128, "Voxel128")

#     # # sum the values with same keys
#     # result = dict(functools.reduce(operator.add,
#     #         map(collections.Counter, total_timings)))
    

#     # with open(final_log_path, 'w') as log_file:
#     #     # log_file.write("Time Stats\n")
#     #     # log_file.write("=================\n\n")

#     #     # log_file.write(f"\n\nSlice64 processing took {result['slice64']:.2f} seconds to complete.\n")
#     #     # log_file.write(f"Slice128 processing took {result['slice128']:.2f} seconds to complete.\n")
#     #     # log_file.write(f"Total time of all operations: {total_time:.2f}\n")

#     #     log_file.write("Size Stats\n")
#     #     log_file.write("=================\n\n")

#     #     log_file.write(f"Slice 64 Total Size: {slice64_stats['total_size_kb']}\n")
#     #     log_file.write(f"Slice 64 Average Size: {slice64_stats['average_size_kb']}\n")
#     #     log_file.write(f"Slice 64 File Count: {slice64_stats['file_count']}\n\n")

#     #     log_file.write(f"Slice 128 Total Size: {slice128_stats['total_size_kb']}\n")
#     #     log_file.write(f"Slice 128 Average Size: {slice128_stats['average_size_kb']}\n")
#     #     log_file.write(f"Slice 128 File Count: {slice128_stats['file_count']}\n\n")


#     print("All Done!\n")

# if __name__ == "__main__":
#     # Define your dataset and output directories here
#     root_dir = "/home/hi5lab/pointcloud_data/ModelNet40"
#     slice64_dir = "/home/hi5lab/pointcloud_data/storage_test_two/slice64"
#     slice128_dir = "/home/hi5lab/pointcloud_data/storage_test_two/slice128"
#     voxel64_dir = "/home/hi5lab/pointcloud_data/storage_test_two/voxel64"
#     voxel128_dir = "/home/hi5lab/pointcloud_data/storage_test_two/voxel128"

#     iterate_modelnet40(root_dir, slice64_dir, slice128_dir, voxel64_dir, voxel128_dir)



from metrics.metrics import modified_hausdorff_mean
from binary_encoder import rle_decode_variable_length, decode_binary, sc_decode_variable_length_with_bounds, rle_decode_variable_length_voxels
from utils.utils import normalize

import os

import numpy as np
import open3d as o3d
from tqdm import tqdm

from bitarray import bitarray

def process_off_file(off_file_path, slice64_rle_dir, slice64_sc_dir, slice128_rle_dir, slice128_sc_dir):

    # Load and Normalize
    mesh = o3d.io.read_triangle_mesh(off_file_path)
    points = np.asarray(mesh.vertices)
    points_normalized = normalize(points)
    point_cloud_normalized = o3d.geometry.PointCloud()
    point_cloud_normalized.points = o3d.utility.Vector3dVector(points_normalized)

    # Extract the label and file name from the input .off file path
    label = os.path.basename(os.path.dirname(os.path.dirname(off_file_path)))
    object_name = os.path.splitext(os.path.basename(off_file_path))[0]

    # slice_file = f"{file_base}_slice64.bin"
    # slice_path = os.path.join(slice64_path, class_name, 'train', slice_file)
    # if not os.path.exists(slice_path):
    #     slice_path = os.path.join(slice64_path, class_name, 'test', slice_file)
    #     if not os.path.exists(slice_path):
    # Construct the paths for the slice64 and slice128 files based on the label and object_name
    slice64_rle_path = os.path.join(slice64_rle_dir, label, "train", f"{object_name}_voxel_rle.bin")
    if not os.path.exists(slice64_rle_path):
        slice64_rle_path = os.path.join(slice64_rle_dir, label, "test", f"{object_name}_voxel_rle.bin")

    slice64_sc_path = os.path.join(slice64_sc_dir, label, "train", f"{object_name}_voxel_sc.bin")
    if not os.path.exists(slice64_sc_path):
        slice64_sc_path = os.path.join(slice64_sc_dir, label, "test", f"{object_name}_voxel_sc.bin")

    slice128_rle_path = os.path.join(slice128_rle_dir, label, "train", f"{object_name}_voxel_rle.bin")
    if not os.path.exists(slice128_rle_path):
        slice128_rle_path = os.path.join(slice128_rle_dir, label, "test", f"{object_name}_voxel_rle.bin")

    slice128_sc_path = os.path.join(slice128_sc_dir, label, "train", f"{object_name}_voxel_sc.bin")
    if not os.path.exists(slice128_sc_path):
        slice128_sc_path = os.path.join(slice128_sc_dir, label, "test", f"{object_name}_voxel_sc.bin")

# /home/hi5lab/pointcloud_data/uniformsampled_ModelNet40_Dataset_2048/slice_64_voxel_rle/glass_box/train/glass_box_0080_voxel_rle.bin
# /home/hi5lab/pointcloud_data/uniformsampled_ModelNet40_Dataset_2048/slice_64_voxel_rle/glass_box/train/glass_box_0080_voxel_rle.bin

    # Check if the slice64 and slice128 files exist
    slice64_rle_data, slice64_sc_data, slice128_rle_data, slice128_sc_data = None, None, None, None
    if os.path.exists(slice64_rle_path):
        ba = bitarray()
        with open(slice64_rle_path, 'rb') as f:
            ba.fromfile(f)
            ba, min_bound, max_bound = rle_decode_variable_length_voxels(ba)
            numpy_array_loaded = np.array(ba.tolist(), dtype=np.uint8)
            size = max_bound - min_bound
            slice64_rle_data = decode_binary(numpy_array_loaded, 64, size, min_bound)
    if os.path.exists(slice64_sc_path):
        ba = bitarray()
        with open(slice64_sc_path, 'rb') as f:
            ba = f.read()
            ba, min_bound, max_bound = sc_decode_variable_length_with_bounds(ba)
            numpy_array_loaded = np.array(ba.tolist(), dtype=np.uint8)
            size = max_bound - min_bound
            slice64_sc_data = decode_binary(numpy_array_loaded, 64, size, min_bound)
    if os.path.exists(slice128_rle_path):
        ba = bitarray()
        with open(slice128_rle_path, 'rb') as f:
            ba.fromfile(f)
            ba, min_bound, max_bound = rle_decode_variable_length_voxels(ba)
            numpy_array_loaded = np.array(ba.tolist(), dtype=np.uint8)
            size = max_bound - min_bound
            slice128_rle_data = decode_binary(numpy_array_loaded, 128, size, min_bound)
    if os.path.exists(slice128_sc_path):
        ba = bitarray()
        with open(slice128_sc_path, 'rb') as f:
            ba = f.read()
            ba, min_bound, max_bound = sc_decode_variable_length_with_bounds(ba)
            numpy_array_loaded = np.array(ba.tolist(), dtype=np.uint8)
            size = max_bound - min_bound
            slice128_sc_data = decode_binary(numpy_array_loaded, 128, size, min_bound)


    # Return the data and timings
    return slice64_rle_data, slice64_sc_data, slice128_rle_data, slice128_sc_data, points_normalized

def iterate_modelnet40(dataset_dir, slice64_rle_dir, slice64_sc_dir, slice128_rle_dir, slice128_sc_dir):
    # Initialize timers and log file
    log_file_path = "processing_log_binaryvoxel_mhd.txt"
    final_log_path = "final_metrics_binaryvoxel_mhd.txt"
    total_cham_slice64_rle = {}
    total_cham_slice64_sc = {}
    total_cham_slice128_rle = {}
    total_cham_slice128_sc = {}

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
            slice64_rle_data, slice64_sc_data, slice128_rle_data, slice128_sc_data, input_pointcloud  = process_off_file(off_file_path, slice64_rle_dir, slice64_sc_dir, slice128_rle_dir, slice128_sc_dir)
            
            # print(np.shape(slice64_rle_data))
            # Compute chamfer values
            cham_slice64_rle = modified_hausdorff_mean(input_pointcloud, slice64_rle_data)
            cham_slice64_sc = modified_hausdorff_mean(input_pointcloud, slice64_sc_data)
            cham_slice128_rle = modified_hausdorff_mean(input_pointcloud, slice128_rle_data)
            cham_slice128_sc = modified_hausdorff_mean(input_pointcloud, slice128_sc_data)

            # Append MHD values to corresponding dictionaries
            total_cham_slice64_rle.setdefault(label, []).append(cham_slice64_rle)
            total_cham_slice64_sc.setdefault(label, []).append(cham_slice64_sc)
            total_cham_slice128_rle.setdefault(label, []).append(cham_slice128_rle)
            total_cham_slice128_sc.setdefault(label, []).append(cham_slice128_sc)

            # Log MHD values for this object
            with open(log_file_path, 'a') as log_file:
                log_file.write(f"{label}: {object_name}\n")
                log_file.write(f"Slice64 RLE: {cham_slice64_rle}\n")
                log_file.write(f"Slice64 SC: {cham_slice64_sc}\n")
                log_file.write(f"Slice128 RLE: {cham_slice128_rle}\n")
                log_file.write(f"Slice128 SC: {cham_slice128_sc}\n\n")

            pbar.update(1)


    # Calculate and log final average MHD per category and overall
    with open(final_log_path, 'w') as final_log:
        final_log.write("Final MHD Averages\n")
        final_log.write("=================\n\n")
        
        def calculate_and_log_average(total_mhd, name):
            final_log.write(f"{name} Averages:\n")
            category_averages = []
            for label, values in total_mhd.items():
                category_average = sum(values) / len(values)
                category_averages.append((label, category_average))
                final_log.write(f"{label}: {category_average}\n")
            
            overall_average = sum([avg for _, avg in category_averages]) / len(category_averages)
            final_log.write(f"Overall {name} Average: {overall_average}\n\n")

        # Calculate and log averages for each MHD dictionary
        calculate_and_log_average(total_cham_slice64_rle, "Slice64 RLE")
        calculate_and_log_average(total_cham_slice64_sc, "Slice64 SC")
        calculate_and_log_average(total_cham_slice128_rle, "Slice128 RLE")
        calculate_and_log_average(total_cham_slice128_sc, "Slice128 SC")


    print("All Done!\n")

if __name__ == "__main__":
    # Define your dataset and output directories here
    root_dir = "/home/hi5lab/pointcloud_data/ModelNet40"
    slice64_rle_dir = "/home/hi5lab/pointcloud_data/ModelNet40_binary_voxel/slice_64_voxel_rle"
    slice64_sc_dir = "/home/hi5lab/pointcloud_data/ModelNet40_binary_voxel/slice_64_voxel_sc"
    slice128_rle_dir = "/home/hi5lab/pointcloud_data/ModelNet40_binary_voxel_128/slice_128_voxel_rle"
    slice128_sc_dir = "/home/hi5lab/pointcloud_data/ModelNet40_binary_voxel_128/slice_128_voxel_sc"

    iterate_modelnet40(root_dir, slice64_rle_dir, slice64_sc_dir, slice128_rle_dir, slice128_sc_dir)