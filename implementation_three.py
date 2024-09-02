import open3d as o3d
import numpy as np
from bitarray import bitarray
from tqdm import tqdm

import math

from utils import normalize, visualize_grid, create_xyz_line, create_box

def binary_your_pointcloud(pcd, slices, max_bound, min_bound):

    pcd_tree = o3d.geometry.KDTreeFlann(pcd)

    # Calculate step sizes for X and Y axes
    size = max_bound - min_bound
    step_x = size[0] / slices
    step_y = size[1] / slices
    step_z = size[2] / slices

    # Determine threshold value
    # This should be configurable
    x_thresh = step_x / 2
    y_thresh = step_y / 2
    z_thresh = step_z / 2
    threshold = np.max([x_thresh, y_thresh, z_thresh], axis=0)

    # Initialize the grid as a 1D binary array
    grid = bitarray(slices * slices * slices)
    grid.setall(0)  # Initialize all bits to 0
    x_count = 0
    y_count = 0
    z_count = 0
    num_of_ones = 0

    for i in range(len(grid)):
        x_pos = min_bound[0] + step_x * x_count
        y_pos = min_bound[1] + step_y * y_count
        z_pos = min_bound[2] + step_z * z_count

        query_point = np.asarray([x_pos, y_pos, z_pos])
        [k, _, _] = pcd_tree.search_radius_vector_3d(query_point, threshold)

        # Update specific x, y, and z counts
        x_count = x_count + 1
        if x_count == slices + 1:
            x_count = 0
            y_count = y_count + 1
            if y_count == slices + 1:
                y_count = 0
                z_count = z_count + 1
                if z_count == slices + 1:
                    z_count = 0

        if k > 0:
            grid[i] = 1
            num_of_ones = num_of_ones + 1
        else:
            grid[i] = 0

    return grid, num_of_ones

def rle_encode_variable_length(bitarr):
    encoded = bitarray()
    length = len(bitarr)
    max_run_length = 0
    run_lengths = []

    # Find the longest run of 0s or 1s
    current_run_length = 0
    current_bit = 0  # We always start encoding with a run of 0s

    for bit in bitarr:
        if bit == current_bit:
            current_run_length += 1
        else:
            run_lengths.append(current_run_length)
            max_run_length = max(max_run_length, current_run_length)
            current_run_length = 1
            current_bit = bit

    # Append the last run length
    run_lengths.append(current_run_length)
    max_run_length = max(max_run_length, current_run_length)

    # Determine the bit length required to encode the longest run length
    bits_needed = math.ceil(math.log2(max_run_length + 1))

    # Encode the bit length required in the first 4 bits of the encoded array
    bits_needed_bin = bin(bits_needed)[2:].zfill(16)
    encoded.extend(bitarray(bits_needed_bin))

    # Encode each run length using the calculated bit length
    for run_length in run_lengths:
        run_length_bin = bin(run_length)[2:].zfill(bits_needed)
        encoded.extend(bitarray(run_length_bin))

    return encoded

def rle_decode_variable_length(encoded_bitarr):
    decoded = bitarray()
    
    # Decode the first 4 bits to determine the bit length of each segment
    bits_needed = int(encoded_bitarr[:16].to01(), 2)
    index = 16

    current_bit = 0  # We start decoding with 0s

    while index < len(encoded_bitarr):
        run_length_bin = encoded_bitarr[index:index + bits_needed].to01()
        run_length = int(run_length_bin, 2)
        index += bits_needed
        
        # Append the decoded run length of 0s or 1s
        decoded.extend(bitarray([current_bit]) * run_length)

        # Toggle current bit for next run length
        current_bit = 1 - current_bit

    return decoded


def decode_binary(points, slices, size, min_bound):
    # Reshape the binary vector into a 3D grid
    grid = points.reshape((slices, slices, slices))
    # Calculate step sizes
    step_x = size[0] / slices
    step_y = size[1] / slices
    step_z = size[2] / slices

    # Generate point cloud from the grid
    grid_points = []
    # Iterate through points and mark the grid
    x_count = 0
    y_count = 0
    z_count = 0
    for i in range(len(grid.flatten())):
        # If 1, point is there. Place point based on index.
        if points[i] == 1:
            x_pos = min_bound[0] + x_count * (step_x)
            y_pos = min_bound[1] + y_count * (step_y)
            z_pos = min_bound[2] + z_count * (step_z)
            grid_points.append([x_pos, y_pos, z_pos])
        
        # Update specific x, y, and z counts
        x_count = x_count + 1
        if x_count == slices + 1:
            x_count = 0
            y_count = y_count + 1
            if y_count == slices + 1:
                y_count = 0
                z_count = z_count + 1
                if z_count == slices + 1:
                    z_count = 0

        # Error checking
        if x_count >= slices + 1:
            print(f"Error in counting logic at count {i} for x_count")
        if y_count >= slices + 1:
            print(f"Error in counting logic at count {i} for y_count")
        if z_count >= slices + 1:
            print(f"Error in counting logic at count {i} for z_count")

    # Convert to NumPy array
    grid_points = np.array(grid_points)

    return grid_points

if __name__ == '__main__':
    ## Variables and Initial Object loading
    slices = 128
    mesh = o3d.io.read_triangle_mesh("data/sofa_0298.off")
    print(np.shape(mesh.vertices))

    points_normalized = normalize(np.asarray(mesh.vertices))
    min_bound = np.min(points_normalized, axis=0)
    max_bound = np.max(points_normalized, axis=0)
    size = max_bound - min_bound
    mesh.vertices = o3d.utility.Vector3dVector(points_normalized)
    print(f"Max Bound: {max_bound}")
    print(f"Min Bound: {min_bound}")
    print(f"Size: {size}")

    # Lineset
    grid_lines = visualize_grid(min_bound, max_bound, slices)
    xyz_lines = create_xyz_line(min_bound, max_bound, slices)
    
    # Optionally visualize the normalized point cloud
    point_cloud_normalized = o3d.geometry.PointCloud()
    point_cloud_normalized.points = o3d.utility.Vector3dVector(mesh.vertices)
    o3d.visualization.draw_geometries([point_cloud_normalized, xyz_lines])

    ba, _ = binary_your_pointcloud(point_cloud_normalized, slices, max_bound, min_bound)
    # with open('data/bowl_0001_slice64.bin', 'rb') as f:
    #     ba = bitarray()
    #     ba.fromfile(f)

    ba = rle_encode_variable_length(ba)
    with open('data/rle128_encoded_sofa_0298.bin', 'wb') as f:
        ba.tofile(f)

    ba = rle_decode_variable_length(ba)
    numpy_array_loaded = np.array(ba.tolist(), dtype=np.uint8)

    # Decode it
    grid_points = decode_binary(numpy_array_loaded, slices, size, min_bound)

    print(grid_points)
    print(np.shape(grid_points))

    # Create a point cloud from the grid points
    point_cloud_reconstructed = o3d.geometry.PointCloud()
    point_cloud_reconstructed.points = o3d.utility.Vector3dVector(grid_points)

    # Visualize the point cloud
    o3d.visualization.draw_geometries([point_cloud_reconstructed, xyz_lines])

