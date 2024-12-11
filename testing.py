from data_augmentation_experiments import reordering_bitarray
from binary_encoder_test import rle_decode_variable_length, decode_binary

from bitarray import bitarray
import numpy as np
import open3d as o3d
from utils.utils import normalize, create_xyz_line

og_path = '/home/hi5lab/pointcloud_data/ModelNet40/sofa/train/sofa_0166.off'
ba_path = '/home/hi5lab/pointcloud_data/storage_test_two/slice64_og/sofa/sofa_0166_slice64.bin'
vox_path = '/home/hi5lab/pointcloud_data/storage_test_two/voxel64/sofa/sofa_0166_voxel64.pcd'


slices = 64
mesh = o3d.io.read_triangle_mesh(og_path)
print(np.shape(mesh.vertices))

points_normalized = normalize(np.asarray(mesh.vertices))
min_bound = np.min(points_normalized, axis=0)
max_bound = np.max(points_normalized, axis=0)
size = max_bound - min_bound
mesh.vertices = o3d.utility.Vector3dVector(points_normalized)
print(f"Max Bound: {max_bound}")
print(f"Min Bound: {min_bound}")
print(f"Size: {size}")
point_cloud_input = o3d.geometry.PointCloud()
point_cloud_input.points = mesh.vertices
# o3d.visualization.draw_geometries([point_cloud_input])


ba = bitarray()
with open(ba_path, 'rb') as f:
        ba.fromfile(f)

ba, min_bound, max_bound = rle_decode_variable_length(ba)
numpy_array_loaded = np.array(ba.tolist(), dtype=np.uint8)
size = max_bound - min_bound

# Decode it
grid_points = decode_binary(numpy_array_loaded, 64, size, min_bound)
xyz_lines = create_xyz_line(min_bound, max_bound, 64)


# xyz_lines = create_xyz_line(min_bound, max_bound, slices)

print(grid_points)
print(np.shape(grid_points))

# Create a point cloud from the grid points
point_cloud_reconstructed = o3d.geometry.PointCloud()
point_cloud_reconstructed.points = o3d.utility.Vector3dVector(grid_points)

point_cloud_voxel = o3d.io.read_point_cloud(vox_path)


# Transpose the points
transpose_axes = (0, 2, 1)  # Example: Swap Y and Z
# vectorized_points, new_min_bound, new_max_bound = vectorization(
#     numpy_array_loaded, 
#     transpose_axes, 
#     (65, 65, 65), 
#     min_bound, 
#     max_bound
# )

vectorized_points, n_m_b, n_ma_b = reordering_bitarray(numpy_array_loaded, 65, (1, 0, 2), min_bound, max_bound)
n_size = n_m_b - n_ma_b

print(f"New Max Bound: {n_ma_b}")
print(f"New Min Bound: {n_m_b}")
print(f"New Size: {n_size}")

# Ensure min_bound and max_bound are NumPy arrays
# new_min_bound = np.array(new_min_bound)
# new_max_bound = np.array(new_max_bound)

# # Calculate the size after transposing
# new_size = new_max_bound - new_min_bound  # This works with NumPy arrays

# print(f"New Max Bound: {new_max_bound}")
# print(f"New Min Bound: {new_min_bound}")
# print(f"New Size: {new_size}")


# Decode the transposed binary array with updated bounds
vectorized_grid_points = decode_binary(vectorized_points, 64, n_size, n_m_b)
print(np.shape(vectorized_grid_points))

new_xyz_lines = create_xyz_line(n_m_b, n_ma_b, 64)

# Create and visualize the new point cloud
vectorized_point_cloud_reconstructed = o3d.geometry.PointCloud()
vectorized_point_cloud_reconstructed.points = o3d.utility.Vector3dVector(vectorized_grid_points)



# Visualize the point cloud
# o3d.visualization.draw_geometries([point_cloud_input, xyz_lines])
# o3d.visualization.draw_geometries([point_cloud_voxel, xyz_lines])
o3d.visualization.draw_geometries([point_cloud_reconstructed, xyz_lines])
o3d.visualization.draw_geometries([vectorized_point_cloud_reconstructed, new_xyz_lines])


