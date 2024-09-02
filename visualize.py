from bitarray import bitarray
import numpy as np
import open3d as o3d
from utils import normalize, create_xyz_line

from implementation_three import rle_decode_variable_length, decode_binary

og_path = '/home/hi5lab/github/github_ander/Fall 2024/data/ModelNet40/airplane/train/airplane_0130.off'
ba_path = '/home/hi5lab/github/github_ander/Fall 2024/data/storage_test_two/slice64/airplane/airplane_0130_slice64.bin'


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

xyz_lines = create_xyz_line(min_bound, max_bound, slices)


ba = bitarray()
with open(ba_path, 'rb') as f:
        ba.fromfile(f)

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