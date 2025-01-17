from bitarray import bitarray
import numpy as np
import open3d as o3d
from utils.utils import normalize, create_xyz_line

from binary_encoder import rle_decode_variable_length, decode_binary, sc_decode_variable_length_with_bounds, rle_decode_variable_length_voxels

# og_path = '/home/hi5lab/pointcloud_data/ModelNet40/glass_box/train/glass_box_0139.off'
# ba_path = '/home/hi5lab/pointcloud_data/storage_test_three/slice64_test/glass_box/glass_box_0139_slice64_test.bin'
# # vox_path = '/home/hi5lab/pointcloud_data/storage_test_two/voxel64/sofa/sofa_0166_voxel64.pcd'


# slices = 64
# mesh = o3d.io.read_triangle_mesh(og_path)
# print(np.shape(mesh.vertices))

# points_normalized = normalize(np.asarray(mesh.vertices))
# min_bound = np.min(points_normalized, axis=0)
# max_bound = np.max(points_normalized, axis=0)
# size = max_bound - min_bound
# mesh.vertices = o3d.utility.Vector3dVector(points_normalized)
# print(f"Max Bound: {max_bound}")
# print(f"Min Bound: {min_bound}")
# print(f"Size: {size}")
# point_cloud_input = o3d.geometry.PointCloud()
# point_cloud_input.points = mesh.vertices
# # o3d.visualization.draw_geometries([point_cloud_input])


# ba = bitarray()
# with open(ba_path, 'rb') as f:
#         ba.fromfile(f)

# ba, min_bound, max_bound = rle_decode_variable_length(ba)
# numpy_array_loaded = np.array(ba.tolist(), dtype=np.uint8)
# size = max_bound - min_bound

# # Decode it
# grid_points = decode_binary(numpy_array_loaded, 64, size, min_bound)
# xyz_lines = create_xyz_line(min_bound, max_bound, 64)


# # xyz_lines = create_xyz_line(min_bound, max_bound, slices)

# # print(grid_points)
# print(np.shape(grid_points))

# # Create a point cloud from the grid points
# point_cloud_reconstructed = o3d.geometry.PointCloud()
# point_cloud_reconstructed.points = o3d.utility.Vector3dVector(grid_points)

# # point_cloud_voxel = o3d.io.read_point_cloud(vox_path)



# # Visualize the point cloud
# o3d.visualization.draw_geometries([point_cloud_input, xyz_lines])
# # o3d.visualization.draw_geometries([point_cloud_voxel, xyz_lines])
# o3d.visualization.draw_geometries([point_cloud_reconstructed, xyz_lines])

def visualize_point_clouds(pcds, spacing=1.5, colors=None):
    """
    Visualize multiple point clouds in the same window, spaced apart and with specific colors.

    Parameters:
        pcds (list of open3d.geometry.PointCloud): List of point clouds to visualize.
        spacing (float): Distance to space apart each point cloud.
        colors (list of tuples): List of RGB color values for each point cloud.

    Returns:
        None
    """
    for i, pcd in enumerate(pcds):
        # Apply translation to space apart point clouds
        translation = np.array([i * spacing, 0, 0])
        pcd.translate(translation)

        # Apply color if provided
        if colors and i < len(colors):
            pcd.paint_uniform_color(colors[i])

    # Visualize all point clouds together
    o3d.visualization.draw_geometries(pcds)

# File paths
# og_path = "/home/hi5lab/pointcloud_data/ModelNet40/sofa/train/sofa_0166.off"
# ba_path = '/home/hi5lab/wsl_github/github_ander/Fall 2024/binary-pointclouds/data/rle_encoded_sofa_0166.bin'
# vox_path = '/home/hi5lab/wsl_github/github_ander/Fall 2024/binary-pointclouds/data/rle_encoded_sofa_test_0166.bin'
# sc_path = '/home/hi5lab/wsl_github/github_ander/Fall 2024/binary-pointclouds/data/sc_encoded_sofa_test_0166.bin'


og_path = "/home/hi5lab/pointcloud_data/ModelNet40/dresser/test/dresser_0228.off"
ba_path = '/home/hi5lab/pointcloud_data/storage_test_two/slice64/dresser/train/dresser_0228_slice64.bin'
vox_path = '/home/hi5lab/pointcloud_data/dataset/slice_64_voxel_rle/dresser/dresser_0228_slice64_rle.bin'
sc_path = '/home/hi5lab/pointcloud_data/dataset/slice_64_voxel_sc/dresser/dresser_0228_slice64_sc.bin'


slices = 64

# Load and normalize the mesh
mesh = o3d.io.read_triangle_mesh(og_path)
points_normalized = normalize(np.asarray(mesh.vertices))
min_bound = np.min(points_normalized, axis=0)
max_bound = np.max(points_normalized, axis=0)
size = max_bound - min_bound
mesh.vertices = o3d.utility.Vector3dVector(points_normalized)
# print(f"Max Bound: {max_bound}")
# print(f"Min Bound: {min_bound}")
# print(f"Size: {size}")

# Create the original point cloud
point_cloud_input = o3d.geometry.PointCloud()
point_cloud_input.points = mesh.vertices



## Original Binary Array
# Load the binary array
ba = bitarray()
with open(ba_path, 'rb') as f:
    ba.fromfile(f)

ba, min_bound, max_bound = rle_decode_variable_length(ba)
numpy_array_loaded = np.array(ba.tolist(), dtype=np.uint8)
size = max_bound - min_bound

# Decode the binary array
grid_points = decode_binary(numpy_array_loaded, 64, size, min_bound)
# xyz_lines = create_xyz_line(min_bound, max_bound, 64)

# Create a reconstructed point cloud from the grid points
point_cloud_reconstructed = o3d.geometry.PointCloud()
point_cloud_reconstructed.points = o3d.utility.Vector3dVector(grid_points)



## RLE Voxelization Binary Array
# Load the test binary array
ba_test = bitarray()
with open(vox_path, 'rb') as f:
    ba_test.fromfile(f)

ba_test, min_bound, max_bound = rle_decode_variable_length_voxels(ba_test)
numpy_array_loaded = np.array(ba_test.tolist(), dtype=np.uint8)
size = max_bound - min_bound

print(f"Max Bound: {max_bound}")
print(f"Min Bound: {min_bound}")
print(f"Size: {size}")


# Decode the binary array
grid_points_test = decode_binary(numpy_array_loaded, 64, size, min_bound)
# xyz_lines = create_xyz_line(min_bound, max_bound, 64)

# Create a reconstructed point cloud from the grid points
point_cloud_reconstructed_test = o3d.geometry.PointCloud()
point_cloud_reconstructed_test.points = o3d.utility.Vector3dVector(grid_points_test)



## SC Voxelization Binary Array
# Load the test binary array
ba_sc_test = bitarray()
with open(sc_path, 'rb') as f:
    ba_sc_test = f.read()

ba_sc_test, min_bound, max_bound = sc_decode_variable_length_with_bounds(ba_sc_test)
numpy_array_loaded = np.array(ba_sc_test.tolist(), dtype=np.uint8)
size = max_bound - min_bound

# Decode the binary array
grid_points_test_sc = decode_binary(numpy_array_loaded, 64, size, min_bound)
# xyz_lines = create_xyz_line(min_bound, max_bound, 64)

# Create a reconstructed point cloud from the grid points
point_cloud_reconstructed_test_sc = o3d.geometry.PointCloud()
point_cloud_reconstructed_test_sc.points = o3d.utility.Vector3dVector(grid_points_test_sc)

# Uncomment and load the voxelized point cloud if needed
# point_cloud_voxel = o3d.io.read_point_cloud(vox_path)

# Prepare the point clouds for visualization
point_clouds = [point_cloud_input, point_cloud_reconstructed, point_cloud_reconstructed_test, point_cloud_reconstructed_test_sc]
# Add voxelized point cloud to the list if available
# point_clouds.append(point_cloud_voxel)

# Define colors for the point clouds (red, green, blue, etc)
colors = [
    (1.0, 0.0, 0.0),  # Red for the original input point cloud
    (0.0, 1.0, 0.0),  # Green for the reconstructed point cloud
    (0.0, 0.0, 1.0),  # Blue for the test rle point cloud
    (1.0, 0.0, 1.0)   # Purple for the test sc point cloud
]

# Visualize all point clouds together
visualize_point_clouds(point_clouds, spacing=0.5, colors=colors)

# Add XYZ lines for context (optional)
# o3d.visualization.draw_geometries([*point_clouds, xyz_lines])
