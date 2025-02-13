from bitarray import bitarray
import numpy as np
import open3d as o3d
from utils.utils import normalize, create_xyz_line

from binary_encoder import rle_decode_variable_length, decode_binary, sc_decode_variable_length_with_bounds, rle_decode_variable_length_voxels


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



# Plant
og_path = "/home/hi5lab/pointcloud_data/ModelNet40/plant/train/plant_0228.off"
ba_path = '/home/hi5lab/pointcloud_data/storage_test_two/slice64/plant/test/plant_0228_slice64.bin'
vox_path = '/home/hi5lab/pointcloud_data/uniformsampled_ModelNet40_Dataset_2048/slice_64_voxel_rle/plant/train/plant_0228_voxel_rle.bin'
sc_path = '/home/hi5lab/pointcloud_data/uniformsampled_ModelNet40_Dataset_2048/slice_64_voxel_sc/plant/train/plant_0228_voxel_sc.bin'




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

print(np.shape(grid_points_test_sc))

# Create a reconstructed point cloud from the grid points
point_cloud_reconstructed_test_sc = o3d.geometry.PointCloud()
point_cloud_reconstructed_test_sc.points = o3d.utility.Vector3dVector(grid_points_test_sc)

# Uncomment and load the voxelized point cloud if needed
# point_cloud_voxel = o3d.io.read_point_cloud(vox_path)


# Add voxelized point cloud to the list if available
# point_clouds.append(point_cloud_voxel)




# ## Testing loader
# ba_xyz_test = bitarray()
# with open(sc_path, 'rb') as f:
#     ba_xyz_test = f.read()

# ba_xyz_test, min_bound, max_bound = sc_decode_variable_length_with_bounds(ba_xyz_test)
# size = max_bound - min_bound

# ba_unpacked = np.frombuffer(ba_xyz_test.unpack(zero=b'\x00', one=b'\x01'), dtype=np.uint8)

# anchor_tensor = np.reshape(ba_unpacked, (65, 65, 65))
# anchor_tensor = anchor_tensor.transpose(2, 1, 0) # Changes order so it lines up with other visualizations
# occupied_points = np.argwhere(anchor_tensor == 1)  # Shape: (num_points, 3)


# xyz_channels = occupied_points.T  # Shape: (3, num_points)

# current_size = xyz_channels.shape[1]

# if current_size >= 2048:
#     xyz_channels = xyz_channels[:, :2048]  # Trim if already at or above target size

# # Randomly duplicate existing points until reaching the target size
# indices = np.random.choice(current_size, size=(2048 - current_size), replace=True)
# padded_points = np.concatenate([xyz_channels, xyz_channels[:, indices]], axis=1)

# # Convert to torch tensor
# print(np.shape(padded_points))
# print(padded_points)

# padded_points = padded_points.T


# # Create a reconstructed point cloud from the grid points
# padded_points_pointcloud = o3d.geometry.PointCloud()
# padded_points_pointcloud.points = o3d.utility.Vector3dVector(padded_points)


# Define colors for the point clouds (red, green, blue, etc)
colors = [
    (1.0, 0.0, 0.0),  # Red for the original input point cloud
    (0.0, 1.0, 0.0),  # Green for the reconstructed point cloud
    (0.0, 0.0, 1.0)   # Blue for the test rle point cloud
    # (1.0, 0.0, 1.0),  # Purple for the test sc point cloud
    # (1.0, 1.0, 0.0)
]

o3d.io.write_point_cloud("padded.ply", padded_points_pointcloud)
o3d.io.write_point_cloud("sc_encoded.ply", point_cloud_reconstructed_test_sc)

# Prepare the point clouds for visualization
point_clouds = [point_cloud_input, point_cloud_reconstructed, point_cloud_reconstructed_test, point_cloud_reconstructed_test_sc, padded_points_pointcloud]

# Visualize all point clouds together
visualize_point_clouds(point_clouds, spacing=1.0, colors=colors)

# Add XYZ lines for context (optional)
# o3d.visualization.draw_geometries([*point_clouds, xyz_lines])