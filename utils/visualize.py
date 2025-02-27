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


def compute_curvature(mesh):
    """Estimate curvature using Laplacian smoothing."""
    mesh.compute_vertex_normals()
    curvature = np.linalg.norm(mesh.vertex_normals, axis=1)
    return curvature / np.max(curvature)  # Normalize

def compute_normal_variation(mesh, k=10):
    """Estimate normal variation by comparing normals of neighboring vertices."""
    pcd = mesh.sample_points_uniformly(number_of_points=len(mesh.vertices))  # Convert to point cloud
    pcd.estimate_normals()
    
    # Create KDTree for neighbor search
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    normal_variation = np.zeros(len(pcd.points))

    for i in range(len(pcd.points)):
        _, idx, _ = kdtree.search_knn_vector_3d(pcd.points[i], k)
        normals = np.asarray(pcd.normals)[idx]
        normal_variation[i] = np.linalg.norm(normals - normals.mean(axis=0), axis=1).sum()
    
    return normal_variation / np.max(normal_variation)  # Normalize

def farthest_point_sampling(points, num_samples):
    """Farthest Point Sampling (FPS) to select diverse points."""
    num_points = points.shape[0]
    selected = np.zeros(num_samples, dtype=int)
    
    # Randomly initialize the first point
    selected[0] = np.random.randint(num_points)
    distances = np.full(num_points, np.inf)

    for i in range(1, num_samples):
        last_selected = points[selected[i - 1]]
        dists = np.linalg.norm(points - last_selected, axis=1)
        distances = np.minimum(distances, dists)  # Update min distances
        selected[i] = np.argmax(distances)  # Pick farthest point

    return points[selected]

def normalize_to_unit_cube(pcd):
    """Normalize point cloud to fit within a unit cube centered at the origin."""
    points = np.asarray(pcd.points)
    
    # Compute bounding box
    min_bound = points.min(axis=0)
    max_bound = points.max(axis=0)
    center = (min_bound + max_bound) / 2.0
    scale = np.max(max_bound - min_bound)  # Largest dimension

    # Normalize
    normalized_points = (points - center) / scale

    # Update point cloud
    normalized_pcd = o3d.geometry.PointCloud()
    normalized_pcd.points = o3d.utility.Vector3dVector(normalized_points)
    
    return normalized_pcd

def feature_aware_fps_sampling(mesh, num_samples=1024, alpha=0.5, beta=0.5, oversample_factor=2):
    """Feature-Aware Sampling using Farthest Point Sampling (FPS)."""
    curvature = compute_curvature(mesh)
    normal_variation = compute_normal_variation(mesh)
    
    # Compute weighted probability
    weights = alpha * curvature + beta * normal_variation
    weights /= weights.sum()  # Normalize to sum to 1

    # Oversample based on weights
    oversample_count = num_samples * oversample_factor
    sampled_indices = np.random.choice(len(mesh.vertices), oversample_count, p=weights)
    sampled_points = np.asarray(mesh.vertices)[sampled_indices]

    # Apply FPS to enforce spatial separation
    sampled_points = farthest_point_sampling(sampled_points, num_samples)

    # Convert to Open3D PointCloud
    sampled_pcd = o3d.geometry.PointCloud()
    sampled_pcd.points = o3d.utility.Vector3dVector(sampled_points)
    
    # Normalize to unit cube
    return normalize_to_unit_cube(sampled_pcd)

# File paths
# Sofa
og_path = "/home/hi5lab/pointcloud_data/ModelNet40/sofa/train/sofa_0225.off"
ba_path = '/home/hi5lab/pointcloud_data/storage_test_two/slice64/sofa/train/sofa_0225_slice64.bin'
vox_path = '/home/hi5lab/pointcloud_data/uniformsampled_ModelNet40_Dataset_2048/slice_64_voxel_rle/sofa/train/sofa_0225_voxel_rle.bin'
sc_path = '/home/hi5lab/pointcloud_data/uniformsampled_ModelNet40_Dataset_2048/slice_64_voxel_sc/sofa/train/sofa_0225_voxel_sc.bin'

# Dresser
# og_path = "/home/hi5lab/pointcloud_data/ModelNet40/dresser/test/dresser_0228.off"
# ba_path = '/home/hi5lab/pointcloud_data/storage_test_two/slice64/dresser/train/dresser_0228_slice64.bin'
# vox_path = '/home/hi5lab/pointcloud_data/dataset/slice_64_voxel_rle/dresser/dresser_0228_slice64_rle.bin'
# sc_path = '/home/hi5lab/pointcloud_data/dataset/slice_64_voxel_sc/dresser/dresser_0228_slice64_sc.bin'

# Airplane
# og_path = "/home/hi5lab/pointcloud_data/ModelNet40/dresser/test/dresser_0228.off"
# ba_path = '/home/hi5lab/pointcloud_data/storage_test_two/slice64/dresser/train/dresser_0228_slice64.bin'
# vox_path = '/home/hi5lab/pointcloud_data/dataset/slice_64_voxel_rle/dresser/dresser_0228_slice64_rle.bin'
# sc_path = '/home/hi5lab/pointcloud_data/dataset/slice_64_voxel_sc/dresser/dresser_0228_slice64_sc.bin'

# # Plant
# og_path = "/home/hi5lab/pointcloud_data/ModelNet40/plant/train/plant_0228.off"
# ba_path = '/home/hi5lab/pointcloud_data/storage_test_two/slice64/plant/test/plant_0228_slice64.bin'
# vox_path = '/home/hi5lab/pointcloud_data/uniformsampled_ModelNet40_Dataset_2048/slice_64_voxel_rle/plant/train/plant_0228_voxel_rle.bin'
# sc_path = '/home/hi5lab/pointcloud_data/uniformsampled_ModelNet40_Dataset_2048/slice_64_voxel_sc/plant/train/plant_0228_voxel_sc.bin'

# Guitar
# og_path = "/home/hi5lab/pointcloud_data/ModelNet40/guitar/test/guitar_0228.off"
# ba_path = '/home/hi5lab/pointcloud_data/storage_test_two/slice64/guitar/train/guitar_0228_slice64.bin'
# vox_path = '/home/hi5lab/pointcloud_data/uniformsampled_ModelNet40_Dataset/slice_128_voxel_rle/guitar/test/guitar_0228_voxel_rle.bin'
# sc_path = '/home/hi5lab/pointcloud_data/uniformsampled_ModelNet40_Dataset/slice_128_voxel_sc/guitar/test/guitar_0228_voxel_sc.bin'

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


mesh = o3d.io.read_triangle_mesh("/home/hi5lab/pointcloud_data/ModelNet40/sofa/train/sofa_0225.off")
sampled_pcd = feature_aware_fps_sampling(mesh, num_samples=2048)









# Prepare the point clouds for visualization
point_clouds = [point_cloud_input, point_cloud_reconstructed, sampled_pcd, point_cloud_reconstructed_test, point_cloud_reconstructed_test_sc]
# Add voxelized point cloud to the list if available
# point_clouds.append(point_cloud_voxel)

# Define colors for the point clouds (red, green, blue, etc)
colors = [
    (1.0, 0.0, 0.0),   # Red for the original input point cloud
    (0.0, 1.0, 0.0),   # Green for the reconstructed point cloud
    (0.0, 0.0, 1.0),   # Blue for the test rle point cloud
    (1.0, 0.0, 1.0),   # Purple for the test sc point cloud
    (1.0, 1.0, 0.0)    # 
]

# Visualize all point clouds together
visualize_point_clouds(point_clouds, spacing=1.0, colors=colors)

# Add XYZ lines for context (optional)
# o3d.visualization.draw_geometries([*point_clouds, xyz_lines])
