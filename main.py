import open3d as o3d
import numpy as np
from bitarray import bitarray
from tqdm import tqdm

def density_aware_downsampling(pcd, target_size):
    # Convert point cloud to numpy array
    points = np.asarray(pcd.points)
    
    # Compute point densities
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    densities = np.zeros(points.shape[0])
    for i in range(points.shape[0]):
        [_, idx, _] = pcd_tree.search_radius_vector_3d(points[i], 0.1)  # Adjust radius as needed
        densities[i] = len(idx)

    # Create indices based on density
    indices = np.argsort(densities)
    selected_indices = indices[:target_size]  # Keep fewer points in denser regions
    
    # Select points based on indices
    downsampled_pcd = pcd.select_by_index(selected_indices)
    
    return downsampled_pcd

## Load and Display Initial Point Cloud

mesh = o3d.io.read_triangle_mesh("data/sofa_0166.off")
point_cloud = o3d.geometry.PointCloud(mesh.vertices)
#o3d.visualization.draw_geometries([point_cloud])

## Normalize
points = np.asarray(mesh.vertices)
print("Original Shape: ", np.shape(points))

# Compute the bounding box
min_bound = np.min(points, axis=0)
max_bound = np.max(points, axis=0)
center = (min_bound + max_bound) / 2
scale = max(max_bound - min_bound)

# Translate and scale the points
points_normalized = (points - center) / scale

# Create a point cloud from the normalized points
point_cloud_normalized = o3d.geometry.PointCloud()
point_cloud_normalized.points = o3d.utility.Vector3dVector(points_normalized)

# Visualize the normalized point cloud
#o3d.visualization.draw_geometries([point_cloud_normalized])

## Binary time
size = max_bound - min_bound

# Define the number of slices
num_slices_x = 64
num_slices_y = 64
num_slices_z = 64

# Calculate step sizes
step_x = size[0] / num_slices_x
step_y = size[1] / num_slices_y
step_z = size[2] / num_slices_z

# Initialize the grid as a 3D binary array
grid = np.zeros((num_slices_x, num_slices_y, num_slices_z), dtype=int)

# Define a threshold distance for "closeness" (adjust as needed)
threshold = 0.01

# Iterate through points and mark the grid
for point in points:
    x_idx = int((point[0] - min_bound[0]) / step_x)
    y_idx = int((point[1] - min_bound[1]) / step_y)
    z_idx = int((point[2] - min_bound[2]) / step_z)

    # Ensure indices are within bounds
    x_idx = min(max(x_idx, 0), num_slices_x - 1)
    y_idx = min(max(y_idx, 0), num_slices_y - 1)
    z_idx = min(max(z_idx, 0), num_slices_z - 1)

    # Mark the grid cell as occupied
    grid[x_idx, y_idx, z_idx] = 1

# Flatten the 3D grid to a 1D binary vector
binary_vector = grid.flatten()
np.save("data/bin.npy", binary_vector)
print(type(binary_vector))
ba = bitarray(binary_vector.tolist())
with open('data/bitarray.bin', 'wb') as f:
    ba.tofile(f)

with open('data/bitarray.bin', 'rb') as f:
    bar = bitarray()
    bar.fromfile(f)

numpy_array_loaded = np.array(bar.tolist(), dtype=np.uint8)

## Decode it
# Reshape the binary vector into a 3D grid
grid = numpy_array_loaded.reshape((num_slices_x, num_slices_y, num_slices_z))

# Compute the bounding box min bounds based on the grid and step sizes
min_bound = np.array([0, 0, 0])  # Replace with actual min bounds
size = np.array([step_x * num_slices_x, step_y * num_slices_y, step_z * num_slices_z])

# Generate point cloud from the grid
grid_points = []
for i in tqdm(range(num_slices_x)):
    for j in range(num_slices_y):
        for k in range(num_slices_z):
            if grid[i, j, k] == 1:
                x = min_bound[0] + (i + 0.5) * step_x
                y = min_bound[1] + (j + 0.5) * step_y
                z = min_bound[2] + (k + 0.5) * step_z
                grid_points.append([x, y, z])

# Convert to NumPy array
grid_points = np.array(grid_points)
print("Binary Shape: ", np.shape(grid_points))

# Create a point cloud from the grid points
point_cloud_reconstructed = o3d.geometry.PointCloud()
point_cloud_reconstructed.points = o3d.utility.Vector3dVector(grid_points)

o3d.io.write_point_cloud("data/binary_pcd.pcd", point_cloud_reconstructed)

# Visualize the point cloud
o3d.visualization.draw_geometries([point_cloud_reconstructed])



print("Here")
voxel_cloud = point_cloud_normalized.voxel_down_sample(0.05)
print("Here 2")
downsample_cloud = density_aware_downsampling(voxel_cloud, 5763)
print("Here 3")
o3d.visualization([downsample_cloud])
o3d.io.write_point_cloud("data/voxel.pcd", downsample_cloud, write_ascii=False)