import open3d as o3d
import numpy as np
from bitarray import bitarray
from tqdm import tqdm
import time

def density_aware_downsampling(pcd, target_size):
    points = np.asarray(pcd.points)
    
    # Create voxel grid to approximate densities
    voxel_size = 0.1  # Adjust voxel size as needed
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size)
    
    voxel_indices = np.asarray([voxel.grid_index for voxel in voxel_grid.get_voxels()])
    unique_voxels, voxel_counts = np.unique(voxel_indices, axis=0, return_counts=True)
    
    # Create density array
    densities = np.zeros(points.shape[0])
    for i, voxel_index in enumerate(voxel_indices):
        voxel_density = voxel_counts[np.all(unique_voxels == voxel_index, axis=1)][0]
        densities[i] = voxel_density
    
    # Select indices based on density
    indices = np.argsort(densities)
    selected_indices = indices[:target_size]
    downsampled_pcd = pcd.select_by_index(selected_indices)
    
    return downsampled_pcd

def binary_your_pointcloud(points, slices, max_bound, min_bound):
    ## Binary time
    # Compute the bounding box
    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)
    size = max_bound - min_bound

    # Calculate step sizes
    step_x = size[0] / slices
    step_y = size[1] / slices
    step_z = size[2] / slices

    # Initialize the grid as a 3D binary array
    grid = np.zeros((slices, slices, slices), dtype=int)

    # # Define a threshold distance for "closeness" (adjust as needed)
    # threshold = 0.01

    # Iterate through points and mark the grid
    for point in points:
        x_idx = int((point[0] - min_bound[0]) / step_x)
        y_idx = int((point[1] - min_bound[1]) / step_y)
        z_idx = int((point[2] - min_bound[2]) / step_z)

        # Ensure indices are within bounds
        x_idx = min(max(x_idx, 0), slices - 1)
        y_idx = min(max(y_idx, 0), slices - 1)
        z_idx = min(max(z_idx, 0), slices - 1)

        # Mark the grid cell as occupied
        grid[x_idx, y_idx, z_idx] = 1
    print(np.shape(grid))

    # Flatten the 3D grid to a 1D binary vector
    binary_vector = grid.flatten()
    print(np.shape(binary_vector))
    np.save("data/bin.npy", binary_vector)
    print(type(binary_vector))
    ba = bitarray(binary_vector.tolist())
    with open('data/bitarray.bin', 'wb') as f:
        ba.tofile(f)

    with open('data/bitarray.bin', 'rb') as f:
        bar = bitarray()
        bar.fromfile(f)

    return bar

def decode_binary(points, slices, size, min_bound):
    # Reshape the binary vector into a 3D grid
    grid = points.reshape((slices, slices, slices))

    # Calculate step sizes
    step_x = size[0] / slices
    step_y = size[1] / slices
    step_z = size[2] / slices

    # Generate point cloud from the grid
    grid_points = []
    for i in tqdm(range(slices)):
        for j in range(slices):
            for k in range(slices):
                if grid[i, j, k] == 1:
                    x = min_bound[0] + (i + 0.5) * step_x
                    y = min_bound[1] + (j + 0.5) * step_y
                    z = min_bound[2] + (k + 0.5) * step_z
                    grid_points.append([x, y, z])

    # Convert to NumPy array
    grid_points = np.array(grid_points)

    return grid_points


def normalize(points):
    # Compute the bounding box
    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)
    center = (min_bound + max_bound) / 2
    scale = max(max_bound - min_bound)
    size = max_bound - min_bound
    pointcloud = (points - center) / scale

    return pointcloud, min_bound, max_bound, size



def main():
    ## Load and Display Initial Point Cloud
    slices = 64

    mesh = o3d.io.read_triangle_mesh("data/sofa_0166.off")

    #Optionally visualize it as a Point Cloud
    #point_cloud = o3d.geometry.PointCloud(mesh.vertices)
    #o3d.visualization.draw_geometries([point_cloud])

    ## Normalize
    points = np.asarray(mesh.vertices)
    points_normalized, min_bound, max_bound, size = normalize(points)

    # Create a point cloud from the normalized points
    point_cloud_normalized = o3d.geometry.PointCloud()
    point_cloud_normalized.points = o3d.utility.Vector3dVector(points_normalized)
    o3d.visualization.draw_geometries([point_cloud_normalized])

    # Convert point cloud to a binary point cloud
    bar = binary_your_pointcloud(points, slices, max_bound, min_bound)
    numpy_array_loaded = np.array(bar.tolist(), dtype=np.uint8)

    # Decode it
    grid_points = decode_binary(numpy_array_loaded, slices, size, min_bound)

    # Create a point cloud from the grid points
    point_cloud_reconstructed = o3d.geometry.PointCloud()
    point_cloud_reconstructed.points = o3d.utility.Vector3dVector(grid_points)

    o3d.io.write_point_cloud("data/binary_pcd.pcd", point_cloud_reconstructed)

    # Visualize the point cloud
    o3d.visualization.draw_geometries([point_cloud_reconstructed])

    # Voxel Approach for Comparison
    voxel_cloud = point_cloud_normalized.voxel_down_sample(0.025)
    downsample_cloud = density_aware_downsampling(voxel_cloud, 5763)

    o3d.visualization.draw_geometries([downsample_cloud])
    o3d.io.write_point_cloud("data/voxel.pcd", downsample_cloud, write_ascii=False)

if __name__ == '__main__':
    main()