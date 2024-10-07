import numpy as np
from scipy.spatial import KDTree



## Add Evaluation Functions for:
## - Similarity between point clouds
## - - Methods to look into:
## - - Modified Hausdorff Distance [Done]
## - - Chamfer's Distance
## - - Point-to-Point Euclidean Distance
## - - Jaccard Similarity
## - - Earth Mover's Distance
## - - Structural Similarity Index
## - - Entropy Based Measures
## - - Normal Vector Comparison (?)
## - Accuracy of the decoded binary point cloud

## Graphs:
## Scatter Plot
## Bar Graph for each category
## Overall averages

def directed_mean_hausdorff(A, B):
    """
    Compute the mean of the minimum distances from points in A to their nearest neighbors in B.
    """
    # Use KDTree to find the nearest neighbors efficiently
    tree_B = KDTree(B)
    distances, _ = tree_B.query(A, k=1)  # Find the nearest neighbor for each point in A
    return np.mean(distances)

def modified_hausdorff_mean(A, B):
    """
    Compute the mean-based Modified Hausdorff Distance between two point clouds A and B.
    
    Parameters:
    A, B : np.ndarray
        Point clouds with shape (N, D), where N is the number of points and D is the dimensionality.

    Returns:
    float
        The mean-based modified Hausdorff distance.
    """
    
    # Compute mean Hausdorff distances in both directions
    dist_A_to_B = directed_mean_hausdorff(A, B)
    dist_B_to_A = directed_mean_hausdorff(B, A)
    
    # Return the average of the two
    return (dist_A_to_B + dist_B_to_A) / 2

def chamfer_distance(A, B):
    """
    Compute the Chamfer Distance between two point clouds A and B.
    
    Parameters:
    A, B : np.ndarray
        Point clouds with shape (N, D), where N is the number of points and D is the dimensionality.

    Returns:
    float
        The Chamfer distance.
    """
    # Directed Chamfer distance from A to B using KDTree
    def directed_chamfer(A, B):
        # Use KDTree to find the nearest neighbors efficiently
        tree_B = KDTree(B)
        distances, _ = tree_B.query(A, k=1)  # Find the nearest neighbor for each point in A
        return np.sum(distances)  # Sum of the minimum distances from each point in A to B

    # Compute Chamfer distances in both directions
    dist_A_to_B = directed_chamfer(A, B)
    dist_B_to_A = directed_chamfer(B, A)
    
    # Return the average Chamfer distance in both directions
    return (dist_A_to_B + dist_B_to_A) / (A.shape[0] + B.shape[0])



# # Example Usage
# mesh = o3d.io.read_triangle_mesh("data/sofa_0166.off")

# points_normalized = normalize(np.asarray(mesh.vertices))
# min_bound = np.min(points_normalized, axis=0)
# max_bound = np.max(points_normalized, axis=0)
# size = max_bound - min_bound


# ba = bitarray()

# with open('data/rle_encoded_sofa_0166.bin', 'rb') as f:
#     ba.fromfile(f)

# ba, min_bound, max_bound = rle_decode_variable_length(ba)
# numpy_array_loaded = np.array(ba.tolist(), dtype=np.uint8)

# # Decode it
# grid_points = decode_binary(numpy_array_loaded, 64, size, min_bound)

# # Voxel Approach for Comparison
# target_size = np.shape(grid_points)[0]

# # Create a point cloud from the grid points
# point_cloud_reconstructed = o3d.geometry.PointCloud()
# point_cloud_reconstructed.points = o3d.utility.Vector3dVector(points_normalized)

# voxel_cloud = point_cloud_reconstructed.voxel_down_sample(0.001)
# # if (np.shape(voxel_cloud.points)[0] > target_size):
# #     voxel_cloud = density_aware_downsampling(voxel_cloud, target_size=target_size, voxel_size=0.01)

# A = points_normalized
# B = grid_points
# C = np.asarray(voxel_cloud.points)
# mhd_mean = modified_hausdorff_mean(A, B)
# mhd_mean2 = modified_hausdorff_mean(A, C)

# aB_cmdist = chamfer_distance(A,B)
# aC_cmdist = chamfer_distance(A,C)

# print("[A, B] Mean-based Modified Hausdorff Distance:", mhd_mean)
# print("[A, C] Mean-based Modified Hausdorff Distance:", mhd_mean2)

# print("[A, B] Chamfer Distance: ", aB_cmdist)
# print("[A, C] Chamfer Distance: ", aC_cmdist)