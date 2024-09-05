import numpy as np
from scipy.spatial import distance
import open3d as o3d

from implementation_three 




## Add Evaluation Functions for:
## - Similarity between point clouds
## - - Methods to look into:
## - - Modified Hausdorff Distance
## - - Chamfer's Distance
## - - Point-to-Point Euclidean Distance
## - - Jaccard Similarity
## - - Earth Mover's Distance
## - - Structural Similarity Index
## - - Entropy Based Measures
## - - Normal Vector Comparison (?)
## - Accuracy of the decoded binary point cloud


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
    def directed_mean_hausdorff(A, B):
        # Compute the mean of the minimum distances from A to B
        return np.mean(np.min(distance.cdist(A, B), axis=1))
    
    # Compute mean Hausdorff distances in both directions
    dist_A_to_B = directed_mean_hausdorff(A, B)
    dist_B_to_A = directed_mean_hausdorff(B, A)
    
    # Return the average of the two
    return (dist_A_to_B + dist_B_to_A) / 2

# Example Usage
mesh = o3d.io.read_triangle_mesh("data/sofa_0166.off")
A = mesh.points
B = 
mhd_mean = modified_hausdorff_mean(A, B)
print("Mean-based Modified Hausdorff Distance:", mhd_mean)