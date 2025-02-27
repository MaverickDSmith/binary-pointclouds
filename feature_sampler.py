import open3d as o3d
import numpy as np

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

def feature_aware_sampling(mesh, num_samples=1024, alpha=0.5, beta=0.5):
    """Sample points based on feature weights (curvature + normal variation)."""
    curvature = compute_curvature(mesh)
    normal_variation = compute_normal_variation(mesh)
    
    # Compute weighted probability
    weights = alpha * curvature + beta * normal_variation
    weights /= weights.sum()  # Normalize to sum to 1

    # Sample vertices based on weights
    sampled_indices = np.random.choice(len(mesh.vertices), num_samples, p=weights)
    sampled_points = np.asarray(mesh.vertices)[sampled_indices]

    # Convert to Open3D PointCloud
    sampled_pcd = o3d.geometry.PointCloud()
    sampled_pcd.points = o3d.utility.Vector3dVector(sampled_points)
    
    return sampled_pcd

# Load a mesh and apply feature-aware sampling
mesh = o3d.io.read_triangle_mesh("/home/hi5lab/pointcloud_data/ModelNet40/sofa/train/sofa_0225.off")
sampled_pcd = feature_aware_sampling(mesh, num_samples=2048)

# Visualize results
o3d.visualization.draw_geometries([sampled_pcd])
