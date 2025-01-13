import numpy as np
import open3d as o3d

def density_aware_downsampling(pcd, target_size, voxel_size):
    points = np.asarray(pcd.points)
    
    # Create voxel grid to approximate densities
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