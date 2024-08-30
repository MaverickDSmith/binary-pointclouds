import open3d as o3d
import numpy as np

from utils import normalize, visualize_grid
from implementation_two import binary_your_pointcloud, decode_binary
from voxel_downsample import density_aware_downsampling

# Encoding Experiments to try
# Looking for:
#   * Speed
#   * Storage Size
#   * Accuracy
# 1.) [First implementation that works at all]: Fit, then flatten
#       - moves every point to an intersection
#       - makes every other space a zero
# 2.) Fit, then iterate to create the 1D vector
#       - more control over the placement of the binary for interpreting Z from the index
#       - This may be just the same as the flatten command anyways?

def main():

    ## Variables and Initial Object loading
    slices = 128
    mesh = o3d.io.read_triangle_mesh("data/sofa_0166.off")


    #Optionally visualize it as a Point Cloud
    #point_cloud = o3d.geometry.PointCloud(mesh.vertices)
    #o3d.visualization.draw_geometries([point_cloud])

    ## Normalize
    points = np.asarray(mesh.vertices)
    points_normalized = normalize(points)
    min_bound = np.min(points_normalized, axis=0)
    max_bound = np.max(points_normalized, axis=0)
    size = max_bound - min_bound

    ## Create grid
    #grid = visualize_grid(min_bound, max_bound, slices)

    # Create a point cloud from the normalized points
    point_cloud_normalized = o3d.geometry.PointCloud()
    point_cloud_normalized.points = o3d.utility.Vector3dVector(points_normalized)

    # Optionally visualize the normalized point cloud
    # o3d.visualization.draw_geometries([point_cloud_normalized])

    # Convert point cloud to a binary point cloud
    bar = binary_your_pointcloud(point_cloud_normalized, slices, max_bound, min_bound)
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
    target_size = np.shape(point_cloud_reconstructed.points)[0]
    voxel_cloud = point_cloud_normalized.voxel_down_sample(0.001)
    if (np.shape(voxel_cloud.points)[0] > target_size):
        voxel_cloud = density_aware_downsampling(voxel_cloud, target_size=target_size, voxel_size=0.01)

    print("Number of Points in Binary Cloud: ", target_size)
    print("Number of Points in Voxel Cloud: ", np.shape(voxel_cloud.points)[0])

    o3d.visualization.draw_geometries([voxel_cloud])
    o3d.io.write_point_cloud("data/voxel.pcd", voxel_cloud, write_ascii=False)

if __name__ == '__main__':
    main()