from bitarray import bitarray
import numpy as np
import open3d as o3d
from utils.utils import normalize, create_xyz_line

from binary_encoder import rle_decode_variable_length, decode_binary

og_path = '/home/hi5lab/pointcloud_data/ModelNet40/sofa/train/sofa_0166.off'
ba_path = '/home/hi5lab/pointcloud_data/storage_test_two/slice64/sofa/sofa_0166_slice64.bin'
vox_path = '/home/hi5lab/pointcloud_data/storage_test_two/voxel64/sofa/sofa_0166_voxel64.pcd'


slices = 64
mesh = o3d.io.read_triangle_mesh(og_path)
print(np.shape(mesh.vertices))

points_normalized = normalize(np.asarray(mesh.vertices))
min_bound = np.min(points_normalized, axis=0)
max_bound = np.max(points_normalized, axis=0)
size = max_bound - min_bound
mesh.vertices = o3d.utility.Vector3dVector(points_normalized)
print(f"Max Bound: {max_bound}")
print(f"Min Bound: {min_bound}")
print(f"Size: {size}")
point_cloud_input = o3d.geometry.PointCloud()
point_cloud_input.points = mesh.vertices
# o3d.visualization.draw_geometries([point_cloud_input])


ba = bitarray()
with open(ba_path, 'rb') as f:
        ba.fromfile(f)

ba, min_bound, max_bound = rle_decode_variable_length(ba)
numpy_array_loaded = np.array(ba.tolist(), dtype=np.uint8)
size = max_bound - min_bound

# Decode it
grid_points = decode_binary(numpy_array_loaded, 64, size, min_bound)
xyz_lines = create_xyz_line(min_bound, max_bound, 64)


# xyz_lines = create_xyz_line(min_bound, max_bound, slices)

print(grid_points)
print(np.shape(grid_points))

# Create a point cloud from the grid points
point_cloud_reconstructed = o3d.geometry.PointCloud()
point_cloud_reconstructed.points = o3d.utility.Vector3dVector(grid_points)

point_cloud_voxel = o3d.io.read_point_cloud(vox_path)



# Visualize the point cloud
o3d.visualization.draw_geometries([point_cloud_input, xyz_lines])
o3d.visualization.draw_geometries([point_cloud_voxel, xyz_lines])
o3d.visualization.draw_geometries([point_cloud_reconstructed, xyz_lines])