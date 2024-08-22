import open3d as o3d
import numpy as np
from bitarray import bitarray
from tqdm import tqdm

from utils import normalize

# Testing, doesn't work yet. Trying to find a way to save a point in a position assuming it fits a threshold, and also save as a 1D Binary Vector instead of a 3D Binary Vector being flattened.
def binary_your_pointcloud(mesh, slices, max_bound, min_bound):

    points = np.asarray(mesh.vertices)
    scene = o3d.t.geometry.RaycastingScene()
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene.add_triangles(mesh)


    # Calculate step sizes for X and Y axes
    size = max_bound - min_bound
    # print(max_bound)
    # print(min_bound)
    # print(size)
    step_x = size[0] / slices
    step_y = size[1] / slices
    step_z = size[2] / slices

    x_thresh = step_x / 4
    y_thresh = step_y / 4
    z_thresh = step_z / 4

    # query_point = o3d.core.Tensor([[step_x * (0 + 56), step_y * (0 + 25), step_z * (0 + 43)]], dtype=o3d.core.Dtype.Float32)
    # ans = scene.compute_closest_points(query_point)

    # print(ans['points'].numpy())

    # Initialize the grid as a 1D binary array
    grid = np.empty((slices * slices * slices), dtype=int)

    # Iterate through points and mark the grid
    for x in tqdm(range(slices)):
        x_pos = step_x * (x + 1)
        for y in range(slices):
            y_pos = step_y * (y + 1)
            for z in range(slices):
                z_pos = step_z * (z + 1)
                query_point = o3d.core.Tensor([[x_pos, y_pos, z_pos]], dtype=o3d.core.Dtype.Float32)
                ans = scene.compute_closest_points(query_point)
                distance = query_point - ans['points']

                # Compute boolean condition by checking if all components are within the threshold
                condition = (abs(distance.numpy()) < np.array([x_thresh, y_thresh, z_thresh])).all()

                if condition:
                    np.append(grid, 1)
                else:
                    np.append(grid, 0)

    print(grid)
    print(np.shape(grid))
        
    # Flatten the 2D grid to a 1D binary vector
    #binary_vector = grid.flatten()

    # Optionally compress and save as before
    ba = bitarray(grid.tolist())
    with open('data/bitarray_2d.bin', 'wb') as f:
        ba.tofile(f)

    with open('data/bitarray_2d.bin', 'rb') as f:
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

# def decode_binary(points, slices, size, min_bound):
#     # Reshape the binary vector into a 2D grid
#     grid = points.reshape((slices, slices))

#     # Calculate step sizes for X, Y, and Z axes
#     step_x = size[0] / slices
#     step_y = size[1] / slices
#     step_z = size[2] / slices

#     # Generate point cloud from the grid
#     grid_points = []
#     for i in range(slices):
#         for j in range(slices):
#             if grid[i, j] == 1:
#                 # Compute X and Y from the slice indices
#                 x = min_bound[0] + (i + 0.5) * step_x
#                 y = min_bound[1] + (j + 0.5) * step_y
                
#                 # Iterate through all possible Z positions
#                 for k in range(slices):
#                     z = min_bound[2] + k * step_z
#                     grid_points.append([x, y, z])

#     # Convert to NumPy array
#     grid_points = np.array(grid_points)

#     return grid_points

if __name__ == '__main__':
    ## Variables and Initial Object loading
    slices = 64
    mesh = o3d.io.read_triangle_mesh("data/sofa_0166.off")
    _, min_bound, max_bound, size = normalize(np.asarray(mesh.vertices))
    ba = binary_your_pointcloud(mesh, slices, max_bound, min_bound)
    numpy_array_loaded = np.array(ba.tolist(), dtype=np.uint8)
    # Decode it
    grid_points = decode_binary(numpy_array_loaded, slices, size, min_bound)

    print(grid_points)
    print(np.shape(grid_points))

    # Create a point cloud from the grid points
    point_cloud_reconstructed = o3d.geometry.PointCloud()
    point_cloud_reconstructed.points = o3d.utility.Vector3dVector(grid_points)

    # Visualize the point cloud
    o3d.visualization.draw_geometries([point_cloud_reconstructed])

