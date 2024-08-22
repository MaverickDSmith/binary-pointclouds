# Testing, doesn't work yet. Trying to find a way to save a point in a position assuming it fits a threshold, and also save as a 1D Binary Vector instead of a 3D Binary Vector being flattened.
# def binary_your_pointcloud(points, slices, max_bound, min_bound):
#     # Calculate step sizes for X and Y axes
#     size = max_bound - min_bound
#     print(max_bound)
#     print(min_bound)
#     print(size)
#     step_x = size[0] / slices
#     step_y = size[1] / slices
#     step_z = size[2] / slices

#     x_thresh = step_x / 4
#     y_thresh = step_y / 4
#     z_thresh = step_z / 4

#     # Initialize the grid as a 1D binary array
#     grid = np.empty((slices * slices * slices))

#     # Iterate through points and mark the grid
#     for x in range(slices):
#         x_pos = step_x * (x + 1)
#         for y in range(slices):
#             y_pos = step_y * (y + 1)
#             for z in range(slices):
#                 z_pos = step_z * (z + 1)
#                 center = np.asarray([x_pos, y_pos, z_pos])

#                  # Define the bounding box (AABB) centered on the point
#                 min_bound = center - 0.4
#                 max_bound = center + 0.4
                
#                 aabb = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
#                 print("here")
                
#                 # Crop the point cloud using the bounding box
#                 cropped_pcd = points.crop(aabb)
                
#                 # Check if the cropped point cloud is non-empty
#                 if (np.asarray(cropped_pcd.points).shape[0] > 0):
#                     # Mark the grid cell as occupied
#                     grid.append(1)
#                 else:
#                     grid.append(0)
#                 print(x)
        
#     # Flatten the 2D grid to a 1D binary vector
#     #binary_vector = grid.flatten()

#     # Optionally compress and save as before
#     ba = bitarray(grid)
#     with open('data/bitarray_2d.bin', 'wb') as f:
#         ba.tofile(f)

#     return ba


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