import numpy as np
from tqdm import tqdm
from bitarray import bitarray

#Notes
# This is my initial implementation. I did in fact use Chat-GPT for most of the code-writing,
# albeit with a lot of my own necessary edits in place to get the point across.
# This does not work with my initial idea for the project, but it does serve as a good
# starting point, as it does the spirit of the thing.

# This implementation encodes in O(N) time, and decodes in O(n^3) time. Since each position
# is encoded in Binary and stored in a standard bitarray format, each point roughly takes
# only 1/8th of a byte of storage. One of the constraints of this format is that we save every
# empty space in the bounding box, which is why we will always have the same size bitarray
# despite having potentially variable points in the point cloud, assuming you used the same
# number of slices for different point clouds.

# At low number of slices, for all my test cases (this doesn't mean it works in every case ever),
# we achieve high visual similarity with lower storage space than voxelization at roughly the same
# total speed that voxelization takes. I have not yet performed DL accuracy tests yet. Low number of
# slices refers to around the 64 range. Not much higher than that and we begin to see where voxelization
# stores more compactly and runs faster. However, for DL purposes, the structure of the point cloud
# with 64 slices for each axis is still in tact enough to presumably be trained.


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

    # Flatten the 3D grid to a 1D binary vector
    binary_vector = grid.flatten()
    np.save("data/bin.npy", binary_vector)
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