import numpy as np
import open3d as o3d


def visualize_grid(min_bound, max_bound, slices):
    # Calculate step sizes
    size = max_bound - min_bound
    step_x = size[0] / slices
    step_y = size[1] / slices
    step_z = size[2] / slices

    # Initialize points and lines arrays
    points = []
    lines = []

    # Generate grid lines along X, Y, and Z axes
    point_count = 0
    for i in range(slices + 1):
        for j in range(slices + 1):
            # X-axis lines
            x_start = [min_bound[0] + i * step_x, min_bound[1], min_bound[2] + j * step_z]
            x_end = [min_bound[0] + i * step_x, max_bound[1], min_bound[2] + j * step_z]
            points.extend([x_start, x_end])
            lines.append([point_count, point_count + 1])
            point_count += 2

            # Y-axis lines
            y_start = [min_bound[0], min_bound[1] + i * step_y, min_bound[2] + j * step_z]
            y_end = [max_bound[0], min_bound[1] + i * step_y, min_bound[2] + j * step_z]
            points.extend([y_start, y_end])
            lines.append([point_count, point_count + 1])
            point_count += 2

            # Z-axis lines
            z_start = [min_bound[0] + i * step_x, min_bound[1] + j * step_y, min_bound[2]]
            z_end = [min_bound[0] + i * step_x, min_bound[1] + j * step_y, max_bound[2]]
            points.extend([z_start, z_end])
            lines.append([point_count, point_count + 1])
            point_count += 2

    # Convert to NumPy arrays and ensure correct types
    points = np.array(points, dtype=np.float64)
    lines = np.array(lines, dtype=np.int32)

    # Create the LineSet object
    try:
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            lines=o3d.utility.Vector2iVector(lines)
        )
    except Exception as e:
        print(f"Error creating LineSet: {e}")
        return None
    

    # Optionally, you can add colors to the lines
    colors = [[0, 0, 0] for _ in range(len(lines))]
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set

def create_xyz_line(min_bound, max_bound, slices):
    size = max_bound - min_bound

    points = [
        [0, 0, 0],
        [size[0], 0, 0], # X
        [0, size[1], 0], # Y
        [0, 0, size[2]]  # Z
    ]

    lines = [
        [0, 1],
        [0, 2],
        [0, 3]
    ]

    colors = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ]

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set

def create_box(min_bound, max_bound, slices, point):
    size = max_bound - min_bound

    step_x = size[0] / slices
    step_y = size[1] / slices
    step_z = size[2] / slices

    # Convert step sizes to Float32
    step_x = np.float32(step_x)
    step_y = np.float32(step_y)
    step_z = np.float32(step_z)

    # Convert the NumPy array to an Open3D tensor with dtype Float32
    step_tensor = o3d.core.Tensor([step_x, step_y, step_z], dtype=o3d.core.Dtype.Float32)

    # Calculate the points as Open3D tensors
    point_1 = point + step_tensor
    point_2 = point + o3d.core.Tensor([-step_x, step_y, step_z], dtype=o3d.core.Dtype.Float32)
    point_3 = point + o3d.core.Tensor([-step_x, step_y, -step_z], dtype=o3d.core.Dtype.Float32)
    point_4 = point + o3d.core.Tensor([step_x, step_y, -step_z], dtype=o3d.core.Dtype.Float32)
    point_5 = point + o3d.core.Tensor([step_x, -step_y, step_z], dtype=o3d.core.Dtype.Float32)
    point_6 = point + o3d.core.Tensor([-step_x, -step_y, step_z], dtype=o3d.core.Dtype.Float32)
    point_7 = point + o3d.core.Tensor([-step_x, -step_y, -step_z], dtype=o3d.core.Dtype.Float32)
    point_8 = point + o3d.core.Tensor([step_x, -step_y, -step_z], dtype=o3d.core.Dtype.Float32)

    # Convert Open3D tensors to NumPy arrays for creating LineSet
    points_np = np.array([
        point_1.numpy().flatten(),
        point_2.numpy().flatten(),
        point_3.numpy().flatten(),
        point_4.numpy().flatten(),
        point_5.numpy().flatten(),
        point_6.numpy().flatten(),
        point_7.numpy().flatten(),
        point_8.numpy().flatten(),
    ])

    # Define lines for LineSet
    lines = [
        [0, 1], [0, 2], [1, 3], [2, 3],
        [4, 5], [4, 6], [5, 7], [6, 7],
        [0, 4], [1, 5], [2, 6], [3, 7],
    ]

    # Create the LineSet
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points_np),
        lines=o3d.utility.Vector2iVector(lines)
    )
    return line_set



def normalize(points):
    # Compute the bounding box
    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)
    center = (min_bound + max_bound) / 2
    scale = max(max_bound - min_bound)
    size = max_bound - min_bound
    pointcloud = (points - center) / scale

    return pointcloud