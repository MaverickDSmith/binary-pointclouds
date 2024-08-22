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

def normalize(points):
    # Compute the bounding box
    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)
    center = (min_bound + max_bound) / 2
    scale = max(max_bound - min_bound)
    size = max_bound - min_bound
    pointcloud = (points - center) / scale

    return pointcloud, min_bound, max_bound, size