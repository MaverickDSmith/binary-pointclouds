import os
import numpy as np
import open3d as o3d
from utils import normalize
from tqdm import tqdm


def sample_and_normalize_mesh(mesh_file, num_points):
    """
    Uniformly samples points from a mesh and normalizes them into a unit sphere.
    """
    # Load the mesh
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    
    # Check if the mesh has triangles
    if mesh.triangles is None or len(mesh.triangles) == 0:
        raise ValueError(f"{mesh_file} does not contain any triangles and is not a valid triangle mesh.")

    # Compute uniform sampling
    sampled_pcd = mesh.sample_points_uniformly(number_of_points=num_points)

    # Convert to numpy array
    points = np.asarray(sampled_pcd.points)
    print(np.shape(points))

    points_normalized = normalize(points)
    min_bound = np.min(points_normalized, axis=0)
    max_bound = np.max(points_normalized, axis=0)

    voxel_size = (max_bound - min_bound)
    if voxel_size[0] == 0 or voxel_size[1] == 0 or voxel_size[2] == 0:
        return None

    return points_normalized


def process_dataset(root_dir, output_dir, num_points=1024, split="train", log_file=None):
    """
    Processes a dataset of meshes to generate point clouds.

    Args:
        root_dir (str): Root directory of the dataset (e.g., ModelNet40 structure).
        output_dir (str): Directory to save the generated point clouds.
        num_points (int): Number of points to sample per mesh.
        split (str): Dataset split to process ('train' or 'test').
    """
    for class_name in os.listdir(root_dir):
        class_dir = os.path.join(root_dir, class_name, split)
        if not os.path.isdir(class_dir):
            continue

        output_class_dir = os.path.join(output_dir, class_name, split)
        os.makedirs(output_class_dir, exist_ok=True)

        # Get list of mesh files
        mesh_files = [f for f in os.listdir(class_dir) if f.endswith(('.obj', '.off', '.ply'))]

        # Initialize progress bar
        with tqdm(total=len(mesh_files), desc=f"Processing class: {class_name}", unit="file") as pbar:
            for mesh_file in mesh_files:
                mesh_path = os.path.join(class_dir, mesh_file)
                try:
                    points = sample_and_normalize_mesh(mesh_path, num_points)
                    if points is None:
                        with open(log_file, "a") as f:
                            f.write(f"Failed to process {mesh_path}: Invalid mesh\n")
                        continue
                    output_file = os.path.join(output_class_dir, mesh_file.replace('.obj', '.npy')
                                               .replace('.off', '.npy')
                                               .replace('.ply', '.npy'))
                    np.save(output_file, points)
                except Exception as e:
                    tqdm.write(f"Failed to process {mesh_path}: {e}")

                # Update progress bar with object name
                pbar.set_postfix_str(f"Object: {mesh_file}")
                pbar.update(1)


if __name__ == "__main__":
    # Update these paths
    input_root_dir = "/home/hi5lab/pointcloud_data/ModelNet40"
    output_root_dir = "/home/hi5lab/pointcloud_data/ModelNet40_Pointclouds_2048_test"
    log_file = "pointcloud_sampler.log"
    num_points_to_sample = 2048

    for dataset_split in ["train", "test"]:
        process_dataset(input_root_dir, output_root_dir, num_points_to_sample, split=dataset_split, log_file=log_file)
