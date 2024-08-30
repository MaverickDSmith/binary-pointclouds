import os
from tqdm import tqdm

def get_directory_stats(directory_path, size_threshold_kb):
    total_size_bytes = 0
    file_count = 0
    larger_than_threshold = 0
    smaller_than_threshold = 0
    equal_to_threshold = 0

    # Traverse through all files in the directory and subdirectories
    for root, _, files in os.walk(directory_path):
        with tqdm(total=len(files), desc="Processing files") as pbar:
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    # Get the size of each file in bytes
                    file_size_bytes = os.path.getsize(file_path)
                    total_size_bytes += file_size_bytes
                    file_count += 1

                    # Convert file size to kilobytes
                    file_size_kb = file_size_bytes / 1024

                    # Count files based on the size threshold
                    if file_size_kb > size_threshold_kb:
                        larger_than_threshold += 1
                    elif file_size_kb < size_threshold_kb:
                        smaller_than_threshold += 1
                    else:
                        equal_to_threshold += 1
                except FileNotFoundError:
                    print(f"File not found: {file_path}")
                except PermissionError:
                    print(f"Permission denied: {file_path}")
                except OSError as e:
                    print(f"Error processing file {file_path}: {e}")
                pbar.update(1)
    print("Done with a dataset\n")


    # Calculate the average file size in kilobytes
    average_size_kb = (total_size_bytes / 1024) / file_count if file_count > 0 else 0

    return {
        'total_size_kb': total_size_bytes / 1024,
        'average_size_kb': average_size_kb,
        'file_count': file_count,
        'larger_than_threshold': larger_than_threshold,
        'smaller_than_threshold': smaller_than_threshold,
        'equal_to_threshold': equal_to_threshold
    }


slice64_output_dir = "/home/hi5lab/github/github_ander/Fall 2024/data/storage_test/slice64"
slice128_output_dir = "/home/hi5lab/github/github_ander/Fall 2024/data/storage_test/slice128"
voxel64_output_dir = "/home/hi5lab/github/github_ander/Fall 2024/data/storage_test/voxel64"
voxel128_output_dir = "/home/hi5lab/github/github_ander/Fall 2024/data/storage_test/voxel128"
final_log_path = "final_metrics.txt"

# get sizes
slice64_stats = get_directory_stats(slice64_output_dir, 32)
slice128_stats = get_directory_stats(slice128_output_dir, 256)
voxel64_stats = get_directory_stats(voxel64_output_dir, 32)
voxel128_stats = get_directory_stats(voxel128_output_dir, 256)

with open(final_log_path, 'w') as log_file:
    # log_file.write("Time Stats\n")
    # log_file.write("=================\n\n")

    # log_file.write(f"\n\nSlice64 processing took {result['slice64']:.2f} seconds to complete.\n")
    # log_file.write(f"Slice128 processing took {result['slice128']:.2f} seconds to complete.\n")
    # log_file.write(f"Voxel 64 took {result['voxel64']:.2f} seconds to complete.\n")
    # log_file.write(f"Voxel 128 took {result['voxel128']:.2f} seconds to complete.\n")
    # log_file.write(f"Total time of all operations: {total_time:.2f}\n")

    log_file.write("Size Stats\n")
    log_file.write("=================\n\n")

    log_file.write(f"Slice 64 Total Size: {slice64_stats['total_size_kb']}\n")
    log_file.write(f"Slice 64 Average Size: {slice64_stats['average_size_kb']}\n")
    log_file.write(f"Slice 64 File Count: {slice64_stats['file_count']}\n\n")

    log_file.write(f"Slice 128 Total Size: {slice128_stats['total_size_kb']}\n")
    log_file.write(f"Slice 128 Average Size: {slice128_stats['average_size_kb']}\n")
    log_file.write(f"Slice 128 File Count: {slice128_stats['file_count']}\n\n")

    log_file.write(f"Voxel 64 Total Size: {voxel64_stats['total_size_kb']}\n")
    log_file.write(f"Voxel 64 Average Size: {voxel64_stats['average_size_kb']}\n")
    log_file.write(f"Voxel 64 File Count: {voxel64_stats['file_count']}\n")
    log_file.write(f"Voxel 64 files larger than 32KB: {voxel64_stats['larger_than_threshold']}\n")
    log_file.write(f"Voxel 64 files smaller than 32KB: {voxel64_stats['smaller_than_threshold']}\n")
    log_file.write(f"Voxel 64 files equal to 32KB: {voxel64_stats['equal_to_threshold']}\n\n")

    log_file.write(f"Voxel 128 Total Size: {voxel128_stats['total_size_kb']}\n")
    log_file.write(f"Voxel 128 Average Size: {voxel128_stats['average_size_kb']}\n")
    log_file.write(f"Voxel 128 File Count: {voxel128_stats['file_count']}\n")
    log_file.write(f"Voxel 128 files larger than 256KB: {voxel128_stats['larger_than_threshold']}\n")
    log_file.write(f"Voxel 128 files smaller than 256KB: {voxel128_stats['smaller_than_threshold']}\n")
    log_file.write(f"Voxel 128 files equal to 256KB: {voxel128_stats['equal_to_threshold']}\n\n")

print("All Done!\n")