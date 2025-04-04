import os
from tqdm import tqdm

def get_directory_stats(directory_path, slice64_path=None, slice128_path=None):
    total_size_bytes = 0
    file_count = 0
    larger_than_threshold = 0
    smaller_than_threshold = 0
    equal_to_threshold = 0
    class_sizes = {}
    
    for class_name in os.listdir(directory_path):
        class_dir = os.path.join(directory_path, class_name)
        if not os.path.isdir(class_dir):
            continue
        
        class_total_size = 0
        class_file_count = 0
        for root, _, files in os.walk(class_dir):
            with tqdm(total=len(files), desc=f"Processing {class_name}") as pbar:
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        file_size_bytes = os.path.getsize(file_path)
                        total_size_bytes += file_size_bytes
                        class_total_size += file_size_bytes
                        file_count += 1
                        class_file_count += 1

                        file_base = file.replace("_voxel_rle.bin", "").replace("_slice64.bin", "").replace("_slice128.bin", "").replace("_voxel_sc.bin", "")
                        # print(f"File Path: {file_path},           {file_base}")
                        if 'slice_64' in directory_path and slice64_path:
                            slice_file = f"{file_base}_slice64.bin"
                            slice_path = os.path.join(slice64_path, class_name, 'train', slice_file)
                            if not os.path.exists(slice_path):
                                slice_path = os.path.join(slice64_path, class_name, 'test', slice_file)
                                if not os.path.exists(slice_path):
                                    slice_path = os.path.join(slice64_path, class_name, 'val', slice_file)
                        elif 'slice_128' in directory_path and slice128_path:
                            slice_file = f"{file_base}_slice128.bin"
                            slice_path = os.path.join(slice128_path, class_name, slice_file)
                        else:
                            slice_path = None

                        if slice_path and os.path.exists(slice_path):
                            slice_size_bytes = os.path.getsize(slice_path)
                            if file_size_bytes > slice_size_bytes:
                                larger_than_threshold += 1
                            elif file_size_bytes < slice_size_bytes:
                                smaller_than_threshold += 1
                            else:
                                equal_to_threshold += 1
                        # else:
                        #     print(f"Missing corresponding file: {slice_path}")
                    
                    except FileNotFoundError:
                        print(f"File not found: {file_path}")
                    except PermissionError:
                        print(f"Permission denied: {file_path}")
                    except OSError as e:
                        print(f"Error processing file {file_path}: {e}")
                    pbar.update(1)
        
        class_sizes[class_name] = {'total_size_kb': class_total_size / 1024, 'file_count': class_file_count}
    
    average_size_kb = (total_size_bytes / 1024) / file_count if file_count > 0 else 0
    
    return {
        'total_size_kb': total_size_bytes / 1024,
        'average_size_kb': average_size_kb,
        'file_count': file_count,
        'larger_than_threshold': larger_than_threshold,
        'smaller_than_threshold': smaller_than_threshold,
        'equal_to_threshold': equal_to_threshold,
        'class_sizes': class_sizes
    }

slice64_output_dir = "/home/hi5lab/pointcloud_data/storage_test_two/slice64"
slice128_output_dir = "/home/hi5lab/pointcloud_data/storage_test_two/slice128"
voxel64_output_dir = "/home/hi5lab/pointcloud_data/ModelNet40_binary_voxel/slice_64_voxel_sc"
voxel128_output_dir = "/home/hi5lab/pointcloud_data/ModelNet40_binary_voxel_128/slice_128_voxel_sc"
final_log_path = "final_metrics_four_sc.txt"

slice64_stats = get_directory_stats(slice64_output_dir)
slice128_stats = get_directory_stats(slice128_output_dir)
voxel64_stats = get_directory_stats(voxel64_output_dir, slice64_output_dir, slice128_output_dir)
voxel128_stats = get_directory_stats(voxel128_output_dir, slice64_output_dir, slice128_output_dir)

with open(final_log_path, 'w') as log_file:
    log_file.write("Size Stats\n=================\n\n")

    for label, stats in zip(['Slice 64', 'Slice 128', 'Voxel 64', 'Voxel 128'],
                            [slice64_stats, slice128_stats, voxel64_stats, voxel128_stats]):
        log_file.write(f"{label} Total Size: {stats['total_size_kb']:.2f} KB\n")
        log_file.write(f"{label} Average Size: {stats['average_size_kb']:.2f} KB\n")
        log_file.write(f"{label} File Count: {stats['file_count']}\n\n")
        
    log_file.write("Comparison Stats\n=================\n\n")
    log_file.write(f"Voxel 64 files larger than Slice 64: {voxel64_stats['larger_than_threshold']}\n")
    log_file.write(f"Voxel 64 files smaller than Slice 64: {voxel64_stats['smaller_than_threshold']}\n")
    log_file.write(f"Voxel 64 files equal to Slice 64: {voxel64_stats['equal_to_threshold']}\n\n")

    log_file.write(f"Voxel 128 files larger than Slice 128: {voxel128_stats['larger_than_threshold']}\n")
    log_file.write(f"Voxel 128 files smaller than Slice 128: {voxel128_stats['smaller_than_threshold']}\n")
    log_file.write(f"Voxel 128 files equal to Slice 128: {voxel128_stats['equal_to_threshold']}\n\n")

    log_file.write("Per-Class Stats\n=================\n\n")
    for dataset_name, stats in [('Voxel 64', voxel64_stats), ('Voxel 128', voxel128_stats), ('Slice 64', slice64_stats), ('Slice 128', slice128_stats)]:
        log_file.write(f"{dataset_name} Per-Class Breakdown:\n")
        for class_name, class_stats in stats['class_sizes'].items():
            log_file.write(f"  {class_name}: {class_stats['total_size_kb']:.2f} KB, {class_stats['file_count']} files\n")
        log_file.write("\n")

print("All Done!\n")