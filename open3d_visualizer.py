import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import os
from bitarray import bitarray
import numpy as np

from binary_encoder import sc_decode_variable_length_with_bounds, decode_binary

# WIP
# Intended to be a GUI for visualizing point clouds
# Customizable to what we're concerned with
# Should also be able to run structural similarity tests among other things we're looking at

def list_point_clouds(directory):
    """Returns a list of point cloud files in the given directory."""
    return [f for f in os.listdir(directory) if f.endswith('.ply') or f.endswith('.pcd') or f.endswith('.bin')]

class Open3DVisualizer:
    def __init__(self, root_dir):
        self.root_dir = os.path.expanduser(root_dir)
        self.current_dir = self.root_dir
        self.loaded_clouds = {}
        

        # Create Open3D GUI application
        self.app = gui.Application.instance
        self.app.initialize()
        
        # Create a window
        self.window = self.app.create_window("Open3D Point Cloud Viewer", 1024, 768)
        
        # # Main layout
        self.layout = gui.Horiz()
        self.layout.tooltip = "This is Horiz"
        # self.layout.preferred_height = 20
        
        # # Sidebar panel
        self.panel = gui.Vert()
        # self.panel.preferred_width = 200  # Fixed width for the sidebar
        self.panel.tooltip = "This is Vert"
        
        # Scrollable container for sidebar
        # self.scroll_panel = gui.ScrollableVert()

        
        # Listbox for directory navigation
        self.listbox = gui.ListView()
        self.listbox.set_items(self.get_directory_items(self.current_dir))
        self.listbox.set_max_visible_items(40)
        self.listbox.set_on_selection_changed(self.on_item_selected)
        
        self.panel.add_child(gui.Label("Select a point cloud:"))
        self.panel.add_child(self.listbox)
        
        # self.panel.add_child(self.scroll_panel)
        
        # Scene widget for point cloud visualization
        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(self.window.renderer)
        self.scene_widget.frame = gui.Rect(200, self.window.content_rect.y, 900, self.window.content_rect.height)
        self.scene_widget.tooltip = "This is Scene"
        self.scene_widget.visible = True
        # self.scene_widget.set_on_mouse(self.on_scene_click)
        
        # # Add widgets to the layout
        self.layout.add_child(self.panel)
        # self.layout.add_child(self.scene_widget)
        
        # Add layout to the window
        self.window.add_child(self.layout)
        self.window.add_child(self.scene_widget)
        # self.window.add_child(self.scene_widget)

        # Load default point cloud if available
        self.load_default_point_cloud()
        
    # def on_scene_click(self, event):
    #     """Prevents unintended GUI behavior when clicking in the viewer."""
    #     return gui.Widget.EventCallbackResult.HANDLED

    def get_directory_items(self, directory):
        """Returns a list of directories and point cloud files in the current directory."""

        items = [".. (Back)"] if directory != self.root_dir else []
        items.extend(sorted(d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))))
        items.extend(list_point_clouds(directory))
        return items

        
    def on_item_selected(self, listbox, _):
        """Handles item selection from the sidebar."""
        index = self.listbox.selected_index
        if index < 0:
            return  # Ignore invalid selections

        items = self.get_directory_items(self.current_dir)
        if index >= len(items):
            return

        selected_item = items[index]
        selected_path = os.path.join(self.current_dir, selected_item)

        if selected_item == ".. (Back)":
            new_dir = os.path.dirname(self.current_dir)
        elif os.path.isdir(selected_path):
            new_dir = selected_path
        else:
            new_dir = None  # Not a directory, attempt to load instead

        if new_dir and new_dir != self.current_dir:
            self.current_dir = new_dir
            self.listbox.set_items(self.get_directory_items(self.current_dir))  # Update listbox safely
        elif selected_item.endswith(('.ply', '.pcd', 'bin')):
            self.load_point_cloud(selected_path)




    def load_point_cloud(self, file_path):
        """Loads and displays a point cloud in the visualizer."""
        if file_path.endswith('bin'):
            ba_sc_test = bitarray()
            with open(file_path, 'rb') as f:
                ba_sc_test = f.read()

            ba_sc_test, min_bound, max_bound = sc_decode_variable_length_with_bounds(ba_sc_test)
            numpy_array_loaded = np.array(ba_sc_test.tolist(), dtype=np.uint8)
            size = max_bound - min_bound

            # Decode the binary array
            grid_points_test_sc = decode_binary(numpy_array_loaded, 128, size, min_bound)

            # Create a reconstructed point cloud from the grid points
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(grid_points_test_sc)
        else:
            pcd = o3d.io.read_point_cloud(file_path)
        entity_id = self.scene_widget.scene.add_geometry(file_path, pcd, rendering.MaterialRecord())
        self.loaded_clouds[file_path] = entity_id
        

    def load_default_point_cloud(self):
        """Attempts to load the first available point cloud in the root directory."""
        self.load_point_cloud("/home/hi5lab/pointcloud_data/uniformsampled_ModelNet40_Dataset_2048_128_slices/slice_128_voxel_sc/bed/train/bed_0056_voxel_sc.bin")
        print(f"Loaded default point cloud: /home/hi5lab/pointcloud_data/uniformsampled_ModelNet40_Dataset_2048_128_slices/slice_128_voxel_sc/bed/train/bed_0056_voxel_sc.bin")


    def run(self):
        self.app.run()
        
if __name__ == "__main__":
    visualizer = Open3DVisualizer("/home/hi5lab/pointcloud_data/uniformsampled_ModelNet40_Dataset_2048_128_slices/slice_128_voxel_sc")
    visualizer.run()
