# binary-pointclouds

## Abstract
3D Deep Learning techniques suffer from a variety of computational restraints, primarily resulting from the representation of the data necessary to train a robust model. In order to train models efficiently, methods must be implemented to ensure data does not bloat the model, while maintaining a significant structural similarity to the target after manipulating the data. Existing methods utilize down-sampling techniques to reduce 3D model sizes, but still prove to be too large even after significant compression. Our approach leverages inherent properties in point clouds to maximize the compression of point clouds in a truly binary method, maintains significant structural similarity, and has potential to increase training speeds during deep learning tasks.

## Introduction
All documentation here is a work-in-progress, as this GitHub is a work-in-progress.

This project aims to answer two questions; can we push the state of the art in 3D Point Cloud compression, and can we use this compression to use less resources in 3D Deep Learning tasks? As a result, we break the project into two parts. We are focusing specifically on point cloud objects, as opposed to full 3D scenes or segmentation tasks. We currently have two compression methods, both with their own pros and cons, and we are still actively working on neural network architectures to support our point cloud representations.

## Approach
Our compression methods start the same:

1.) Normalize
  * We normalize the initial point cloud to a unit cube.

2.) Define Bounding Box
  * Since the point cloud is centered at an origin, we find the furthest points from (0, 0, 0) in all three axis. We then create a bounding box that encompasses the entirety of the point cloud.

3.) Subdivide
  * As the box is rectangular, we can essentially "cubify" the box evenly. Choosing a higher number of slices will increase the resolution of the point cloud, but also increases the total run-time of the compression algorithm and file size of the object. This cubified box is now our second point cloud. It is empty, and shaped as a 3D Grid.

4.) Fit
  * We now run a fitting algorithm on the point cloud. Each point is set to the nearest point in the grid cloud, so long as it is within a pre-defined distance of the point. If there is already a point in the closest grid cloud point, or if the input point is not near enough to any grid point, then the point is ignored.

5.) Binary
  * At every grid point, determine if a point has been moved there. If there is a point, then we set a 1 for that position in the final data structure for the binary point cloud. If there is not a point, that index gets a 0.

6.) Encode
  * We now create a bitarray, encoded with only 26 bytes of necessary header information, and further compress the bitarray with either RLE or a sparse encoder.


The most computationally intensive portion is the fitting portion. The higher our slice count is, the longer it takes to find where each point should go. We currently have two methods of fitting points to the grid - vector quantization, and a modified approach to voxelization. Our vector quantization approach looks at each point in the grid cloud and searches through a KDTree of the input cloud to see if there are any points within a certain threshold of the grid cloud. The modified voxelization uses each point in the grid cloud as the center of the voxel, and looks for any points from the input cloud within this voxel to determine occupancy.

## Experiments

A full description of the Experiments ran can be found [here.](assets/docs/experiments.md)

## Results

For further information on the results, please refer to the [extended results page.](assets/docs/results.md)
