# binary-pointclouds

## Abstract
3D Deep Learning techniques suffer from a variety of computational restraints, primarily resulting from the representation of the data necessary to train a robust model. In order to train models efficiently, methods must be implemented to ensure data does not bloat the model, while maintaining a significant structural similarity to the target after manipulating the data. Existing methods utilize down-sampling techniques to reduce 3D model sizes, but still prove to be too large even after significant compression. Our approach leverages inherent properties in point clouds to maximize the compression of point clouds in a truly binary method, maintains significant structural similarity, and has potential to increase training speeds during deep learning tasks.

## Introduction
All documentation here is a work-in-progress, as this GitHub is a work-in-progress.

This GitHub repo intends to find a method of efficiently encoding and decoding a 3D Point Cloud into a binary format in order to achieve higher levels of compression and faster speeds for reading and writing. Existing techniques are already quite effective at doing this, but to my awareness none of them currently leverage my favorite theory of probability - everything is 50/50 always. It either happens or it doesn't. In our case, there's either a point there or there isn't. Utilizing this absolutely true and highly scientific theory, we intend on fitting a point cloud into a position that can leverage this coin flip property for a hypothetically more efficient way of storing, reading, and writing of point clouds.

## Approach
Our driving theory follows a 6 step approach:

1.) Normalize
  * We normalize the initial point cloud to a unit cube.

2.) Define Bounding Box
  * Since the point cloud is centered at an origin, we find the furthest points from (0, 0, 0) in all three axis. We then create a bounding box that encompasses the entirety of the point cloud.

3.) Subdivide
  * As the box is rectangular, we can essentially "cubify" the box evenly. Choosing a higher number of slices will increase the resolution of the point cloud, but also increases the total run-time and storage. 

4.) Fit
  * We now run a fitting algorithm on the point cloud. Each point is set to the nearest intersection, so long as it is within a pre-defined distance from any intersection. If there is already a point in the closest intersection or if the point is not near enough to any intersection, then the point is ignored.

5.) Binary
  * At every intersection, determine if a point has been moved there. If there is a point, then we set a 1 for that position in the final data structure for the binary point cloud. If there is not a point, that index gets a 0.

6.) Encode
  * We now create a bit-array, encoded with only 26 bytes of necessary header information, as well as encoding the bit array with a Run-Length Encoding algorithm.


With higher number of slices, the voxelization technique outperforms the binary technique at every metric with the tests we've performed so far. The binary technique's highest and best stable number of slices is 64 for each axis.

I am now currently working on ways to further condense the binary vectors, and am beginning to work on a method to implement the binary vector directly in Machine Learning pipelines.

Here's a rough graphic that sort of shows a bit of what we're trying to do:

![Voxel techniques look inside the cube and makes the cube a 1, whereas our method looks at an intersection and places a 1 in that position if it is within a close enough distance](assets/example_bin_vs_vox.png)

### Implementation One
Our initial method (implementation_one.py) does not strictly follow this theory, but serves as a decent starting example to show that this could work the way we intend. We are currently testing with objects for ModelNet40. We've found with our initial implementation that at lower slices we achieve a smaller point cloud in terms of storage space than the current Open3D Voxelization method with the same number of points. The speed with these lower slices is comparable to the voxelization method. The visual appearance of the point cloud is of similar quality, albeit the voxelization technique could be fine-tuned to do better with the same number of points in terms of visual quality. However, as we are focused on eventually sending these point clouds to perform Deep Learning tasks, the initial appearance of the quality of both point clouds intuitively seem as if they would perform well. We will perform qualatative tests in the future.

### Implementation Two
Implementation Two follows the approach from above. For each intersection in the bounding box, we determine if a point is within a certain threshold. If so, that intersection is marked with a 1, and if not, the intersection is marked with a 0. We then store this binary vector as a bitarray, which has a compression effect of (number of slices^3) / 8. With 64 slices per axis, we will always achieve a storage size of 32KB. I am looking into methods to further compress this. 

We determine if a point is within a radius by using a pre-determined K-D Tree Algorithm provided by Open3D. This significantly cuts down on computation for each intersection - with 64 slices per axis, we can turn a point cloud into a binary encoding in under a second. With 128 slices per axis, it only takes three seconds.

We also find that with our current test target of the sofa, assuming the point clouds have the same number of points, with 128 slices per axis the binary method stores slightly more compact than the voxelization method. However, the binary method appears to be more cluttered than the voxelization method, although this may not necessarily be an issue as it is a closer representation of the input point cloud's appearance than if it were more spaced out.

### Implementation Three
We utilize the exact same process from Implementation Two, but add an aditional encoding step. We perform RLE on the bitarray, allowing us to compress the sequential runs of 1s and 0s into even smaller representations. This significantly helps reduce the bloat from the empty space of the point cloud. I go into further detail in the Experiments section "RLE Test." This is our current best method of compressing data. 

## Experiments

A full description of the Experiments ran can be found [here.](assets/docs/experiments.md)

## Results

For further information on the results, please refer to the [extended results page.](assets/docs/results.md)


### Storage Test

In these tests, we compare our technique against other downsampling techniques to see which one stores data more compact. More information the experiments can be found in the Experiments section.

#### Current method's results

| Dataset  | Total Size (KB) | Average Size (KB) | Objects larger than binary | Objects smaller than binary | Objects equal to binary |
|----------|-----------------|-------------------|----------------------------|----------------------------|-------------------------|
| Slice64  |     56602KB     |       4.6KB       |           N/A              |             N/A            |           N/A           |
| Voxel64  |    280371KB     |        22KB       |          12003             |             308            |            0            |
| Slice128 |    144555KB     |      11.7KB       |           N/A              |             N/A            |           N/A           |
| Voxel128 |    400118KB     |        32KB       |          11481             |             828            |            2            |

This table shows the results of our current method. The results show that our method is generally superior to Voxelization, and appears to only be outperformed when point clouds are so small that there is no real reason to down-sample the data.


### Similarity Tests

In these tests, we use different point cloud similarity metrics to determine how similar the down-sampled point cloud is to the ground truth point cloud.

#### Modified Hausdorff Distance

| Dataset  |    Overall Average    |
|----------|-----------------------|
| Slice64  | 0.00616841119289346   |
| Voxel64  | 0.0017316045592079908 |
| Slice128 | 0.003651373099715794  |
| Voxel128 | 0.001071547890892029  |

We found that for each category, Voxelization performed better than our technique in the similarity test. This is to be expected, especially for the low resolution we are using to obtain our point clouds. However, we find that even though Voxelization performs better, our technique still consistently achieves a less than 1% difference score for the majority of the point clouds in each data set. We find that in data sets with extremely small numbers of points, our technique fails. However, our technique is not addressing these tiny data sets, and is an expected flaw. 

#### Chamfer's Distance

| Dataset  |    Overall Average    |
|----------|-----------------------|
| Slice64  | 0.006782016374157892  |
| Voxel64  | 0.0029915333896178603 |
| Slice128 | 0.004195367120516455  |
| Voxel128 | 0.0017880446132857424 |

We found similar results to the Modified Hausdorff Distance test.

## References