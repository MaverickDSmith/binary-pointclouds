# Experiments and Validation Tests

Note: Bathtub_0141 does not have triangles, and throws a warning during processing.

## Compression Tests

These tests are storage size comparisons against Open3D's Voxelization technique. The voxelized downsample threshold was set to 0.001, and whenever there were more points in the voxelized point cloud compared to my technique, I added a density aware downsampling technique to further downsample until there were an equal number of points between the two. If the voxelization technique had less points than the binary encoding technique, I would leave it as is.

### Base Test

For this test, we converted the entirety of ModelNet40 into four separate downsampled point cloud versions. Two of these datasets were made with the technique in Implementation Two, and the other two datasets were made using Open3D's Voxelization technique. The following table shows the results in the Size Comparison test, with notes below:

| Dataset  | Total Size (KB) | Average Size (KB) | Objects larger than binary | Objects smaller than binary | Objects equal to binary |
|----------|-----------------|-------------------|----------------------------|----------------------------|-------------------------|
| Slice64  |    393952KB     |        32KB       |           N/A              |             N/A            |           N/A           |
| Voxel64  |    272184KB     |        22KB       |          2398              |            9913            |            0            |
| Slice128 |   3151616KB     |       256KB       |           N/A              |             N/A            |           N/A           |
| Voxel128 |    395505KB     |        32KB       |           151              |           12160            |            0            |

The rough average speed for an operation on the data sets were as follows:
**Slice64:** ~0.3-0.4 seconds
**Slice128:** ~2.6-2.8 seconds
**Voxel64:** less than 0.1 seconds
**Voxel128:** less than 0.1 seconds

The total time to create all four Datasets was 10 Hours and 24 Minutes exactly. This ran on a single NVIDIA RTX 4090 and an Intel i9-14900KF.

I rounded the Total Sizes of **Voxel64** and **Voxel128** to the nearest whole number.

In more compact terms, the total sizes for each Dataset is:
* **Slice64:** 384MB
* **Voxel64:** 265MB
* **Slice128:** 3GB
* **Voxel128:** 386MB

**Slice64** and **Slice128** are the datasets created by the binary technique. We differentiate between the two with their relevant number by how many slices there are on each axis. With the current technique, we are storing the data as one long bitarray with no extra compression. This means that each point is equivalent to 1 bit of space, and each point cloud is storing every single possible point in the point cloud as a 1 or a 0. Therefore, each object has the same storage size in their respective data sets. **Slice64** has 32KB for every point cloud, and **Slice128** has 256KB for every point cloud.

**Voxel64** and **Voxel128** are the datasets created using Open3D's Voxelization technique. Some miscallaneous notes about the way this is implemented:
* Each point cloud was downsampled with the open3d.geometry.voxel_down_sample() function, and the voxel size was set to 0.001 for each point cloud. The number is this low because we normalized each point cloud prior to downsampling. 
* To make things fair, I decided to also apply a density aware downsampling technique to the voxelization technique when the overall points of the point cloud after voxelization was higher than the number of 1s in the corresponding binary point cloud. The density aware downsampling ensures that no voxelized point cloud has more points than its binary point cloud counterpart. This technique was not applied to any point clouds with equal or lower number of points than the binary point cloud counterpart. I believe this makes the comparison fair, as I find that density aware downsampling is an easy to implement addition to voxelization when you're trying to perform data preprocessing, and most Deep Learning tasks find their job easier when the number of points is homogenous across the entire dataset. I did not pad points when the total points were lower, as this is ultimately a test of which technique stores in a more compact form.
* The threshold for "larger / smaller / equal to" is based on the storage size of the binary. So for **Voxel64**, the threshold is 32KB, and for **Voxel128**, the threshold is 256KB.

### RLE Tests

For this test, we added an extra step to the encoding process. After encoding the point cloud as a 1D binary bitarray, we take the bitarray and encode it using Run Length Encoding. Essentially, RLE takes a sequence of numbers and converts it to a tuple. So if part of the bitarray had a run of 0s such as "0000000000", RLE would convert that to (10, 0), representing ten 0s in that spot.

To keep everything capable of being represented in bits, we convert the RLE representation into binary as well. In doing so we incur a header cost, but this cost only requires two bytes of data. After we run RLE on the bitarray, our new bitarray is formatted as such:

24 Bytes, 6 sets of 32-bit strings
First 192 bits: Min Bound and Max Bound as float values

    min_bound = (X, Y, Z)
    max_bound = (X, Y, Z)
    6 sets of 32-bit strings
    1 set for each coordinate

Next 16 bits: binary number that represents how long each bit-string is in the rest of the bitarray
Next N bits: N is the number decoded from the 16-bit header. The first N bits after the header is a binary number for how many 0s go in that spot.
Next N bits: The N bits after that header is a binary number for how many 1s go in that spot.
We repeat, switching between 0 and 1 until the bitarray ends.
*acknowledgment to Alex Sommers for the intuitive approach of starting with 0 each time, even if there are no 0s in the first part of the bitarray. this saves 1 bit of data and some runtime every time we switch sequences.

Until I determine if someone else has done something similar, I'm calling this implementation of RLE "Dynamic Binary Run Length Encoding." Dynamic, because the bit-string length is not a static number for each object being stored, and Binary because it's all 1s and 0s. The purpose of determing the bit-string length and requiring the header is so that the implementation can be even further compacted as necessary. In normal point clouds, we're expecting large stretches of 0-space. This number can grow significantly larger than the average length of all the rest of the sequences. We need to make sure that we have enough bits to represent these large numbers, but we don't want to create a static number of bits. This could cause unnecessary space to be taken up, or could cause failure if the size of the bit-string is not large enough to represent these large stretches, which grow even larger for higher amounts of slices. Allowing the bit-size of the whole RLE bitarray to be dynamic enchances the compactness and allows this to be used for future cases where resolution is much higher than what we're testing.

Voxelization is run the same way as before. We compare the direct file counterpart to its binary RLE counterpart to see how many are larger, smaller, or equal to. Further notes on the test are below the table.

#### RLE Test 1

This test was run before fixing some issues with how the points were placed. Originally, the First 16 bits represented how long each bit-string was in the rest of the bitarray, and that was the only information in the header. 

| Dataset  | Total Size (KB) | Average Size (KB) | Objects larger than binary | Objects smaller than binary | Objects equal to binary |
|----------|-----------------|-------------------|----------------------------|----------------------------|-------------------------|
| Slice64  |     53058KB     |       4.3KB       |           N/A              |             N/A            |           N/A           |
| Voxel64  |    272184KB     |        22KB       |          12081             |             229            |            1            |
| Slice128 |    139476KB     |      11.3KB       |           N/A              |             N/A            |           N/A           |
| Voxel128 |    395505KB     |        32KB       |          11631             |             679            |            1            |

I rounded the total size of the datasets to the nearest whole number.

In more compact terms, the total sizes for each Dataset is:
* **Slice64:** 51.8MB
* **Voxel64:** 265MB
* **Slice128:** 136MB
* **Voxel128:** 386MB

The total time to create all four Datasets was roughly 11 hours. For this test I did not re-run the Voxelization datasets as I did not change the method of creating the binary encoding or the voxelization, I only added an extra encoding step after the initial binary encoding. This ran on a single NVIDIA RTX 4090 and an Intel i9-14900KF.

Despite running an extra encoding step, the RLE did not add significant time overhead. 
Average processing for **Slice64:** 0.35 - 0.4 seconds
Average processing for **Slice128:** 2.7 - 2.9 seconds

#### RLE Test 2

This test is on the current best implementation of the encoding process. This test was ran after a fix to the encoding process. Prior to this test, I did not account for the two extra intersection points that come from slicing the bounding box as many times as we did. Originally, we accounted for one extra point by going from slice 0 to slice 64, inclusive of slice 0. However, we still need to get the end edge from these intersections, which means we actually have to iterate slices + 1 times, inclusive of 0. To find how many slices are in the point cloud now, you must take the cube root of the total points, and subtract one.

| Dataset  | Total Size (KB) | Average Size (KB) | Objects larger than binary | Objects smaller than binary | Objects equal to binary |
|----------|-----------------|-------------------|----------------------------|----------------------------|-------------------------|
| Slice64  |     56602KB     |       4.6KB       |           N/A              |             N/A            |           N/A           |
| Voxel64  |    280371KB     |        22KB       |          12003             |             308            |            0            |
| Slice128 |    144555KB     |      11.7KB       |           N/A              |             N/A            |           N/A           |
| Voxel128 |    400118KB     |        32KB       |          11481             |             828            |            2            |

In more compact terms, the total sizes for each Dataset is:
* **Slice64:** 51.8MB
* **Voxel64:** 265MB
* **Slice128:** 136MB
* **Voxel128:** 386MB

The total time to create all four Datasets was roughly 11 hours. For this test I did re-run the Voxelization datasets, as I had fixed some issues I found with how I was slicing the bounding box and determining where intersections were. This ran on a single NVIDIA RTX 4090 and an Intel i9-14900KF.

## Similarity Tests

For these tests, I ran different point cloud similarity tests between the downsampled point clouds and the ground truth. Averages of each category for each data set can be found [here.](/assets/docs/results.md)

### Modified Hausdorff Distance

The test was conducted by comparing the point cloud of the data set with the point cloud of the ground truth data set. Modified Hausdorff Distance compares the nearest point from point cloud A to the nearest point in the same position in point cloud B, then from B to A, and takes the average of these distances for each point cloud.

| Dataset  |    Overall Average    |
|----------|-----------------------|
| Slice64  | 0.00616841119289346   |
| Voxel64  | 0.0017316045592079908 |
| Slice128 | 0.003651373099715794  |
| Voxel128 | 0.001071547890892029  |


### Chamfer's Distance

Chamfer's Distance compares the nearest point from point cloud A to the nearest point in the same position in point cloud B, then from B to A, and takes the sum of the minimum distances for each point cloud, and averages it out between the two.


| Dataset  |    Overall Average    |
|----------|-----------------------|
| Slice64  | 0.006782016374157892  |
| Voxel64  | 0.0029915333896178603 |
| Slice128 | 0.004195367120516455  |
| Voxel128 | 0.0017880446132857424 |