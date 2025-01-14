import numpy as np
import bitarray as bitarray
import random
import torch

## My original approach
# def reordering_bitarray(input_array, min_bound, max_bound):
#        x = 0
#        y = 0
#        z = 0
#        i = 0
#        P = 4225
#        N = 65
#        new_array = np.zeros_like(input_array)

#        for z in range(65):
#               for x in range(65):
#                      for y in range(65):
#                             index = x + (y * 65) + (z * P)
#                             new_array[i] = input_array[index]
#                             # print(new_array[i])
#                             i = i + 1
#        transpose_axes = (1, 0, 2)
#        new_min_bound = [min_bound[i] for i in transpose_axes]
#        new_max_bound = [max_bound[i] for i in transpose_axes]

#        return new_array, np.asarray(new_min_bound), np.asarray(new_max_bound)
                            

## Chat-GPT's Vectorized approach (with my tinkering since it couldn't properly figure it out)
# def reordering_bitarray(input_array, size):
#        transpose_axes = tuple(random.sample([0, 1, 2], 3))
#        if (transpose_axes == (0, 1, 2)):
#               return input_array
#        N = size # Grid size, 65 in your case
#        reshaped = np.zeros_like(input_array)
    
#        # Iterate over original axes
#        for i in range(len(input_array)):
#               # Convert 1D index to 3D coordinates
#               z = i // (N * N)
#               y = (i // N) % N
#               x = i % N
        
#               # New indices after transposing
#               new_coords = [x, y, z]
#               new_coords = [new_coords[axis] for axis in transpose_axes]
              
#               # Convert new 3D coordinates back to 1D index
#               new_index = new_coords[0] + (new_coords[1] * N) + (new_coords[2] * N * N)
#               reshaped[new_index] = input_array[i]
       

#        return reshaped

def reordering_bitarray(input_tensor, size):
    # Ensure input is a 1D tensor with N^3 elements
    N = size
    if input_tensor.numel() != N * N * N:
        raise ValueError(f"Expected {N**3} elements, but got {input_tensor.numel()}")

    # Reshape to 3D tensor
    input_3d = input_tensor.view(N, N, N)

    # Randomly permute axes
    transpose_axes = torch.randperm(3).tolist()  # Generates a random permutation of [0, 1, 2]
    reordered_3d = input_3d.permute(*transpose_axes)  # Rearranges axes based on the permutation

    # Flatten back to 1D
    reshaped = reordered_3d.flatten()

    return reshaped


# Version with distances for visualization
# def reordering_bitarray(input_array, size, transpose_axes, min_bound, max_bound):
#        # transpose_axes = tuple(random.sample([0, 1, 2], 3))
#        if (transpose_axes == (0, 1, 2)):
#               return input_array, min_bound, max_bound
#        N = size # Grid size, 65 in your case
#        reshaped = np.zeros_like(input_array)
    
#        # Iterate over original axes
#        for i in range(len(input_array)):
#               # Convert 1D index to 3D coordinates
#               z = i // (N * N)
#               y = (i // N) % N
#               x = i % N
        
#               # New indices after transposing
#               new_coords = [x, y, z]
#               new_coords = [new_coords[axis] for axis in transpose_axes]
              
#               # Convert new 3D coordinates back to 1D index
#               new_index = new_coords[0] + (new_coords[1] * N) + (new_coords[2] * N * N)
#               reshaped[new_index] = input_array[i]
       
#        # Adjust bounds by swapping axes accordingly
#        new_min_bound = np.array(min_bound)[list(transpose_axes)]
#        new_max_bound = np.array(max_bound)[list(transpose_axes)]

#        return reshaped, new_min_bound, new_max_bound