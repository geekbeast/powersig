import cmath
from typing import Tuple
import torch


def build_chebychev_dataset(
    num_samples: int,
    num_timestamps: int,
    dimensions: int,
    dtype=torch.float64,
    device=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build a dataset where each dimension consists of cumulative sums of Chebyshev nodes 
    with dimension-specific offsets (d * timestamp added to the index).
    
    Args:
        num_samples: Number of samples in the dataset
        num_timestamps: Number of timestamps per sample
        dimensions: Number of dimensions per timestamp
        dtype: Data type for the tensor
        device: Device to place the tensor on
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (data, y) where data has shape 
        (num_samples, num_timestamps, dimensions) and y has shape (num_samples, dimensions)
        containing the next cumulative sum for each dimension
    """
    # Create the dataset tensor
    data = torch.zeros(
        num_samples, num_timestamps, dimensions, dtype=dtype, device=device
    )
    
    # Generate base indices once
    indices = torch.arange(num_timestamps, dtype=dtype, device=device)
    
    # Calculate total nodes needed for all dimensions
    total_nodes = dimensions * num_timestamps * num_samples
    total_nodes **= 2

    
    # Generate Chebyshev nodes and their cumulative sums for each dimension with dimension-specific offsets
    for i in range(num_samples):
        for d in range(dimensions):
            # Add both sample and dimension offsets to the Chebyshev index
            offset_indices = indices + i * dimensions * num_timestamps + d * num_timestamps 
            offset_indices **= 2
            chebychev_nodes = torch.cos(torch.pi * (2 * offset_indices + 1) / (2 * total_nodes))
            
            # Calculate cumulative sum for this dimension
            cumulative_sum = torch.cumsum(chebychev_nodes, dim=0)
            
            # Fill this sample for this dimension with cumulative sums
            data[i, :, d] = cumulative_sum
    
    # Generate y tensor with the next cumulative sum for each dimension
    y = torch.zeros((num_samples, dimensions), dtype=dtype, device=device)
    
    for d in range(dimensions):
        for i in range(num_samples):
            # Calculate the next index after the last timestamp for this sample and dimension
            next_index = num_timestamps + i * dimensions * num_timestamps + d * num_timestamps
            next_index **= 2
            next_chebychev_node = cmath.cos(torch.pi * (2 * next_index + 1) / (2 * total_nodes))
            
            # Get the last cumulative sum for this sample and dimension
            last_cumulative_sum = data[i, -1, d]
            
            # The next cumulative sum is the last cumulative sum plus the next Chebyshev node
            y[i, d] = last_cumulative_sum + next_chebychev_node
    
    return data, y


def build_dataset(
    history_length: int,
    num_samples: int,
    num_timestamps: int,
    dimensions: int,
    dtype=torch.float64,
    device=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    data = torch.zeros(
        num_samples, num_timestamps, dimensions, dtype=torch.float64, device=device
    )
    data[:, :history_length, :] = torch.randn(
        [num_samples, history_length, dimensions], dtype=dtype, device=device
    )

    indices = torch.arange(history_length, device=device, dtype=dtype)
    base_weights = (history_length - indices) ** 2
    base_weights = base_weights / base_weights.norm()
    base_weights *= (-1**indices)/2.75

    weights = torch.zeros(
        (num_samples, history_length), dtype=dtype, device=device
    )

    for i in range(num_samples):
        weights[i] =  (
         base_weights + (torch.rand(history_length, dtype=dtype, device=device)-.5)/4
        )

    for t in range(history_length, num_timestamps):
        for i in range(num_samples):
            data[i, t, :] += weights[i] @ data[i, t - history_length : t, :]
            data[:, t, :] += (torch.rand((num_samples, dimensions), dtype=dtype, device=device)-.5)/16

    y= torch.zeros((num_samples, dimensions), dtype=dtype, device=device)

    for i in range(num_samples):
        y[i] = weights[i] @ data[i, t - history_length : t, :]
    y += (torch.rand((num_samples, dimensions), dtype=dtype, device=device)-.5)/16
    # y = base_weights @ data[:, -history_length:, :]

    return data, y
