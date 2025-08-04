import torch


def build_dataset(
    history_length: int,
    num_samples: int,
    num_timestamps: int,
    dimensions: int,
    dtype=torch.float64,
    device=None,
):
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
         base_weights + (torch.rand(history_length, dtype=dtype, device=device) -1/2)
        )

    for t in range(history_length, num_timestamps):
        for i in range(num_samples):
            data[i, t, :] += base_weights @ data[i, t - history_length : t, :]
            data[:, t, :] += torch.randn((num_samples, dimensions), dtype=dtype, device=device)/4

    y = base_weights @ data[:, -history_length:, :]

    return data, y
