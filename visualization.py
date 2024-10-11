import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from typing import Dict, Tuple, List, Optional, Union

def find_global_limits(tensors: List[torch.Tensor]) -> Tuple[float, float]:
    """Find global min and max values across all tensors."""
    global_min = min(tensor.min().item() for tensor in tensors)
    global_max = max(tensor.max().item() for tensor in tensors)
    return global_min, global_max

def plot_single_image(ax: plt.Axes, img: np.ndarray, title: str, vmin: Optional[float] = None, vmax: Optional[float] = None) -> plt.colorbar:
    """Plot a single image on the given axes."""
    im = ax.imshow(img, cmap='viridis', vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis('off')
    return im

def add_colorbar(fig: plt.Figure, im: plt.AxesImage, label: str, position: Optional[List[float]] = None) -> None:
    """Add a colorbar to the figure."""
    if position is None:
        position = [1.05, 0.15, 0.05, 0.7]
    cbar_ax = fig.add_axes(position)
    fig.colorbar(im, cax=cbar_ax, label=label)

def plot_timestep_comparison(tensor: torch.Tensor, title: str, num_timesteps: Optional[int] = None, 
                             base_figsize: Tuple[int, int] = (4, 4), vmin: Optional[float] = None, 
                             vmax: Optional[float] = None) -> None:
    """Plot a single tensor's timesteps and their differences."""
    if num_timesteps is None:
        num_timesteps = tensor.shape[-1]
    
    n_diff_plots = num_timesteps - 1
    figsize = (base_figsize[0] * num_timesteps, base_figsize[1] * 2.5)

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.3)

    gs_top = gridspec.GridSpecFromSubplotSpec(1, num_timesteps, subplot_spec=gs[0])
    gs_bottom = gridspec.GridSpecFromSubplotSpec(1, n_diff_plots, subplot_spec=gs[1])

    axes_top = [fig.add_subplot(gs_top[0, i]) for i in range(num_timesteps)]
    axes_bottom = [fig.add_subplot(gs_bottom[0, i]) for i in range(n_diff_plots)]

    fig.suptitle(f"{title} Timesteps and Differences", fontsize=16, y=1.02)

    # Plot timesteps
    for i, ax in enumerate(axes_top):
        img = tensor[0, 0, ..., i].cpu().numpy()
        im = plot_single_image(ax, img, f'T{i}', vmin, vmax)

    add_colorbar(fig, im, f"{title} Values")

    # Calculate and plot differences
    diff_imgs = [tensor[0, 0, ..., i+1] - tensor[0, 0, ..., i] for i in range(n_diff_plots)]
    diff_imgs_np = [diff.cpu().numpy() for diff in diff_imgs]
    diff_vmin, diff_vmax = find_global_limits(diff_imgs)

    for i, (ax, diff_np) in enumerate(zip(axes_bottom, diff_imgs_np)):
        im_diff = plot_single_image(ax, diff_np, f'T{i+1} - T{i}', diff_vmin, diff_vmax)

    add_colorbar(fig, im_diff, f"{title} Differences")

    plt.show()

def plot_model_comparison(tensors: List[Tuple[torch.Tensor, str]], num_timesteps: Optional[int] = None, 
                          base_figsize: Tuple[int, int] = (4, 4)) -> None:
    """Plot a comparison of model inputs, ground truth, and predictions."""
    if num_timesteps is None:
        num_timesteps = tensors[0][0].shape[-1]

    tensor_data = [t[0] for t in tensors]
    vmin, vmax = find_global_limits(tensor_data)

    figsize = (base_figsize[0] * num_timesteps, base_figsize[1] * len(tensors))
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(len(tensors), 1, height_ratios=[1] * len(tensors), hspace=0.4)

    for row, (tensor, title) in enumerate(tensors):
        gs_row = gridspec.GridSpecFromSubplotSpec(1, num_timesteps, subplot_spec=gs[row])
        axes = [fig.add_subplot(gs_row[0, i]) for i in range(num_timesteps)]

        for i, ax in enumerate(axes):
            img = tensor[0, 0, ..., i].cpu().numpy()
            im = plot_single_image(ax, img, f'{title} T{i}', vmin, vmax)

        add_colorbar(fig, im, title, [1.01, 1 - (row + 1) * (1 / len(tensors)), 0.02, 0.8 / len(tensors)])

    fig.suptitle("Model Inputs, Ground Truth, and Predictions", fontsize=16, y=1.02)
    plt.show()

def plot_prediction_error(y: torch.Tensor, out: torch.Tensor, num_timesteps: Optional[int] = None, 
                          base_figsize: Tuple[int, int] = (4, 4)) -> None:
    """Plot the prediction error over time."""
    if num_timesteps is None:
        num_timesteps = y.shape[-1]

    error_vmin, error_vmax = find_global_limits([(y - out)])

    figsize = (base_figsize[0] * num_timesteps, base_figsize[1] * 1.5)
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, num_timesteps)
    axes = [fig.add_subplot(gs[0, i]) for i in range(num_timesteps)]

    for i, ax in enumerate(axes):
        error = (y[0, 0, ..., i] - out[0, 0, ..., i]).cpu().numpy()
        im = plot_single_image(ax, error, f'Error T{i}', error_vmin, error_vmax)

    add_colorbar(fig, im, 'Prediction Error')

    fig.suptitle("Prediction Error Over Time", fontsize=16, y=1.02)
    plt.show()

def visualize_model_predictions(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    sample_idx: int = 512,
    num_timesteps: Optional[int] = None,
    base_figsize: Tuple[int, int] = (4, 4)
) -> None:
    """
    Visualize model predictions and timestep differences with consistent color scaling.

    Args:
        model: The trained model
        test_loader: DataLoader containing test samples
        device: Device to run the model on
        sample_idx: Index of the sample to visualize
        num_timesteps: Number of timesteps to visualize (default: None, uses all available timesteps)
        base_figsize: Base size for a single subplot
    """
    model.eval()

    # Get data sample
    try:
        batch = next(iter(test_loader))
    except StopIteration:
        print("Error: Test loader is empty")
        return

    # Move data to device and get model prediction
    x = batch['x'].to(device=device)
    y = batch['y'].to(device=device)
    with torch.no_grad():
        out = model(x)

    if num_timesteps is None:
        num_timesteps = x.shape[-1]

    # Main comparison plot
    tensors = [(x, 'Input'), (y, 'Ground Truth'), (out, 'Prediction')]
    plot_model_comparison(tensors, num_timesteps, base_figsize)

    # Individual timestep comparisons
    for tensor, title in tensors:
        plot_timestep_comparison(tensor, title, num_timesteps, base_figsize)

    # Prediction error plot
    plot_prediction_error(y, out, num_timesteps, base_figsize)