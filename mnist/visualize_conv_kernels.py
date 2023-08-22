import torch as t
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from datetime import datetime
from glob import glob
import re
import numpy as np

def extract_frame_number(filename):
    # Extracting the frame number from the filename using regex
    match = re.search(r'frame_(\d+\.\d+)\.png$', filename)
    if match:
        return float(match.group(1))
    return None

def hook_fn(module, input, output, activations):
    activations.append(output)

def average_activations_for_class(model, data_loader, target_class, layer_name, handles=None):
    device = next(model.parameters()).device
    activations = []
    handle = dict(model.named_children())[layer_name].register_forward_hook(lambda module, input, output: hook_fn(module, input, output, activations))
    if handles is not None:
        handles.append(handle)
    for inputs, labels in tqdm(data_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        mask = labels == target_class
        if mask.sum() == 0:
            continue
        with t.no_grad():
            model(inputs[mask])
    handle.remove()
    avg_activations = t.cat(activations, dim=0).mean(dim=0)
    return avg_activations.cpu().numpy(), handle

def visualize_all_classes(model, data_loader, layer_name, save_path=None):
    plt.clf()
    unique_labels = list(set([label.item() for _, labels in data_loader for label in labels]))
    # Temporary activation to find the number of kernels
    temp_activations, _ = average_activations_for_class(model, data_loader, unique_labels[0], layer_name)
    num_kernels = temp_activations.shape[0]
    fig, axes = plt.subplots(len(unique_labels), num_kernels, figsize=(num_kernels * 2, len(unique_labels) * 2))
    handles = []
    for idx, class_label in enumerate(unique_labels):
        # The modification here is to capture the handle from the average_activations_for_class function
        avg_activations, handle = average_activations_for_class(model, data_loader, class_label, layer_name, handles)
        for i in range(num_kernels):
            if len(unique_labels) > 1:
                ax = axes[idx, i]
            else:
                ax = axes[i]
            ax.imshow(avg_activations[i], cmap='gray')
            ax.axis('off')
            if idx == 0:
                ax.set_title(f'Kernel {i + 1}')
        if len(unique_labels) > 1:
            axes[idx, 0].set_ylabel(f'Class {class_label}', size='large')
    plt.tight_layout()
    plt.savefig(save_path)
    for handle in handles:
        handle.remove()
    plt.close()

def visualize_weights(model, layer_name, save_path=None):
    plt.clf()
    if layer_name == 'conv1':
        weights = model.conv1[0].weight.detach().cpu().numpy()
    else:
        weights = model.conv2[0].weight.detach().cpu().numpy()

    num_channels = weights.shape[0]
    num_kernels = weights.shape[1]

    fig, axes = plt.subplots(num_channels, num_kernels, figsize=(num_kernels * 2, num_channels * 2))
    
    # If we have only 1 channel or 1 kernel, make axes 2D for consistent indexing
    if num_channels == 1:
        axes = np.expand_dims(axes, axis=0)
    if num_kernels == 1:
        axes = np.expand_dims(axes, axis=1)
        
    for i in range(num_channels):
        for j in range(num_kernels):
            ax = axes[i, j]
            ax.imshow(weights[i, j], cmap='gray')
            ax.axis('off')
            if i == 0:
                ax.set_title(f'Kernel {j + 1}')
        axes[i, 0].set_ylabel(f'Channel {i + 1}', size='large')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()


def run_movie_cmd(framedir, moviename):
    if not os.path.exists("movies"):
        os.mkdir("movies")
    mp4_name = os.path.join('movies', moviename)
    os.system(f'ffmpeg -framerate 3 -i {framedir}/frame_%04d.png -c:v libx264 -pix_fmt yuv420p {mp4_name}')
    print(f'movie saved as {mp4_name}')

def save_frames(step, model, dataloader):
    if not os.path.exists("frames"):
        os.mkdir("frames")
    if not os.path.exists(f"frames/conv1"):
        os.mkdir(f"frames/conv1")
    if not os.path.exists(f"frames/conv2"):
        os.mkdir(f"frames/conv2")
    visualize_all_classes(model, dataloader, 'conv1', f"frames/conv1/frame_{step:04}.png")
    visualize_all_classes(model, dataloader, 'conv2', f"frames/conv2/frame_{step:04}.png")

def save_frames_weights(step, model):
    if not os.path.exists("frames"):
        os.mkdir("frames")
    if not os.path.exists(f"frames/conv1"):
        os.mkdir(f"frames/conv1")
    if not os.path.exists(f"frames/conv2"):
        os.mkdir(f"frames/conv2")
    visualize_weights(model, 'conv1', f"frames/conv1/frame_{step:04}.png")
    visualize_weights(model, 'conv2', f"frames/conv2/frame_{step:04}.png")

def make_conv_movies():
    run_movie_cmd("frames/conv1", f"conv1_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
    run_movie_cmd("frames/conv2", f"conv2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")

if __name__ == "__main__":
    make_conv_movies()