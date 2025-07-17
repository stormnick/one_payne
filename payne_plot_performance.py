from __future__ import annotations

import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import time
import argparse
import sys
import datetime
import math
from sys import argv
from load_train_data_scripts import load_data


def load_model_parameters(data, model):
    """
    Load model parameters and scaling info from a .npz file into the given model.
    
    Parameters
    ----------
    file_path : str
        Path to the .npz file containing the saved model parameters and scaling info.
    model : torch.nn.Sequential
        The model instance with the same architecture as was used during saving.
    
    Returns
    -------
    model : torch.nn.Sequential
        The model with loaded parameters.
    x_min : np.ndarray
        The saved minimum values for input scaling.
    x_max : np.ndarray
        The saved maximum values for input scaling.
    """
    
    with torch.no_grad():
        # Assuming your Linear layers are at indices 0, 2, and 4
        model[0].weight.copy_(torch.from_numpy(data['w_array_0']))
        model[0].bias.copy_(torch.from_numpy(data['b_array_0']))
        
        model[2].weight.copy_(torch.from_numpy(data['w_array_1']))
        model[2].bias.copy_(torch.from_numpy(data['b_array_1']))
        
        model[4].weight.copy_(torch.from_numpy(data['w_array_2']))
        model[4].bias.copy_(torch.from_numpy(data['b_array_2']))

        model[6].weight.copy_(torch.from_numpy(data['w_array_3']))
        model[6].bias.copy_(torch.from_numpy(data['b_array_3']))
    
    return model


def build_model(dim_in, num_pix, hidden_neurons):
    """
    Build a simple feed-forward neural network with two hidden layers.

    Parameters
    ----------
    dim_in : int
        Dimensionality of input features.
    num_pix : int
        Number of output pixels (flux values).
    hidden_neurons : int
        Number of neurons in each hidden layer.

    Returns
    -------
    model : torch.nn.Sequential
        The defined neural network.
    """
    model = torch.nn.Sequential(
        torch.nn.Linear(dim_in, hidden_neurons),
        torch.nn.SiLU(),
        torch.nn.Linear(hidden_neurons, hidden_neurons),
        torch.nn.SiLU(),
        torch.nn.Linear(hidden_neurons, hidden_neurons),
        torch.nn.SiLU(),
        torch.nn.Linear(hidden_neurons, num_pix),
        torch.nn.Sigmoid()  # enforce [0, 1] outputs
    )
    return model

def scale_back(x, x_min, x_max, label_name=None):
    return_value = (x + 0.5) * (x_max - x_min) + x_min
    if label_name == "teff":
        return_value = return_value * 1000
    return return_value

def do_plot(data_file, model_path, train_fraction):
    if '/' in model_path:
        model_path_save = model_path.split('/')[-1].split('.')[0]
    else:
        model_path_save = model_path.split('.')[0]

    # load model
    x, y, x_valid, y_valid, x_min, x_max, num_pix, dim_in, wvl = load_data(data_file, train_fraction=train_fraction)

    data = np.load(model_path)
    x_min = data['x_min']
    x_max = data['x_max']
    hidden_neurons = data['hidden_neurons']
    label_names = data['label_names']

    model = build_model(dim_in, num_pix, hidden_neurons)
    model = load_model_parameters(data, model)

    with torch.no_grad():
        predictions = model(x_valid)

    predictions = predictions.numpy()
    x_valid_np = x_valid.numpy()
    y_valid_np = y_valid.numpy()  # if you want to compare against the actual targets

    # Assume `model` is your pre-trained model and `target` is a 6000-long array (torch tensor)
    target = y_valid  # shape: (1, 6000)
    # Start with a random guess for the 10 labels
    input_est = torch.full((len(y_valid[:, 0]), len(x_valid[0])), 0.0, requires_grad=True)
    print("Estimated labels:", input_est.detach().numpy())
    optimizer = torch.optim.Adam([input_est], lr=0.01)

    for i in range(3000):  # adjust number of iterations as needed
        optimizer.zero_grad()
        pred = model(input_est)
        loss = ((pred - target) ** 2).mean()
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print(f"Iteration {i}, loss: {loss.item()}")
        if i % 500 == 0:
            print("Estimated labels:", input_est.detach().numpy())

    # input_est should now be an approximation of the 10 labels that produce an output close to target
    print("Estimated labels:", input_est.detach().numpy())

    predicted_labels = input_est.detach().numpy()
    actual_labels = x_valid.detach().numpy()

    time_to_save = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    np.savez(f"actual_vs_predicted_labels_{time_to_save}_{model_path_save}.npz", actual_labels=actual_labels, predicted_labels=predicted_labels, label_names=label_names, x_min=x_min, x_max=x_max) 

    num_labels = len(label_names)
    num_cols = 3
    num_rows = math.ceil(num_labels / num_cols)

    # Create a grid of subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(3 * num_cols, 3 * num_rows), squeeze=False)

    for i in range(num_labels):
        row = i // num_cols
        col = i % num_cols
        ax = axes[row, col]

        # Scale back actual and predicted values for label i
        actual_i = scale_back(actual_labels[:, i], x_min[i], x_max[i], label_names[i])
        predicted_i = scale_back(predicted_labels[:, i], x_min[i], x_max[i], label_names[i])

        # Scatter plot: Actual (x-axis) vs. Predicted (y-axis)
        ax.scatter(actual_i, predicted_i, s=0.5, color='black')

        # Compute bias and std
        diff = actual_i - predicted_i
        bias_val = np.mean(diff)
        std_val = np.std(diff)

        ax.set_xlabel(f"Actual {label_names[i]}")
        ax.set_ylabel(f"Predicted {label_names[i]}")
        ax.set_title(f"{label_names[i]}\nBias={bias_val:.2f}, Std={std_val:.2f}")

        # Set plot limits based on actual values
        ax.set_xlim(np.min(actual_i), np.max(actual_i) * 1.05)
        ax.set_ylim(np.min(actual_i), np.max(actual_i) * 1.10)

    # Turn off any unused subplots if num_labels isn't a multiple of num_cols
    for j in range(num_labels, num_rows * num_cols):
        row = j // num_cols
        col = j % num_cols
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig(f"predicted_vs_actual_{time_to_save}_model_{model_path_save}.png")


    num_labels = len(label_names)
    num_cols = 3
    num_rows = math.ceil(num_labels / num_cols)

    # Create a grid of subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(3 * num_cols, 3 * num_rows), squeeze=False)

    # Determine FeH index (if you know it's the third label)
    feh_index = 2

    # Scale back the FeH values just once (applies to all plots)
    feh_vals = scale_back(actual_labels[:, feh_index],
                        x_min[feh_index],
                        x_max[feh_index],
                        label_names[feh_index])
    feh_min, feh_max = feh_vals.min(), feh_vals.max()


    for i in range(num_labels):
        row = i // num_cols
        col = i % num_cols
        ax = axes[row, col]

        # Scale back actual and predicted values for label i
        actual_i = scale_back(actual_labels[:, i],
                            x_min[i],
                            x_max[i],
                            label_names[i])
        predicted_i = scale_back(predicted_labels[:, i],
                                x_min[i],
                                x_max[i],
                                label_names[i])

        # Compute bias and std
        diff = actual_i - predicted_i
        bias_val = np.mean(diff)
        std_val = np.std(diff)

        # Scatter plot: color by FeH
        sc = ax.scatter(
            actual_i, predicted_i,
            s=1,
            c=feh_vals,          # color by FeH
            cmap='coolwarm',
            vmin=feh_min,
            vmax=feh_max
        )
        ax.set_xlabel(f"Actual {label_names[i]}")
        ax.set_ylabel(f"Predicted {label_names[i]}")
        ax.set_title(
            f"{label_names[i]}\nBias={bias_val:.2f}, Std={std_val:.2f}"
        )

        # Set plot limits based on actual values
        ax.set_xlim(actual_i.min(), actual_i.max() * 1.05)
        ax.set_ylim(actual_i.min(), actual_i.max() * 1.10)

        # Add a colorbar for FeH to each subplot (optional).
        # If you want a single colorbar for the entire figure,
        # move it outside the loop.
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label('FeH')

    # Turn off any unused subplots if num_labels isn't a multiple of 3
    for j in range(num_labels, num_rows * num_cols):
        row = j // num_cols
        col = j % num_cols
        axes[row, col].axis('off')

    #plt.title(model_path_save)

    plt.tight_layout()
    plt.savefig(f"predicted_vs_actual_{time_to_save}_model_{model_path_save}_colorby_FeH.png")

