from __future__ import annotations

import numpy as np
import torch
from torch.autograd import Variable
from sklearn.model_selection import train_test_split


def load_and_stack(file_paths):
    """
    Load one or more NPZ spectral files and concatenate their spectra.

    Parameters
    ----------
    file_paths : sequence of str or Path
        Paths to .npz files that each contain the keys
        'wvl', 'labels', 'flxn', and 'label_names'.

    Returns
    -------
    wvl : ndarray
        Shared wavelength grid (1-D).
    lbl : ndarray
        Labels stacked column-wise, shape (n_labels, total_spectra).
    flx : ndarray
        Fluxes stacked column-wise, shape (n_pixels, total_spectra).
    label_names : ndarray
        The label names array (1-D).
    """
    lbl_blocks, flx_blocks = [], []
    wvl = label_names = None

    for idx, fp in enumerate(file_paths, 1):
        with np.load(fp) as f:
            # --- consistency checks (first file defines the reference) ----
            if idx == 1:
                wvl = f["wvl"]
                label_names = f["label_names"]
            else:
                if not np.allclose(f["wvl"], wvl):
                    raise ValueError(f"{fp} has a different wavelength grid.")
                if not np.array_equal(f["label_names"], label_names):
                    raise ValueError(f"{fp} has different label_names ordering.")

            # --- collect blocks ------------------------------------------
            lbl_blocks.append(f["labels"])
            flx_blocks.append(f["flxn"])

    # --- final concatenation --------------------------------------------
    lbl = np.concatenate(lbl_blocks, axis=1)
    flx = np.concatenate(flx_blocks, axis=1)

    return wvl, lbl, flx, label_names

def load_data(file_path, train_fraction=0.75, random_state=42):
    """
    Load and prepare the data from a given npz file.

    Parameters
    ----------
    file_path : list of str or str
        Path to the .npz file containing 'wvl', 'labels', and 'flxn'.
    train_fraction : float
        Fraction of data to use for training. The rest is used for validation.
    random_state : int
        Random seed for reproducibility in train-test split.

    Returns
    -------
    x : torch.FloatTensor
        Training input data.
    y : torch.FloatTensor
        Training target data.
    x_valid : torch.FloatTensor
        Validation input data.
    y_valid : torch.FloatTensor
        Validation target data.
    x_min : np.ndarray
        Minimum values used for input scaling.
    x_max : np.ndarray
        Maximum values used for input scaling.
    num_pix : int
        Number of output pixels.
    dim_in : int
        Dimensionality of input parameters.
    """

    if type(file_path) is list:
        # If multiple files are provided, stack them
        wvl, lbl, flx, label_names = load_and_stack(file_path)
    else:
        temp = np.load(file_path)
        wvl = temp["wvl"]
        lbl = temp["labels"]
        flx = temp["flxn"]
        label_names = temp["label_names"]
        temp.close()

    # Example condition to filter data:
    #new = lbl[1] <= 5.0
    #lbl = lbl[:, new]
    #flx = flx[:, new]

    #new = lbl[1] >= 0.5
    #lbl = lbl[:, new]
    #flx = flx[:, new]

    x_train_raw, x_valid_raw, y_train_raw, y_valid_raw = train_test_split(
        lbl.T, flx.T, test_size=1 - train_fraction, random_state=random_state, shuffle=True
    )

    # Scale the inputs
    x_max = np.max(x_train_raw, axis=0)
    x_min = np.min(x_train_raw, axis=0)

    x_train_scaled = (x_train_raw - x_min) / (x_max - x_min) - 0.5
    x_valid_scaled = (x_valid_raw - x_min) / (x_max - x_min) - 0.5

    x = Variable(torch.from_numpy(x_train_scaled)).float()
    y = Variable(torch.from_numpy(y_train_raw), requires_grad=False).float()
    x_valid = Variable(torch.from_numpy(x_valid_scaled)).float()
    y_valid = Variable(torch.from_numpy(y_valid_raw), requires_grad=False).float()

    dim_in = x.shape[1]
    num_pix = y.shape[1]

    return x, y, x_valid, y_valid, x_min, x_max, num_pix, dim_in, wvl, label_names