from __future__ import annotations

import torch, numpy as np, matplotlib.pyplot as plt
from torch.autograd import Variable
import pandas as pd
import datetime
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


if __name__ == "__main__":
    print("Plotting pixel-wise error of the neural network model")

    data_file = argv[1]
    model_path = argv[2]

    data_file = ["/nexus/posix0/MIA-bergemann-cluster/nisto/payne/grid_nlte_4mostified_inf_noise_may2025_hrs_batch0.npz", 
                 "/nexus/posix0/MIA-bergemann-cluster/nisto/payne/grid_nlte_4mostified_inf_noise_may2025_hrs_batch1.npz",
                 "/nexus/posix0/MIA-bergemann-cluster/nisto/payne/grid_nlte_4mostified_inf_noise_may2025_hrs_batch2.npz",]
    

    time_to_save = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    if '/' in model_path:
        model_path_save = model_path.split('/')[-1].split('.')[0]
    else:
        model_path_save = model_path.split('.')[0]

    # load model
    x, y, x_valid, y_valid, x_min, x_max, num_pix, dim_in, wvl = load_data(data_file, train_fraction=0.93)

    print("Loaded data: file", data_file)

    data = np.load(model_path)
    x_min = data['x_min']
    x_max = data['x_max']
    hidden_neurons = data['hidden_neurons']
    label_names = data['label_names']

    model = build_model(dim_in, num_pix, hidden_neurons)
    model = load_model_parameters(data, model)

    print("Loaded model: file", model_path)

    model.eval()

    print("Model is in evaluation mode")

    # --- 2. Forward pass on the validation labels --------------------------------
    with torch.no_grad():
        x_val_scaled = x_valid      # same scaling as training
        x_val_tensor = torch.as_tensor(x_val_scaled, dtype=torch.float32)
        y_pred       = model(x_val_tensor).cpu().numpy()              # shape (Nval, num_pix)

    # --- 3. Residuals & pixel-level metrics --------------------------------------
    residual      = y_pred - y_valid.numpy()                                  # (Nval, num_pix)
    rmse_per_pix  = np.sqrt(np.mean(residual**2, axis=0))             # (num_pix,)
    mean_err      = residual.mean(axis=0)                             # signed bias
    std_per_pix   = residual.std(axis=0)

    print("Residuals calculated")

    # --- 4. Plotting --------------------------------------------------------------
    plt.figure(figsize=(8,4))
    plt.plot(wvl, rmse_per_pix, lw=1.2, label='RMSE')
    plt.xlabel(r'Wavelength [Å]')
    plt.ylabel('Flux error')
    plt.title('Pixel-wise validation error')
    plt.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(f"sqrt_mean_error_{time_to_save}_model_{model_path_save}.png")
    #plt.show()
    plt.close()

    # (Optional) add ±1 σ envelope of signed errors
    plt.figure(figsize=(8,4))
    plt.plot(wvl, mean_err, lw=1, label='mean residual')
    plt.fill_between(wvl, mean_err-std_per_pix, mean_err+std_per_pix, alpha=0.25,
                    label='±1 σ over validation set')
    plt.xlabel(r'Wavelength [Å]')
    plt.ylabel('Flux residual')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"error_sigma_{time_to_save}_model_{model_path_save}.png")
    #plt.show()
    plt.close()
    rmse_spec = np.sqrt(np.mean(residual**2, axis=1))  # one scalar per validation star

    # ---------------------------------------------------------------------------
    # 0.  Choose the columns that store each label
    # ---------------------------------------------------------------------------
    label_names_list = list(label_names)
    idx_teff = label_names_list.index('teff')
    idx_logg = label_names_list.index('logg')
    idx_feh  = label_names_list.index('feh')
    idx_vmic = label_names_list.index('vmic')      # change the name if yours differs

    # ---------------------------------------------------------------------------
    # 1.  Convenience: pick a common colour scale (0 → 98-percentile)
    # ---------------------------------------------------------------------------
    vmax = np.percentile(rmse_spec, 98)       # robust upper limit
    norm = plt.Normalize(vmin=0, vmax=vmax)

    # ---------------------------------------------------------------------------
    # 2.  Define a helper to avoid repetition
    # ---------------------------------------------------------------------------
    def scatter_ax(ax, x, y, c, xlabel, ylabel, **kw):
        sc = ax.scatter(x, y, c=c, cmap='viridis', norm=norm,
                        s=2, alpha=0.8, rasterized=True, **kw)
        ax.set_xlabel(xlabel);  ax.set_ylabel(ylabel)
        ax.grid(alpha=.2)
        return sc

    # ---------------------------------------------------------------------------
    # 3.  Build a 2×2 panel of the four requested planes
    # ---------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(8, 7))#, sharex='col', sharey='row')
    axes      = axes.ravel()

    sc1 = scatter_ax(axes[0], scale_back(x_valid[:,idx_teff], x_min[idx_teff], x_max[idx_teff], label_name=label_names[idx_teff]), scale_back(x_valid[:,idx_logg], x_min[idx_logg], x_max[idx_logg], label_name=label_names[idx_logg]), rmse_spec,
                    r'$T_{\mathrm{eff}}$', r'$\log g$')
    axes[0].set_title('RMSE over (Teff, log g)')
    # reverse y and x axes
    axes[0].set_xlim(axes[0].get_xlim()[::-1])
    axes[0].set_ylim(axes[0].get_ylim()[::-1])

    scatter_ax(axes[1], scale_back(x_valid[:,idx_teff], x_min[idx_teff], x_max[idx_teff], label_name=label_names[idx_teff]), scale_back(x_valid[:,idx_feh], x_min[idx_feh], x_max[idx_feh], label_name=label_names[idx_feh]),  rmse_spec,
            r'$T_{\mathrm{eff}}$', '[Fe/H]')
    axes[1].set_title('RMSE over (Teff, [Fe/H])')

    scatter_ax(axes[2], scale_back(x_valid[:,idx_logg], x_min[idx_logg], x_max[idx_logg], label_name=label_names[idx_logg]), scale_back(x_valid[:,idx_feh], x_min[idx_feh], x_max[idx_feh], label_name=label_names[idx_feh]), rmse_spec,
            r'$\log g$', '[Fe/H]')
    axes[2].set_title('RMSE over (log g, [Fe/H])')

    scatter_ax(axes[3], scale_back(x_valid[:,idx_feh], x_min[idx_feh], x_max[idx_feh], label_name=label_names[idx_feh]), scale_back(x_valid[:,idx_vmic], x_min[idx_vmic], x_max[idx_vmic], label_name=label_names[idx_vmic]), rmse_spec,
            '[Fe/H]',  r'$\xi_\mathrm{mic}$')
    axes[3].set_title('RMSE over ([Fe/H], vmic)')

    # Add one shared colour-bar
    #cb = fig.colorbar(sc1, ax=axes, shrink=0.9, pad=0.02, label='per-spectrum RMSE')
    plt.tight_layout()
    plt.savefig(f"performance_stellar_params_{time_to_save}_model_{model_path_save}.png")
    #plt.show()

    for idx in range(4, len(label_names)):
        if idx == idx_teff:             # skip Teff-vs-Teff
            continue

        plt.figure(figsize=(5, 4))
        plt.scatter(scale_back(x_valid[:,idx_teff], x_min[idx_teff], x_max[idx_teff], label_name=label_names[idx_teff]),              # Teff on x
                    scale_back(x_valid[:, idx], x_min[idx], x_max[idx]),                   # current label on y
                    c=rmse_spec, cmap='viridis', norm=norm,
                    s=2, alpha=0.8, rasterized=True)
        plt.xlabel(r'$T_{\mathrm{eff}}$')
        plt.ylabel(label_names[idx])
        plt.title(f'RMSE over (Teff, {label_names[idx]})')
        plt.grid(alpha=.2)

        # attach colour-bar for this plot only
        cbar = plt.colorbar(label='per-spectrum RMSE')
        cbar.ax.set_ylabel('RMSE', rotation=270, labelpad=15)

        plt.tight_layout()
        plt.savefig(f"performance_stellar_params_{label_names[idx]}_{time_to_save}_model_{model_path_save}.png")
        #plt.show()