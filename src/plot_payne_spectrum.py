from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import spectral_model
from convolve import conv_macroturbulence, conv_rotation, conv_res

# Created by storm at 06.08.25


if __name__ == '__main__':
    path_model = "your_model_name.npz"
    payne_coeffs, wavelength_payne, labels = spectral_model.load_payne(path_model)
    x_min = list(payne_coeffs[-2])
    x_max = list(payne_coeffs[-1])

    # in case you want to degrade the resolution of the Payne model
    resolution_val = None

    # you can load your spectrum here
    #wavelength_obs, flux_obs = np.loadtxt("path.txt", dtype=float, unpack=True, usecols=(0, 1))
    wavelength_obs, flux_obs = None, None

    # teff, logg, feh, vmic, individual abundances
    # teff is in kK, because that is how we input labels into the training set. Adjust if needed.
    payne_values = [5.777, 4.44, 0.0, 1.0] + [0.0] * (len(labels) - 4)
    vmac = 3
    vsini = 1.6
    doppler_shift = 0

    scaled_labels = (payne_values - payne_coeffs[-2]) / (payne_coeffs[-1] - payne_coeffs[-2]) - 0.5
    payne_fitted_spectra = spectral_model.get_spectrum_from_neural_net(scaled_labels=scaled_labels, NN_coeffs=payne_coeffs)

    wavelength_payne_plot = wavelength_payne
    if vmac > 1e-3:
        wavelength_payne_plot, payne_fitted_spectra = conv_macroturbulence(wavelength_payne_plot, payne_fitted_spectra, vmac)
    if vsini > 1e-3:
        wavelength_payne_plot, payne_fitted_spectra = conv_rotation(wavelength_payne_plot, payne_fitted_spectra, vsini)
    if resolution_val is not None:
        wavelength_payne_plot, payne_fitted_spectra = conv_res(wavelength_payne_plot, payne_fitted_spectra, resolution_val)

    plt.figure(figsize=(18, 6))
    if wavelength_obs is not None and flux_obs is not None:
        plt.scatter(wavelength_obs, flux_obs, label="Observed", s=3, color='k')
    plt.plot(wavelength_payne_plot * (1 + (doppler_shift / 299792.)), payne_fitted_spectra, label="Payne", color='r')
    plt.ylim(0.0, 1.05)
    plt.xlim(wavelength_payne_plot[0], wavelength_payne_plot[-1])
    plt.show()