from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import spectral_model
from convolve import conv_macroturbulence, conv_rotation, conv_res

# Created by storm at 06.08.25


if __name__ == '__main__':
    # we include a mini 4MOST Payne model here for testing purposes.
    # note that the mini Payne model is different from the one used in the Payne paper.
    # it is a smaller model and may give different results.
    # replace the path with your own Payne model.
    path_model = "../test_network/payne_tsnlte_fgk_4most_hr_mini_2025-08-01-11-57-03_storm.npz"
    payne_coeffs, wavelength_payne, labels = spectral_model.load_payne(path_model)
    x_min = list(payne_coeffs[-2])
    x_max = list(payne_coeffs[-1])

    # in case you want to degrade the resolution of the Payne model
    resolution_val = None

    wavelength_obs, flux_obs = None, None
    # you can load your spectrum here
    #wavelength_obs, flux_obs = np.loadtxt("your_path.txt", dtype=float, unpack=True, usecols=(0, 1))

    # teff, logg, feh, vmic, individual abundances
    # teff is in kK, because that is how we input labels into the training set. Adjust if needed.
    payne_values = [5.777, 4.44, 0.0, 1.0] + [0.0] * (len(labels) - 4)
    vmac = 3            # vmac in km/s
    vsini = 1.6         # vsini in km/s
    doppler_shift = 0   # RV in km/s

    scaled_labels = (payne_values - payne_coeffs[-2]) / (payne_coeffs[-1] - payne_coeffs[-2]) - 0.5
    payne_fitted_spectra = spectral_model.get_spectrum_from_neural_net(scaled_labels=scaled_labels, NN_coeffs=payne_coeffs)

    wavelength_payne_plot = wavelength_payne
    if doppler_shift != 0:
        wavelength_payne_plot = wavelength_payne_plot * (1 + (doppler_shift / 299792.))
    if vmac > 1e-3:
        wavelength_payne_plot, payne_fitted_spectra = conv_macroturbulence(wavelength_payne_plot, payne_fitted_spectra, vmac)
    if vsini > 1e-3:
        wavelength_payne_plot, payne_fitted_spectra = conv_rotation(wavelength_payne_plot, payne_fitted_spectra, vsini)
    if resolution_val is not None:
        wavelength_payne_plot, payne_fitted_spectra = conv_res(wavelength_payne_plot, payne_fitted_spectra, resolution_val)

    extra_dlam = 5

    # Define the windows to display in the main (broken-axis) plot
    windows = [(3926 - extra_dlam, 4355 + extra_dlam), (5160 - extra_dlam, 5730 + extra_dlam),
               (6100 - extra_dlam, 6790 + extra_dlam)]

    mosaic = [['m1', 'm2', 'm3']]
    fig, axs = plt.subplot_mosaic(mosaic, figsize=(16, 8), gridspec_kw={'wspace': 0.05})

    for i, ax_key in enumerate(['m1', 'm2', 'm3']):
        ax = axs[ax_key]
        w0, w1 = windows[i]
        if wavelength_obs is not None and flux_obs is not None:
            ax.scatter(wavelength_obs, flux_obs, color='red', s=0.7, rasterized=True)
        ax.plot(wavelength_payne_plot, payne_fitted_spectra, color='black', lw=0.6)
        ax.set_xlim(w0, w1)
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis='both', which='major', labelsize=11)
        if i != 0:
            ax.spines['left'].set_visible(False)
            ax.set_yticks([])
        if i != len(windows) - 1:
            ax.spines['right'].set_visible(False)
        if i == 0:
            ax.set_ylabel('Normalised Flux')

    # Draw diagonal lines to indicate breaks
    d = 0.015 * (axs['m1'].get_ylim()[1] - axs['m1'].get_ylim()[0])
    for i in [0, 1]:
        ax_left = axs[f'm{i+1}']
        ax_right = axs[f'm{i+2}']
        kwargs = dict(transform=ax_left.transAxes, color='k', clip_on=False)
        ax_left.plot((1, 1 + 0.02), (-d, +d), **kwargs)
        kwargs = dict(transform=ax_right.transAxes, color='k', clip_on=False)
        ax_right.plot((-0.02, 0), (-d, +d), **kwargs)

    axs['m2'].set_xlabel('Wavelength [Å]')
    plt.show()