from __future__ import annotations

import numpy as np
import os
import datetime, time
import pandas as pd
from tqdm import tqdm


if __name__ == '__main__':
    print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))

    specname_column_name = "specname"
    # Teff logg Fe/H vmic Al/Fe Cr/Fe Na/Fe Ni/Fe Si/Fe A(Li) C/Fe Ca/Fe Ba/Fe O/Fe Mn/Fe Co/Fe Sr/Fe Eu/Fe Mg/Fe Ti/Fe Y/Fe
    params_to_use = [
    specname_column_name, "teff", "logg", "feh", "vmic", "Al_Fe", "Cr_Fe", "Na_Fe", "Ni_Fe", "Si_Fe", "Li_Fe", "C_Fe", "Ca_Fe", "Ba_Fe", "O_Fe", "Mn_Fe", "Co_Fe", "Sr_Fe", "Eu_Fe", "Mg_Fe", "Ti_Fe", "Y_Fe"
    ]
    params_to_save = [
    "teff", "logg", "feh", "vmic", "Al_Fe", "Cr_Fe", "Na_Fe", "Ni_Fe", "Si_Fe", "A_Li", "C_Fe", "Ca_Fe", "Ba_Fe", "O_Fe", "Mn_Fe", "Co_Fe", "Sr_Fe", "Eu_Fe", "Mg_Fe", "Ti_Fe", "Y_Fe"
    ]


    file_output_path = (
        f"./grid_test_batch0"
    )

    stellar_params_input_paths = ["../TSFitPy/synthetic_spectra/"]
    stellar_params_files = [f"spectra_parameters.csv"]

    path_files = [
        f"./batch0_spectra/"
    ]

    all_fluxes = []
    labels = []

    new_wavelength = None
    wavelength_mask = None
    wavelength_min, wavelength_max = 3700, 9700

    for path, stellar_params_file, stellar_params_input_path in zip(path_files, stellar_params_files, stellar_params_input_paths):
        print(f"Processing directory: {path}")

        labels_pd = pd.read_csv(f"{stellar_params_input_path}/{stellar_params_file}")
        labels_pd = labels_pd[params_to_use]

        # convert Teff from K to thousands of K
        labels_pd['teff'] = labels_pd['teff'] / 1000

        # convert [Li/Fe] to A(Li)
        labels_pd['A_Li'] = labels_pd['Li_Fe'] + 1.05 + labels_pd['feh']

        # Verify directory exists
        if not os.path.isdir(path):
            print(f"Warning: Directory {path} does not exist.")
            continue

        # Find all relevant files
        files = [f for f in os.listdir(path) if f.endswith(f'.spec')]
        print(f"Found {len(files)} files in '{path}'.")

        for file in tqdm(files):
            data_file_path = os.path.join(path, file)

            wavelength, norm_flux = np.loadtxt(data_file_path, unpack=True, usecols=(0, 1), dtype=float)

            if new_wavelength is None:
                wavelength_mask = (wavelength_min <= wavelength) & (wavelength <= wavelength_max)
                wavelength = wavelength[wavelength_mask]

                new_wavelength = wavelength
            
            norm_flux = norm_flux[wavelength_mask]

            # Check normalization validity
            if np.any(np.isnan(norm_flux)) or np.any(np.isinf(norm_flux)):
                print(f"Warning: Invalid normalization in file {file}. Skipping.")
                continue

            if len(np.where(norm_flux < 0)[0]) > 20:
                print(f"Warning: Negative or zero flux in file {file}. Skipping.")
                continue

            file_name = file.replace(f'.spec', '')

            idx_count = len(labels_pd[labels_pd[specname_column_name] == file_name])
            if idx_count == 1:
                all_fluxes.append(norm_flux)
                labels.append(labels_pd[labels_pd[specname_column_name] == file_name][params_to_save].values[0])
            else:
                print(f"Error in {file}: {idx_count} entries in the labels")


    num_pix = len(new_wavelength)
    num_spectra = len(all_fluxes)
    dim_in = len(params_to_save)
    flx = np.array(all_fluxes).T
    wvl = new_wavelength
    labels = np.array(labels).T
    label_names = params_to_save

    np.savez(f"{file_output_path}.npz", flxn=flx, wvl=wvl, labels=labels, label_names=label_names)
    print(f"Saved to {file_output_path}.npz")
    print(f"flxn.shape: {flx.shape}, wvl.shape: {wvl.shape}, labels.shape: {labels.shape}, label_names: {label_names}")
    print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
