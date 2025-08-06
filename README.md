# one_payne

This repository contains the code for training a neural network to predict stellar parameters from synthetic spectra using the Payne method. The Payne is a neural network that predicts stellar spectra based on stellar parameters by basically interpolating in a precomputed grid of synthetic spectra.

Original Payne paper: [Ting et al. 2019](https://ui.adsabs.harvard.edu/abs/2019ApJ...879...69T/abstract) with corresponding Github repository: [Payne](https://github.com/tingyuansen/The_Payne). Our approach follows closer the implementation in [Kovalev et al. 2019](https://ui.adsabs.harvard.edu/abs/2019A%26A...628A..54K/abstract). Both papers were used in the development of this repository. 

The main advantage of our network is that it is a single network (similar to Kovalev et al. 2019), optimised and tested to ensure the best performance. For details see the paper: Storm et al. in prep. (TBA soon).

A small test network and links to the usage of the network online to be added after submission of the paper.

## Usage instructions

0) Computing synthetic spectra.

This is the first step, and it’s crucial to have a large enough dataset (order ~thousands-tens thousands depending on your problem) of synthetic spectra. I used [TSFitPy](https://github.com/TSFitPy-developers/TSFitPy) to compute synthetic spectra. The most important outputs: wavelength, flux, and labels. The labels should include the parameters you want to predict (e.g., Teff, logg, [Fe/H], individual abundances, etc.).

Important! Make sure there are no problematic spectra (e.g. those containing NaNs, negative fluxes, etc.), as those initially prevented my network from converging properly.

I trained on fully normalised spectra, and my current network constrains the output between 0 and 1 using sigmoid. Feel free to change the architecture if that assumption doesn’t apply to your case.

1) Reformatting the spectra into one large .npz file

This simplifies loading for training using `process_data.py`. Things to change:

- `params_to_use` - a list of labels in a .csv file. The `specname_column_name` column should match the filenames of the synthetic spectra.
- `params_to_save` - parameters to save for training. Here you can change names (e.g. if you want to use A(Li) instead of [Li/Fe], etc.).
- `file_output_path` - name of the output .npz file.
- `stellar_labels_paths` - a list of paths to the .csv files with labels. The first column must be named the same as `specname_column_name` and match the filenames of the synthetic spectra.
- `paths_spectra_files` - a list of paths to the directories with synthetic spectra. 
- `wavelength_min` and `wavelength_max` - the wavelength range to cut the spectra to. This is useful to precut your synthetic spectra to your desired wavelength range. You can also cut into windows if needed.
- `spectral_files_extension` - change the file extension if your synthetic spectra files have a different one.
- `labels_pd['teff'] = labels_pd['teff'] / 1000` - used to convert Teff to kK, but it might not be necessary since we scale the labels anyway.
- `labels_pd['A_Li'] = labels_pd['Li_Fe'] + 1.05 + labels_pd['feh']` - an example of changing the labels. Adjust/remove as needed.
- `# Check normalization validity` - a few lines under this comment check if the spectra are good. Feel free to adjust or remove this check.
- `#file_name = file.replace(f'{spectral_files_extension}', '')` - this can adjust the name of the filename to correspond to the labels in the .csv file. If your filenames already match, you can keep it commented out.

This is the stage where you can adjust spectra as needed (e.g., cut to a specific wavelength range, change labels, remove emission lines, etc.). The output is a single .npz file containing all the spectra and labels.

Tip: I recommend scaling your labels so they behave nicely during training. For instance, you might want to use A(Li) instead of [Li/Fe], etc.

I train on spectra without any vmac or vsini broadening. You can apply broadening, but be aware that it might reduce network performance.

2) Training the network

The script for training is `train_payne.py`. It loads the data from the .npz file created in step 1. You can adjust the parameters in the script, such as:
- `def build_model(dim_in, num_pix, hidden_neurons):` 
  - This function defines the architecture of the network. You can change the number of hidden layers and activation functions here.
- `learning_rate` - The initial learning rate for the optimizer.
- `patience` - After how many `check_interval` to stop training if the validation loss does not improve.
- `check_interval` - How often to check the validation loss (in steps).
- `hidden_neurons` - The number of neurons in the hidden layers.
- `train_fraction` - The fraction of the dataset to use for training. The rest will be used for validation. (0 to 1, where 1 means all data is used for training).
- `t_max` - The number of training steps it takes for learning rate to decay to minimum value via cosine decay. It will keep training until the validation loss does not improve for `patience` times `check_interval`.
- `data_file` - The path to the .npz file with the spectra and labels created in step 1. It can be several files, and they will be concatenated.
- `checkpoint_dir="./tmp/checkpoints"` - The directory where the model checkpoints will be saved. EACH checkpoint will be saved with a datetime string, so please clean up the directory or setup an automatic cleanup.
- `meta_data["author"] = 'storm'` - Just saved in the metadata of the model, so you can change it to your name or nickname to keep track of who trained the model.

Correspondingly, the script `load_train_data_scripts.py` loads the data. You can adjust the following parameters:
- ```
    # Example condition to filter data:
    #new = lbl[1] <= 5.0
    #lbl = lbl[:, new]
    #flx = flx[:, new]

    #new = lbl[1] >= 0.5
    #lbl = lbl[:, new]
    #flx = flx[:, new]
  ```
  - You can prefilter the data even after loading. In this example, it filters out spectra with `lbl[1]` (for me it was logg) > 5.0 and < 0.5. You can adjust these conditions to filter your data as needed.

By training the network, it will print out the validation loss every `check_interval` steps and save each iteration of the model in the `checkpoint_dir`. The best model will be saved an `.npz` file in the current directory.

The last two lines (function calls) automatically evaluate the network (see below).

3) Evaluating the network

There are two scripts for evaluating the network: `payne_plot_performance.py` and `payne_plot_pixel_error.py`.

- `payne_plot_performance.py` tries to do fitting on your validation spectra and saves a figure (1 to 1 comparison of the labels vs predictions). Any horisontal lines mean that the network is not predicting that label (i.e. it stays at the input value). Typically that is because the elemental abundance is too small. That's why it saves a second plot with the predictions coloured by `[Fe/H]`. This gives a rough idea of how the network is performing.
  - The fitting is done using backwards minimisation. It takes quite a bit of time, but at least it gives an idea if the network converged or not and how well it performs on the validation noiseless spectra in different regions of the parameter space.
- `payne_plot_pixel_error.py` plots the pixel error of the network. It predicts the spectrum based on the labels and then compares it to the original spectrum. It saves many figures, showing the pixel error for each label and the average pixel error across all labels. This is good to understand where the network struggles the most and which labels affect the pixel error the most (typically low temperature, high metallicity, anything that causes molecules etc).

4) Using the network

The script `plot_payne_spectrum.py` can be used to plot the spectrum based on the labels. It loads the model and produces a plot. 

- `#wavelength_obs, flux_obs = np.loadtxt("your_path.txt", dtype=float, unpack=True, usecols=(0, 1))` - uncomment this line and change the path to your observed spectrum if you want to plot it together with the synthetic spectrum.
- ```
    payne_values = [5.777, 4.44, 0.0, 1.0] + [0.0] * (len(labels) - 4)
    vmac = 3            # vmac in km/s
    vsini = 1.6         # vsini in km/s
    doppler_shift = 0   # RV in km/s
  ```
  - This is an example of the labels you can use to plot the spectrum. Adjust the values according to your needs. The first four values are Teff, logg, [Fe/H], and vmic, respectively. The rest are individual abundances.
  - Broadening is automatically done posterior using FFT script.