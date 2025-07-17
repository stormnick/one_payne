from __future__ import annotations

import numpy as np
import torch
from torch.autograd import Variable
from torch import nn
import time
import argparse
import sys
import datetime
from sys import argv
from payne_plot_performance import do_plot
import os, math, pathlib
from torch.utils.data import TensorDataset, DataLoader
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from load_train_data_scripts import load_data




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
    return model, "Linear-SiLU-Linear-SiLU-Linear-SiLU-Linear-Sigmoid"


def train_model(
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        x_valid: torch.Tensor,
        y_valid: torch.Tensor,
        *,
        batch_size      : int   = 4096,
        valid_batch_size: int   = 8192,
        base_lr         : float = 1e-3,
        weight_decay    : float = 1e-2,
        t_max           : int|None = None,          # steps per cosine period
        patience        : int   = 20,
        check_interval  : int   = 1_000,            # validate & maybe checkpoint
        checkpoint_dir  : str   = "checkpoints",
        amp             : bool  = True,             # turn mixed precision on/off
        device          : str|None = None,
        hidden_neurons_amount: int = None
    ):
    """
    Memory-safe GPU trainer with AdamW, cosine LR, AMP and robust checkpointing.
    """

    # ───── DEVICE ─────────────────────────────────────────
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)              # single-node multi-GPU
    # For multi-node use torchrun + DistributedDataParallel.

    # ───── DATA LOADERS ──────────────────────────────────
    train_dl = DataLoader(
        TensorDataset(x, y),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True
    )
    valid_dl = DataLoader(
        TensorDataset(x_valid, y_valid),
        batch_size=valid_batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=2,
        persistent_workers=True
    )

    # ───── OPTIMISER & SCHEDULER ─────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr,
        weight_decay=weight_decay
    )

    # If user did not specify T_max, pick “~50 passes over train_dl”
    if t_max is None:
        t_max = 200_000

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=t_max,           # single half-cosine to zero
            eta_min=base_lr * 1e-4       # tiny but >0 keeps Adam state healthy
    )

    # ───── AMP SETUP ─────────────────────────────────────
    scaler = GradScaler(enabled=amp)

    # ───── BOOK-KEEPING ─────────────────────────────────
    checkpoint_dir = checkpoint_dir + f"_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    ckpt_root     = pathlib.Path(checkpoint_dir)
    ckpt_root.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    impatience    = 0
    step          = 0
    start         = time.perf_counter()

    # ───── TRAIN LOOP ───────────────────────────────────
    while impatience < patience:
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast('cuda', enabled=amp):
                pred  = model(xb)
                loss  = (pred - yb).pow(2).mean()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if step <= t_max:        
                scheduler.step()          # keep following the cosine
            step += 1

            # ───── VALIDATE / CHECKPOINT ────────────────
            if step % check_interval == 0:
                model.eval()
                val_losses = []
                with torch.no_grad(), autocast('cuda', enabled=amp):
                    for xb_v, yb_v in valid_dl:
                        xb_v, yb_v = xb_v.to(device, non_blocking=True), yb_v.to(device, non_blocking=True)
                        val_losses.append(((model(xb_v) - yb_v).pow(2)).mean().item())
                val_loss = float(np.mean(val_losses))

                hr = (time.perf_counter() - start) / 3600
                print(f"[{step:>7}]  val_loss={val_loss:.4e} | lr={scheduler.get_last_lr()[0]:.2e} | elapsed={hr:.2f} h")

                if val_loss < best_val_loss:          # ▸ improved
                    best_val_loss = val_loss
                    impatience    = 0
                    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    fname = ckpt_root / f"step{step:07d}_val{val_loss:.4e}_{ts}.pt"
                    torch.save(
                        {
                            "step": step,
                            "model_state": model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                            "optimizer_state": optimizer.state_dict(),
                            "scheduler_state": scheduler.state_dict(),
                            "scaler_state":   scaler.state_dict(),
                            "val_loss": val_loss
                        },
                        fname
                    )
                    print(f"    ★  saved {fname}")
                else:
                    impatience += 1
            
            if step > 3 * t_max:
                impatience += 2  # if we are way past T_max, increase impatience
            
            if impatience >= patience:
                break   # early-stop exits inner loop as well

    # ───── WRAP-UP ───────────────────────────────────────
    hours = (time.perf_counter() - start) / 3600
    print(f"\nStopped after {step} steps ({hours:.2f} h). Best val_loss = {best_val_loss:.5e}")

    # Reload best weights, move to CPU, convert to numpy list
    best_ckpt = sorted(ckpt_root.glob("*.pt"))[-1]
    print(f"Loading best checkpoint: {best_ckpt}")
    state = torch.load(best_ckpt, map_location="cpu")["model_state"]

    model_cpu = (model.module if isinstance(model, nn.DataParallel) else model).cpu()
    model_cpu.load_state_dict(state)

    model_numpy = [p.detach().numpy().copy() for p in model_cpu.parameters()]

    meta = dict(
        base_lr        = base_lr,
        weight_decay   = weight_decay,
        batch_size     = batch_size,
        amp            = amp,
        t_max          = t_max,
        patience       = patience,
        check_interval = check_interval,
        best_val_loss  = best_val_loss,
        total_steps    = step,
        device         = device,
        checkpoint_dir = str(ckpt_root.resolve()),
        hidden_neurons = hidden_neurons_amount
    )
    return model_numpy, meta


def random_network_for_testing(model):
    """
    Randomize the model parameters for testing purposes.

    Parameters
    ----------
    model : torch.nn.Sequential
        The model to randomize.

    Returns
    -------
    model_numpy : list of np.ndarray
        Randomized model parameters.
    """
    model_numpy = [param.data.numpy().copy() for param in model.parameters()]
    for param in model_numpy:
        param[:] = np.random.randn(*param.shape)
    return model_numpy


def save_model_parameters(file_path, model_numpy, x_min, x_max, meta_data):
    """
    Save the model parameters and scaling info to a .npz file.

    Parameters
    ----------
    file_path : str
        Path to the output file (without extension).
    model_numpy : list of np.ndarray
        Model parameters to save.
    x_min : np.ndarray
        Minimum values for input scaling.
    x_max : np.ndarray
        Maximum values for input scaling.
    """
    w_array_0 = model_numpy[0]
    b_array_0 = model_numpy[1]
    w_array_1 = model_numpy[2]
    b_array_1 = model_numpy[3]
    w_array_2 = model_numpy[4]
    b_array_2 = model_numpy[5]
    w_array_3 = model_numpy[6]
    b_array_3 = model_numpy[7]

    np.savez(file_path,
             w_array_0=w_array_0,
             w_array_1=w_array_1,
             w_array_2=w_array_2,
             b_array_0=b_array_0,
             b_array_1=b_array_1,
             b_array_2=b_array_2,
             b_array_3=b_array_3,
             w_array_3=w_array_3,
             x_max=x_max,
             x_min=x_min,
                **meta_data)


if __name__ == "__main__":
    learning_rate = 0.001  # original 0.001
    patience = 10
    check_interval = 1000
    hidden_neurons = 1024
    weight_decay = 0.001
    train_fraction = 0.93
    t_max = 200_000

    data_file = ["test_batch0.npz"]
    output_file = f"{argv[2]}_{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.npz"

    # print current time
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print(f"Training network with {hidden_neurons} hidden neurons, learning rate {learning_rate} and patience {patience} using data from {data_file} and saving to {output_file}")
    # Load the data
    x, y, x_valid, y_valid, x_min, x_max, num_pix, dim_in, wvl, label_names = load_data(data_file, train_fraction=train_fraction)

    # Build the model
    model, model_architecture = build_model(dim_in, num_pix, hidden_neurons)

    # Train the model
    model_numpy, meta_data = train_model(model, x, y, x_valid, y_valid,
                              base_lr=learning_rate,
                              patience=patience,
                              check_interval=check_interval, weight_decay=weight_decay, hidden_neurons_amount=hidden_neurons,
                              checkpoint_dir="./tmp/checkpoints", t_max=t_max)

    meta_data["date"] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    meta_data["wavelength"] = wvl
    meta_data["label_names"] = label_names
    meta_data["author"] = 'storm'
    meta_data['model_architecture'] = model_architecture
    meta_data['data_file_path'] = data_file

    # Save parameters
    if model_numpy is not None:
        save_model_parameters(output_file, model_numpy, x_min, x_max, meta_data)
        print(f"Model parameters saved to {output_file}")
    else:
        print("No best model found, no file saved.")

    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    do_plot(data_file, output_file, train_fraction=train_fraction)