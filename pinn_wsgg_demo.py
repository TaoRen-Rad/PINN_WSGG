import torch
import numpy as np
from pinn_wsgg.nn_wsgg import load_wsgg, nn_wsgg_rte
from pinn_wsgg.plot_setup import setup
from matplotlib import pyplot as plt

device = "cpu"
dtype = torch.float32

if __name__=="__main__":
    setup()
    nn_wsgg = load_wsgg(dtype).to(device)
    profiles = torch.tensor(np.load("data/profiles.npy")).to(device, dtype=dtype)
    results = np.load("data/results.npy")
    
    xs = profiles[:, 0]
    temperature_profiles = profiles[:, 4]
    x_h2o_profiles = profiles[:, 1]
    x_co2_profiles = profiles[:, 2]
    x_co_profiles = profiles[:, 3]
    T_l = torch.tensor(400.0).to(device, dtype=dtype)
    T_r = torch.tensor(400.0).to(device, dtype=dtype)
    
    wsgg_G, wsgg_q, wsgg_dq = nn_wsgg_rte(nn_wsgg, xs, temperature_profiles, 
        x_h2o_profiles, x_co2_profiles, x_co_profiles, T_l, T_r)
    xs = profiles[:, 0].numpy()
    fig, ax = plt.subplots(1, 1, figsize=[3, 2])
    ax.plot(xs, results[:, 2] / 1e6, "b", label="LBL")
    ax.plot(xs, wsgg_dq.detach().cpu().numpy().flatten() / 1e6, "r--", label="WSGG")
    ax.legend()
    ax.set_xlabel("$x$ [m]")
    ax.set_xlim([xs[0], xs[-1]])
    ax.set_ylabel(r"$\nabla \cdot q$ [MW/m$^3$]")
    plt.tight_layout()
    fig.savefig("demo.png")
