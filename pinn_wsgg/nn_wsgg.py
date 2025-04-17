import torch
import json
from scipy.constants import Stefan_Boltzmann
from .rte import solve_rte
from .mlp import MLP
from . import DATA_DIR

class NN_WSGG(torch.nn.Module):
    def __init__(self, hidden_dims, x_mean: torch.Tensor, x_std: torch.Tensor):
        super(NN_WSGG, self).__init__()
        self.mlp_weights = MLP([4, *hidden_dims, 5])
        self.mlp_abscs = MLP([3, *hidden_dims, 4])
        
        self.register_buffer('x_mean', x_mean)
        self.register_buffer('x_std', x_std)
    
    def forward(self, x):
        # the first column is temperature, the remaining are volume fraction
        x = (x - self.x_mean) / self.x_std
        weights = torch.softmax(self.mlp_weights(x), 1)
        abscs = torch.zeros_like(weights)
        abscs[:, 1:] = torch.exp(self.mlp_abscs(x[:, 1:]))
        return weights, abscs

def load_wsgg(dtype):
    with open(f"{DATA_DIR}/wsgg.json") as file:
        model_para = json.load(file)
    for key, value in model_para.items():
        model_para[key] = torch.tensor(value).to(dtype=dtype)
    x_mean, x_std = model_para["x_mean"], model_para["x_std"]
    nn_wsgg = NN_WSGG([32, 32], x_mean, x_std)
    nn_wsgg.load_state_dict(model_para)
    return nn_wsgg

def nn_wsgg_epsilon(weights, abscs, xs, pL):
    sum_xs = torch.sum(xs, axis=1, keepdim=True)
    epsilon = torch.sum(weights * (1 - torch.exp(-abscs * sum_xs * pL)), dim=1)
    return epsilon

def nn_wsgg_rte(nn_wsgg, xs, temperature_profiles, x_h2o_profiles, 
             x_co2_profiles, x_co_profiles, T_l, T_r):
    nn_input = torch.vstack([temperature_profiles, x_h2o_profiles, x_co2_profiles, x_co_profiles]).T
    wsgg_a, wsgg_k = nn_wsgg(nn_input)
    wsgg_k *= (x_h2o_profiles + x_co2_profiles + x_co_profiles).reshape(-1, 1)
    
    wsgg_G = 0.0
    wsgg_q = 0.0
    wsgg_dq = 0.0

    for i in range(5):
        a = wsgg_a[:, i]
        k = wsgg_k[:, i] * 100.0
        
        i_b = Stefan_Boltzmann * (temperature_profiles ** 4) * a / torch.pi
        i_b1 = Stefan_Boltzmann * (T_l ** 4) * a[0] / torch.pi 
        i_b2 = Stefan_Boltzmann * (T_r ** 4) * a[-1] / torch.pi
        cur_G, cur_q, cur_dq = solve_rte(xs, k, i_b, i_b1, i_b2)
        wsgg_G += cur_G
        wsgg_q += cur_q
        wsgg_dq += cur_dq
    return wsgg_G, wsgg_q, wsgg_dq
