import torch

class MLP(torch.nn.Module):
    def __init__(self, layer_dims, activation=torch.nn.ReLU()):
        super(MLP, self).__init__()
        self.layers = torch.nn.ModuleList()
        for i in range(len(layer_dims) - 1):
            self.layers.append(torch.nn.Linear(layer_dims[i], layer_dims[i + 1]))
            if i < len(layer_dims) - 2:
                self.layers.append(activation)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x