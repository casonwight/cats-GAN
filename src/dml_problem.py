import torch
from torch import nn
import torch_directml

gpu_device = torch_directml.device() # When set to 'cpu', code works

x_in = torch.rand(10, 3, 64, 64, requires_grad=True).to(gpu_device)
model = nn.Sequential(nn.Conv2d(3, 16, 3), nn.BatchNorm2d(16)).to(gpu_device) # Without BatchNorm, code works
y_out = model(x_in)

gradient = torch.autograd.grad(
    inputs=x_in,
    outputs=y_out,
    retain_graph=True,
    create_graph=True,
    grad_outputs=torch.ones_like(y_out)                       
)[0]

gp = ((gradient.norm(2, dim=1) - 1) ** 2).mean()
gp.backward()