import torch
import torch_directml
from training import GanTrainer

def main():
    gpu_device = torch_directml.device()
    gan = GanTrainer(device=gpu_device, n_epochs=50, save_every=300, val_every=100)
    gan.train()

if __name__ == "__main__":
    main()