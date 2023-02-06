import torch
import torch_directml
from training import GanTrainer

def main():
    gpu_device = torch_directml.device()
    gan = GanTrainer(device=gpu_device)
    gan.train()

if __name__ == "__main__":
    main()
