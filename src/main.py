import torch
from training import GanTrainer

def main():
    device = device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gan = GanTrainer(device=device, n_epochs=50, save_every=300, val_every=100)
    gan.train()

if __name__ == "__main__":
    main()