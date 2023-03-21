import torch
from training import GanTrainer
from utils.data_loader import get_data_loaders
from models.generator import Generator
from models.discriminator import Discriminator

    

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = '../data/cats/'
    batch_size = 64
    val_batch_size = 64
    train_dataloader, val_dataloader = get_data_loaders(data_dir, batch_size, val_batch_size)
    generator, discriminator = Generator(), Discriminator()
    generator, discriminator = generator.to(device), discriminator.to(device)

    trainer = GanTrainer(train_dataloader, val_dataloader, generator, discriminator, device=device, num_epochs=1)
    trainer.train()

if __name__ == "__main__":
    main()