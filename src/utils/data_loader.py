from .data_set import CatDataset
from torch.utils.data import DataLoader

def get_data_loaders(data_dir, batch_size=64, val_batch_size=128):
    train_dataset = CatDataset(data_dir=data_dir, train=True)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = CatDataset(data_dir=data_dir, train=False)
    val_data_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=True)
    return train_data_loader, val_data_loader

if __name__ == '__main__':
    train_data_loader, val_data_loader = get_data_loaders('./data/cats/')
    print(len(train_data_loader), len(val_data_loader))