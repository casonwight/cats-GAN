import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class CatDataset(Dataset):
    def __init__(self, data_dir='./data/cats/', train=True, seed=42, train_pct=.80):
        super().__init__()
        self.data_dir = data_dir
        all_img_labels = os.listdir(data_dir).pop(".gitkeep")
        num_train_imgs = int(len(all_img_labels) * train_pct)
        np.random.seed(seed)
        all_img_labels = np.random.permutation(all_img_labels)

        if train:
            self.img_labels = all_img_labels[:num_train_imgs]
        else:
            self.img_labels = all_img_labels[num_train_imgs:]
    
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_name = self.img_labels[idx]
        img_path = os.path.join(self.data_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return image
    
    def show_image(self, idx):
        img_name = self.img_labels[idx]
        img_path = os.path.join(self.data_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        image.show()
    
    def transform(self, image):
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        return transform(image)

if __name__ == '__main__':
    dataset = CatDataset()
    print(len(dataset))
    dataset.show_image(0)
    print(dataset[0])
