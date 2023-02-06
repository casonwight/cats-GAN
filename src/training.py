import torch
import pandas as pd
from tqdm import trange
from torchvision import transforms
import torchvision
from matplotlib import pyplot as plt
from datetime import datetime
import matplotlib.ticker as mtick
from torch.utils.tensorboard import SummaryWriter
from utils.data_loader import get_data_loaders
from models.generator import Generator
from models.discriminator import Discriminator


class GanTrainer:
    def __init__(self, **kwargs):
        # Default values for GAN training class
        self.device = 'cpu'
        self.data_dir = './data/cats/'
        self.save_dir='./saved_models/'
        self.save_name = 'cats'
        self.save_every = 100
        self.val_every = 25
        self.n_epochs = 1
        results_cols = ['i', 'type', 'epoch', 'batch', 'D_loss', 'G_loss', 'D_acc', 'G_perf']
        self.results_df = pd.DataFrame(columns=results_cols)
        self.generator = Generator(nz=100)
        self.discriminator = Discriminator()
        self.batch_size = 64
        self.val_batch_size = 128
        self.lr_g=0.0002 
        self.lr_d=0.0002
        self.beta1=0.5
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.tensorboard_path = 'runs/cats'

        # Update all values based on kwargs
        self.__dict__.update(kwargs)
        
        # Reset training variables
        self.nz = self.generator.nz
        self.train_dataloader, self.val_dataloader = get_data_loaders(self.data_dir, self.batch_size, self.val_batch_size)
        self.writer = SummaryWriter(self.tensorboard_path + datetime.now().strftime("_%m-%d-%Y_%H-%M-%S"))
        self.writer.add_graph(self.generator, torch.randn((1, self.nz)))
        self.writer.add_graph(self.discriminator, torch.randn((1, 3, 64, 64)))
        self.generator.to(self.device)
        self.discriminator.to(self.device)
        self.optimizerG = torch.optim.Adam(self.generator.parameters(), lr=self.lr_g, betas=(self.beta1, 0.999))
        self.optimizerD = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr_d, betas=(self.beta1, 0.999))
        self.i = len(self.results_df)
    
    def do_batch(self, train=True):
        if train:
            self.optimizerG.zero_grad()
            self.optimizerD.zero_grad()

        # Real images
        if train:
            real_images = next(iter(self.train_dataloader))
        else:
            real_images = next(iter(self.val_dataloader))
        num_images_batch = real_images.shape[0]
        real_images = real_images.to(self.device)
        real_labels = torch.ones((num_images_batch, 1), device=self.device)
        real_preds = self.discriminator(real_images)

        # Fake images
        noise = torch.randn((num_images_batch, self.nz), device=self.device)
        fake_images = self.generator(noise)
        fake_labels = torch.zeros((num_images_batch, 1), device=self.device)
        fake_preds = self.discriminator(fake_images)

        # Calculate loss
        d_loss = (self.criterion(real_preds, real_labels) + self.criterion(fake_preds, fake_labels)) / 2
        g_loss = self.criterion(fake_preds, real_labels)

        if train:
            # Gradient descent
            d_loss.backward(retain_graph=True)
            g_loss.backward()
            self.optimizerD.step()
            self.optimizerG.step()

        # Calculate discriminator accuracy
        real_acc = torch.mean((real_preds > 0).float()).item()
        fake_acc = torch.mean((fake_preds < 0).float()).item()
        d_acc = (real_acc + fake_acc) / 2

        # Calculate generator performance
        g_perf = torch.mean((fake_preds > 0).float()).item()

        # Append results
        results = {
            'type': 'train',
            'D_loss': d_loss.item(),
            'G_loss': g_loss.item(),
            'D_acc': d_acc * 100,
            'G_perf': g_perf * 100
        }

        if not train:
            self.show_example(fake_images[:4])

        type_str = 'train' if train else 'val'
        self.writer.add_scalars(
            'loss', 
            {
                f"Discriminator / {type_str}": results["D_loss"], 
                f"Generator / {type_str}": results["G_loss"]
            }, 
            self.i)
        self.writer.add_scalars(
            'Performance', 
            {
                f"Discriminator (accuracy) / {type_str}": results["D_acc"],
                f"Generator (trick performance) / {type_str}": results["G_perf"]
            }, 
            self.i)
        self.writer.flush()

        return results
    
    def save_models(self):
        generator_scripted = torch.jit.script(self.generator.cpu())
        discriminator_scripted = torch.jit.script(self.discriminator.cpu())
        generator_scripted.save(self.save_dir + f'generators/generator-{self.save_name}.pt')
        discriminator_scripted.save(self.save_dir + f'discriminators/discriminator-{self.save_name}.pt')

        self.generator.to(self.device)
        self.discriminator.to(self.device)


    def train(self):
        for epoch in range(1, self.n_epochs + 1):
            pbar = trange(len(self.train_dataloader))
            for batch in pbar:
                results_train = self.do_batch(train=True)
                results_train['i'] = self.i
                results_train['epoch'] = epoch
                results_train['batch'] = batch

                self.results_df = pd.concat([self.results_df, pd.DataFrame(results_train, index=[0])], ignore_index=True)

                pbar_desc = f'Epoch {epoch} | Batch {batch} | '
                pbar_desc += f'D_loss: {results_train["D_loss"]:.4f} | G_loss: {results_train["G_loss"]:.4f} | '
                pbar_desc += f'D_acc: {results_train["D_acc"]:.2f}% | G_perf: {results_train["G_perf"]:.2f}%'

                pbar.set_description(pbar_desc)
                
                # Save model checkpoints
                if self.i % self.save_every == 0:
                    self.save_models()

                # Validation 
                if self.i % self.val_every == 0:
                    with torch.no_grad():
                        results_val = self.do_batch(train=False)
                        results_val['i'] = self.i
                        results_val['epoch'] = epoch
                        results_val['batch'] = batch
                        self.results_df = pd.concat([self.results_df, pd.DataFrame(results_val, index=[0])], ignore_index=True)
       
                self.i += 1
        
        self.save_models()
        self.show_results()
    
    def show_example(self, generated_images):
        img_grid = torchvision.utils.make_grid(generated_images)
        self.writer.add_image('Generated Images', img_grid, self.i)

    def show_results(self):
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

        axs[0].plot(self.results_df['G_loss'], label="Generator")
        axs[0].plot(self.results_df['D_loss'], label="Discriminator")
        axs[0].set_title("Loss")
        axs[0].set_xlabel("Batch")
        axs[0].set_ylabel("Binary Cross-Entropy")
        axs[0].legend()
        axs[0].grid()

        axs[1].plot(self.results_df['G_perf'], label="Generator")
        axs[1].plot(self.results_df['D_acc'], label="Discriminator")
        axs[1].set_title("Accuracy / Performance")
        axs[1].set_xlabel("Batch")
        axs[1].set_ylabel("% Correct")
        axs[1].yaxis.set_major_formatter(mtick.PercentFormatter())
        axs[1].legend()
        axs[1].grid()

        fig.tight_layout()
        self.writer.add_figure('Loss and Accuracy', fig)
        self.writer.close()
        plt.show()

if __name__ == "__main__":
    gan = GanTrainer()
    gan.train()