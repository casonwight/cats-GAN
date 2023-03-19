# WGAN for cats
import torch
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt


class GanTrainer:
    def __init__(self, train_dataloader, val_dataloader, generator, discriminator, **kwargs):
        self.num_epochs = 50
        self.discriminator_iter = 5
        self.lambda_gp = 10
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.generator = generator
        self.discriminator = discriminator
        self.device = 'cpu'
        self.discriminator_iter = 5
        self.d_optimizer = torch.optim.RMSprop(self.discriminator.parameters(), lr=5e-5)
        self.g_optimizer = torch.optim.RMSprop(self.generator.parameters(), lr=5e-5)
        self.val_every = 100
        self.show_every = 100
        self.save_every = 100
        self.save_gif = False
        self.display_num_images = 16
        self.display_cols = int(self.display_num_images ** 0.5)
        self.displayed_images = []
        self.constant_noise = torch.randn(self.display_num_images, self.generator.nz, 1, 1).cpu()

        self.fig, self.axes = plt.subplots(nrows=self.display_cols, ncols=self.display_cols, figsize=(8, 8))
        self.fig.subplots_adjust(hspace=0.1, wspace=0.1)

        self.training_results = pd.DataFrame(columns=[
            'i',
            'epoch',
            'batch',
            'type',
            'g_loss',
            'g_acc',
            'd_loss',
            'd_acc'
        ])
        self.__dict__.update(kwargs)

    def calculate_gradient_penalty(self, real_images, fake_images):
        batch_size = real_images.size(0)

        alpha = torch.rand(batch_size, 1, 1, 1).to(self.device)
        interpolated_images = (alpha * real_images) + ((1 - alpha) * fake_images)
        interpolated_images.requires_grad_(True)

        d_interpolated = self.discriminator(interpolated_images)

        gradients = torch.autograd.grad(outputs=d_interpolated, inputs=interpolated_images,
                                        grad_outputs=torch.ones(d_interpolated.size()).to(self.device),
                                        create_graph=True, retain_graph=True)[0]

        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambda_gp

        return gradient_penalty

    def train_discriminator(self, batch_size, real_images, val=False):
        self.discriminator.zero_grad()

        noise = torch.randn(batch_size, self.generator.nz, 1, 1).to(self.device)
        fake_images = self.generator(noise)

        d_real = self.discriminator(real_images)
        d_fake = self.discriminator(fake_images)

        gradient_penalty = self.calculate_gradient_penalty(real_images, fake_images)

        d_loss = d_fake.mean() - d_real.mean() + gradient_penalty

        if not val:
            d_loss.backward()
            self.d_optimizer.step()

        d_accuracy = (d_real > 0).float().mean() / 2 + (d_fake < 0).float().mean() / 2

        return d_loss.item(), d_accuracy

    def train_generator(self, batch_size, val=False):
        self.generator.zero_grad()
        
        noise = torch.randn(batch_size, self.generator.nz, 1, 1).to(self.device)
        fake_images = self.generator(noise)
        d_fake = self.discriminator(fake_images)

        g_loss = -d_fake.mean()

        if not val:
            g_loss.backward()
            self.g_optimizer.step()

        g_accuracy = (d_fake > 0).float().mean()

        return g_loss.item(), g_accuracy

    def show_images(self):    
        with torch.no_grad():
            self.generator = self.generator.cpu()
            fake_images = self.generator(self.constant_noise).detach().cpu()    
            self.generator = self.generator.to(self.device) 
            fake_images = torch.nn.functional.relu(fake_images.clone().detach().cpu().permute(0, 2, 3, 1))
            fake_images = (fake_images * 255).type(torch.uint8)

            for ax in self.axes.flatten():
                ax.clear()

            for i, ax in enumerate(self.axes.flatten()):
                ax.imshow(fake_images[i])
                ax.axis('off')

            plt.show(block=False)
            return fake_images

    def train(self):
        plt.ion()
        i = 0
        for epoch in range(self.num_epochs):
            pbar = tqdm(enumerate(self.train_dataloader, i))
            pbar.set_description(f'Epoch {epoch+1}/{self.num_epochs}, Batch: {1}/{len(self.train_dataloader)}')

            for batch_num, real_images in pbar:
                real_images = real_images.to(self.device)
                batch_size = real_images.size(0)

                # Train discriminator
                d_loss, d_acc = self.train_discriminator(batch_size, real_images)

                # Train generator (every self.discriminator_iter iterations)
                if i % self.discriminator_iter == 0:
                    g_loss, g_acc = self.train_generator(batch_size)
                
                row_results = {
                    'i': i,
                    'epoch': epoch+1,
                    'batch': batch_num+1,
                    'type': 'train',
                    'g_loss': g_loss,
                    'g_acc': g_acc,
                    'd_loss': d_loss,
                    'd_acc': d_acc
                }
                self.training_results = pd.concat([self.training_results, pd.DataFrame(row_results, index=[0])], ignore_index=True)

                pbar.set_description(f"Epoch: {epoch+1}/{self.num_epochs}, Batch: {batch_num+1}/{len(self.train_dataloader)}, d_loss: {d_loss:,.2f}, d_acc: {100*d_acc:.2f}%, g_loss: {g_loss:,.2f}, g_acc: {100*g_acc:.2f}%")

                if i % self.val_every == 0:
                    real_images = next(iter(self.val_dataloader)).to(self.device)
                    d_loss, d_acc = self.train_discriminator(batch_size, real_images, val=True)
                    g_loss, g_acc = self.train_generator(batch_size, val=True)

                    row_results = {
                        'i': i,
                        'epoch': epoch+1,
                        'batch': batch_num+1,
                        'type': 'val',
                        'g_loss': g_loss,
                        'g_acc': g_acc,
                        'd_loss': d_loss,
                        'd_acc': d_acc
                    }
                    self.training_results = pd.concat([self.training_results, pd.DataFrame(row_results, index=[0])], ignore_index=True)

                if i % self.show_every == 0:
                    fake_images = self.show_images()
                    self.displayed_images.append(fake_images)

                if i % self.save_every == 0:
                    # Save generator and discriminator as a .pt file with jit
                    torch.jit.save(torch.jit.script(self.generator), f'saved_models/generator.pt')
                    torch.jit.save(torch.jit.script(self.discriminator), f'saved_models/discriminator.pt')

                i += 1

        plt.ioff()
        plt.show()

if __name__ == '__main__':
    from utils.data_loader import get_data_loaders
    from models.generator import Generator
    from models.discriminator import Discriminator

    data_dir = 'data/cats/'
    batch_size = 64
    val_batch_size = 64
    train_dataloader, val_dataloader = get_data_loaders(data_dir, batch_size, val_batch_size)
    generator, discriminator = Generator(), Discriminator()

    trainer = GanTrainer(train_dataloader, val_dataloader, generator, discriminator)
    trainer.train()