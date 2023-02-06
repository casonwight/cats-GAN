import torch
import pandas as pd
from tqdm import tqdm
from utils.data_loader import get_data_loaders
from models.generator import Generator
from models.discriminator import Discriminator


def train_gan(dataloader_func=get_data_loaders,
              generator=Generator(nz=100),
              discriminator=Discriminator(),
              lr_g=0.0002, 
              lr_d=0.0002,
              beta1=0.5,
              batch_size=64,
              val_batch_size=128,
              n_epochs=1,
              results_df=None,
              data_dir='./data/cats/',
              val_every=25,
              device='cpu',
              save_every=100,
              save_dir='./saved_models/',
              save_name='cats'):
    
    nz = generator.nz
    train_dataloader, val_dataloader = dataloader_func(data_dir, batch_size=batch_size, val_batch_size=val_batch_size)
    
    criterion = torch.nn.BCEWithLogitsLoss()

    optimizerG = torch.optim.Adam(generator.parameters(), lr=lr_g, betas=(beta1, 0.999))
    optimizerD = torch.optim.Adam(discriminator.parameters(), lr=lr_d, betas=(beta1, 0.999))
    
    results_cols = ['i', 'type', 'epoch', 'batch', 'D_loss', 'G_loss', 'D_acc', 'G_perf']

    if results_df is None:
        results_df = pd.DataFrame(columns=results_cols)
    else:
        assert isinstance(results_df, pd.DataFrame), 'results_df must be a pandas DataFrame'
        assert results_df.columns == results_cols, 'results_df must have columns: ' + str(results_cols)
    
    i = 0

    torch.autograd.set_detect_anomaly(True)

    for epoch in range(n_epochs):
        pbar = tqdm(train_dataloader)
        for batch in pbar:
            optimizerD.zero_grad()
            optimizerG.zero_grad()

            # Real images
            real_images = next(iter(train_dataloader))
            num_images_batch = real_images.shape[0]
            real_images = real_images.to(device)
            real_labels = torch.ones((num_images_batch, 1), device=device)
            real_preds = discriminator(real_images)

            # Fake images
            noise = torch.randn((num_images_batch, nz), device=device)
            fake_images = generator(noise)
            fake_labels = torch.zeros((num_images_batch, 1), device=device)
            fake_preds = discriminator(fake_images)

            # Calculate loss
            d_loss = (criterion(real_preds, real_labels) + criterion(fake_preds, fake_labels)) / 2
            g_loss = criterion(fake_preds, real_labels)

            # Gradient descent
            print(f"d_loss: {d_loss.item():.4f}")
            print(f"g_loss: {g_loss.item():.4f}")
            d_loss.backward()
            g_loss.backward()
            optimizerD.step()
            optimizerG.step()

            # Calculate discriminator accuracy
            real_acc = torch.mean((real_preds > 0).float()).item()
            fake_acc = torch.mean((fake_preds < 0).float()).item()
            d_acc = (real_acc + fake_acc) / 2
            print(f"d_acc: {100 * d_acc:.2f}%")

            # Calculate generator performance
            g_perf = torch.mean((fake_preds > 0).float()).item()
            print(f"g_perf: {100 * g_perf:.2f}%")

            # Append results
            results_train = {
                'i': i,
                'type': 'train',
                'epoch': epoch,
                'batch': batch,
                'D_loss': d_loss.item(),
                'G_loss': g_loss.item(),
                'D_acc': d_acc * 100,
                'G_perf': g_perf * 100
            }
            results_df = results_df.append(results_train, ignore_index=True)
            pbar.set_description(f'Epoch {epoch} | D_loss: {d_loss.item():.4f} | G_loss: {g_loss.item():.4f} | D_acc: {100*d_acc:.2f}% | G_perf: {100*g_perf:.2f}%')
            
            # Save model checkpoints
            if i > 0 and i % save_every == 0:
                generator_scripted = torch.jit.script(generator)
                discriminator_scripted = torch.jit.script(discriminator)

                generator_scripted.save(save_dir + f'generators/generator-{save_name}.pt')
                discriminator_scripted.save(save_dir + f'discriminators/discriminator-{save_name}.pt')

            # Validation 
            if i > 0 and i % val_every == 0:
                with torch.no_grad():
                    # Real images
                    real_images = next(iter(val_dataloader))
                    real_images = real_images.to(device)
                    num_images_batch = real_images.shape[0]
                    real_preds = discriminator(real_images)
                    real_labels = torch.ones((num_images_batch, 1), device=device)

                    # Fake images
                    noise = torch.randn((num_images_batch, nz), device=device)
                    fake_images = generator(noise)
                    fake_preds = discriminator(fake_images)
                    fake_labels = torch.zeros((num_images_batch, 1), device=device)

                    # Calculate loss
                    d_loss = (criterion(real_preds, real_labels) + criterion(fake_preds, fake_labels)) / 2
                    g_loss = criterion(fake_preds, real_labels)

                    # Calculate discriminator accuracy
                    real_acc = torch.mean((real_preds > 0).float()).item()
                    fake_acc = torch.mean((fake_preds < 0).float()).item()
                    d_acc = (real_acc + fake_acc) / 2

                    # Calculate generator performance
                    g_perf = torch.mean((fake_preds > 0).float()).item()

                    # Append results
                    results_val = {
                        'i': i,
                        'type': 'val',
                        'epoch': epoch,
                        'batch': batch,
                        'D_loss': d_loss.item(),
                        'G_loss': g_loss.item(),
                        'D_acc': d_acc,
                        'G_perf': g_perf
                    }
                    results_df = results_df.append(results_val, ignore_index=True)
                    
            i += 1
    
    return results_df, generator, discriminator

if __name__ == "__main__":
    results, generator, discriminator = train_gan()