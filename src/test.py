import matplotlib.pyplot as plt
import numpy as np
import torch
from models.generator import Generator


def show_images():    
    with torch.no_grad():
        noise = torch.randn(16, generator.nz, 1, 1).cpu()
        fake_images = generator(noise).detach().cpu()    
        fake_images = torch.nn.functional.relu(fake_images.clone().detach().cpu().permute(0, 2, 3, 1))
        fake_images = (fake_images * 255).type(torch.uint8)

        for ax in axes.flatten():
            ax.clear()

        for i, ax in enumerate(axes.flatten()):
            ax.imshow(fake_images[i])
            ax.axis('off')

        plt.show(block=False)

        plt.pause(.25)


generator = Generator()
display_cols = 4

# Turn on interactive mode
plt.ion()

fig, axes = plt.subplots(nrows=display_cols, ncols=display_cols, figsize=(8, 8))

fig.subplots_adjust(hspace=0.1, wspace=0.1)

# Create a for loop
for i in range(10):
    show_images()

# Turn off interactive mode
plt.ioff()

# Show the final image
plt.show()