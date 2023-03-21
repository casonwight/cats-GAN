import streamlit as st
import torch
from torchvision import utils
import numpy as np
import mediapy as media


def main():
    st.title("Cats GAN")

    # Load pytorch .pt model
    model = torch.jit.load("saved_models/generator (4).pt", map_location=torch.device('cpu')).cpu()

    # Generate a random cat
    if st.button("Generate some cats"):
        # Generate a random cat
        cat = utils.make_grid(model(torch.randn(16, 100, 1, 1)), nrow=4, normalize=True)
        
        # Convert the tensor to numpy array
        cat = cat.squeeze().permute(1, 2, 0).detach().cpu().numpy()
        
        # Show the image
        st.image(cat, use_column_width=True)

if __name__=="__main__":
    main()