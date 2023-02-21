import streamlit as st
import torch


def main():
    st.title("Cats GAN")

    # Load pytorch .pt model
    model = torch.jit.load("saved_models/generators/generator-cats.pt")

    # Generate a random cat
    if st.button("Generate a cat"):
        # Generate a random cat
        cat = torch.clamp(model(torch.randn(1, 100, 1, 1)), 0, 1)
        
        # Convert the tensor to numpy array
        cat = cat.squeeze().permute(1, 2, 0).detach().cpu().numpy()
        
        # Show the image
        st.image(cat, use_column_width=True)


if __name__=="__main__":
    main()