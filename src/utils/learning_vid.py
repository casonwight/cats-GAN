import torch
import numpy as np
import mediapy as media

def main():
    progression = torch.load("saved_models/image_progression.pt").cpu().detach().numpy().astype(np.uint8).reshape(-1, 16, 64, 64, 3)
            
    cats = [progression[:, i, :, :, :] for i in range(16)]

    video_out = np.zeros((progression.shape[0], 64*4, 64*4, 3), dtype=np.uint8)

    for i in range(4):
        for j in range(4):
            video_out[:, i*64:(i+1)*64, j*64:(j+1)*64, :] = cats[4*i + j]


    media.write_video("saved_models/progression.mp4", video_out, fps=30)

if __name__ == "__main__":
    main()