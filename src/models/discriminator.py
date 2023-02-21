from torch import nn
import torch


class Discriminator(nn.Module):
  def __init__(self):
    super().__init__()

    self.layer1 = nn.Sequential(
      nn.Conv2d(3, 16, kernel_size=3, padding=1),
      nn.InstanceNorm2d(16, affine=True),
      nn.LeakyReLU(.2, inplace=False),
      nn.Conv2d(16, 64, kernel_size=3, padding=1),
      nn.InstanceNorm2d(64, affine=True),
      nn.LeakyReLU(.2, inplace=False),
      nn.MaxPool2d(2)
    )

    self.layer2 = nn.Sequential(
      nn.Conv2d(64, 128, kernel_size=3, padding=1),
      nn.InstanceNorm2d(128, affine=True),
      nn.LeakyReLU(.2, inplace=False),
      nn.Conv2d(128, 256, kernel_size=3, padding=1),
      nn.InstanceNorm2d(256, affine=True),
      nn.LeakyReLU(.2, inplace=False),
      nn.MaxPool2d(2),
    )

    self.layer3 = nn.Sequential(
      nn.Conv2d(256, 512, kernel_size=3, padding=1),
      nn.InstanceNorm2d(512, affine=True),
      nn.LeakyReLU(.2, inplace=False),
      nn.Conv2d(512, 1024, kernel_size=3, padding=1),
      nn.InstanceNorm2d(1024, affine=True),
      nn.LeakyReLU(.2, inplace=False),
      nn.MaxPool2d(2),
    )

    self.layer4 = nn.Sequential(
      nn.Conv2d(1024, 512, kernel_size=3, padding=1),
      nn.InstanceNorm2d(512, affine=True),
      nn.LeakyReLU(.2, inplace=False),
      nn.Conv2d(512, 256, kernel_size=8, padding=0),
      nn.LeakyReLU(0.2, inplace=False),
    )

    self.layer5 = nn.Sequential(
        nn.Flatten(),
        nn.Linear(256, 1, bias=False)
    )
  
  def forward(self, x):
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.layer5(x)
    return x


if __name__ == '__main__':
  import torch_directml
  gpu_device = torch_directml.device()

  model = Discriminator().to(gpu_device)
  x = torch.randn(10, 3, 64, 64).to(gpu_device)
  print(x.shape)

  y_pred = model(x)
  print(y_pred.shape)
