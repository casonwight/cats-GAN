from torch import nn
import torch


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Sequential(
          nn.Conv2d(3, 16, kernel_size=3, padding=1),
          nn.BatchNorm2d(16),
          nn.LeakyReLU(.2),
          nn.Conv2d(16, 64, kernel_size=3, padding=1),
          nn.BatchNorm2d(64),
          nn.LeakyReLU(.2),
          nn.MaxPool2d(2)
        )

        self.layer2 = nn.Sequential(
          nn.Conv2d(64, 128, kernel_size=3, padding=1),
          nn.BatchNorm2d(128),
          nn.LeakyReLU(.2),
          nn.Conv2d(128, 256, kernel_size=3, padding=1),
          nn.BatchNorm2d(256),
          nn.LeakyReLU(.2),
          nn.MaxPool2d(2),
        )

        self.layer3 = nn.Sequential(
          nn.Conv2d(256, 512, kernel_size=3, padding=1),
          nn.BatchNorm2d(512),
          nn.LeakyReLU(.2),
          nn.Conv2d(512, 1024, kernel_size=3, padding=1),
          nn.BatchNorm2d(1024),
          nn.LeakyReLU(.2),
          nn.MaxPool2d(2),
        )

        self.layer4 = nn.Sequential(
          nn.Conv2d(1024, 512, kernel_size=3, padding=1),
          nn.BatchNorm2d(512),
          nn.LeakyReLU(.2),
          nn.Conv2d(512, 256, kernel_size=8, padding=0),
          nn.BatchNorm2d(256),
          nn.LeakyReLU(.2)
        )

        self.layer5 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x


if __name__ == '__main__':
    model = Discriminator()
    print(model)

    x = torch.randn(10, 3, 64, 64)
    print(x.shape)

    y_pred = model(x)
    print(y_pred.shape)

    y_real = torch.ones(10, 1)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))
    loss = criterion(y_pred, y_real)

    loss.backward()
    optimizer.step()
