import torch
import torch.nn as nn


class SupportCNN(nn.Module):
    """
    CNN for estimating object support from coherent diffraction intensity.
    """

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                64, 32,
                kernel_size=2,
                stride=2
            ),
            nn.ReLU(),

            nn.Conv2d(
                32, 16,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),

            nn.ConvTranspose2d(
                16, 16,
                kernel_size=2,
                stride=2
            ),
            nn.ReLU(),

            nn.Conv2d(
                16, 1,
                kernel_size=3,
                padding=1
            )
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
