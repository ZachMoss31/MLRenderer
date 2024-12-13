import torch
import torch.nn as nn

class MultiModalDiscriminator(nn.Module):
    def __init__(self, mode='SYNTHESIS'):
        """
        Initialize the discriminator.
        
        Args:
            mode: 'SYNTHESIS', 'STYLE_TRANSFER', or 'UPSCALE'.
        """
        super(MultiModalDiscriminator, self).__init__()
        assert mode in ['SYNTHESIS', 'STYLE_TRANSFER', 'UPSCALE'], \
            "Mode must be 'SYNTHESIS', 'STYLE_TRANSFER', or 'UPSCALE'."
        self.mode = mode

        if mode == 'SYNTHESIS':
            self.model = self._synthesis_discriminator()
        elif mode == 'STYLE_TRANSFER':
            self.model = self._style_discriminator()
        elif mode == 'UPSCALE':
            self.model = self._upscale_discriminator()

        # Output layer for binary classification (real or fake)
        self.fc = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1),
            nn.Sigmoid()  # Output a probability
        )

    def forward(self, x):
        features = self.model(x)
        features_flat = features.view(features.size(0), -1)  # Flatten for the fully connected layer
        return self.fc(features_flat)

    def _synthesis_discriminator(self):
        """
        Define the discriminator for SYNTHESIS mode.
        Typically evaluates generated images for realism.
        """
        return nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def _style_discriminator(self):
        """
        Define the discriminator for STYLE_TRANSFER mode.
        Evaluates the quality of styled images compared to real style samples.
        """
        return nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def _upscale_discriminator(self):
        """
        Define the discriminator for UPSCALE mode.
        Evaluates high-resolution images against ground truth.
        """
        return nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
