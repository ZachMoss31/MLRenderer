import torch
import torch.nn as nn

class MultiModalGenerator(nn.Module):
    def __init__(self, mode='SYNTHESIS', pixel_size=256):
        """
        Initialize the generator.
        
        Args:
            mode: 'SYNTHESIS', 'STYLE_TRANSFER', or 'UPSCALE'.
        """
        super(MultiModalGenerator, self).__init__()
        assert mode in ['SYNTHESIS', 'STYLE_TRANSFER', 'UPSCALE'], \
            "Mode must be 'SYNTHESIS', 'STYLE_TRANSFER', or 'UPSCALE'."
        self.mode = mode
        self.pixel_size = pixel_size

        if mode == 'SYNTHESIS':
            self.model = self._synthesis_generator()
        elif mode == 'STYLE_TRANSFER':
            self.model = self._style_generator()
        elif mode == 'UPSCALE':
            self.model = self._upscale_generator()

    def forward(self, x):
        return self.model(x)

    def _synthesis_generator(self):
        """
        Define the generator for SYNTHESIS mode.
        Takes noise as input and generates an image.
        """
        # Dynamically calculate number of layers required based on pixel size
        layers = []
        current_size = 4  # Start with a 4x4 image after the first transposed convolution layer
        latent_dim = 100
        
        # Calculate the number of layers needed to match the pixel_size
        while current_size < self.pixel_size:
            # Add a transposed convolution layer
            layers.append(nn.ConvTranspose2d(latent_dim if latent_dim == 100 else 512, 512, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(512))
            layers.append(nn.ReLU(True))
            latent_dim = 512  # Update the number of input channels after the first layer
            current_size *= 2  # Double the size of the image after each layer

        # Final layer to generate the desired output size
        layers.append(nn.ConvTranspose2d(512, 3, kernel_size=4, stride=2, padding=1, bias=False))
        layers.append(nn.Tanh())  # Output image with values in [-1, 1]

        return nn.Sequential(*layers)

    def _style_generator(self):
        """
        Define the generator for STYLE_TRANSFER mode.
        Takes a content image as input and applies the learned style.
        """
        return nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=4),
            nn.Tanh()  # Output image with values in [-1, 1]
        )

    def _upscale_generator(self):
        """
        Define the generator for UPSCALE mode.
        Takes a low-resolution image and outputs a high-resolution image.
        """
        return nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),  # Upscaling by 2
            nn.ReLU(True),
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),  # Upscaling by 2
            nn.ReLU(True),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()  # Output image with values in [-1, 1]
        )
