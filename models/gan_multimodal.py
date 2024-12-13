# Multi-modal GAN able to compute synthesis, style transfer, and upscale

import torch
import torch.nn as nn
import torch.optim as optim

class MultiModalGAN:
    def __init__(self, generator, discriminator, mode='SYNTHESIS', latent_dim=100, lr=0.0002, beta1=0.5, pixel_size=256):
        """
        Generalized GAN framework supporting multiple modes: SYNTHESIS, STYLE_TRANSFER, and UPSCALE.

        Args:
            generator: The generator model.
            discriminator: The discriminator model.
            mode: One of 'SYNTHESIS', 'STYLE_TRANSFER', or 'UPSCALE'.
            latent_dim: Size of the latent noise vector for SYNTHESIS.
            lr: Learning rate for optimizers.
            beta1: Beta1 hyperparameter for Adam optimizer.
        """
        assert mode in ['SYNTHESIS', 'STYLE_TRANSFER', 'UPSCALE'], \
            "Mode must be 'SYNTHESIS', 'STYLE_TRANSFER', or 'UPSCALE'."
        self.generator = generator
        self.discriminator = discriminator
        self.mode = mode
        self.latent_dim = latent_dim  # Store latent_dim for SYNTHESIS mode
        self.pixel_size = pixel_size # Target Pixel_Size

        # Define optimizers
        self.opt_g = optim.Adam(self.generator.parameters(), lr=lr, betas=(beta1, 0.999))
        self.opt_d = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

        # Loss functions
        self.adversarial_loss = nn.BCELoss()  # Binary Cross Entropy for adversarial
        self.pixel_loss = nn.MSELoss()       # Mean Squared Error for pixel-level differences
        self.perceptual_loss = nn.L1Loss()   # Perceptual/content loss (can replace with a feature-based loss)

    def compute_loss(self, real_images, fake_images):
        """
        Compute the loss for the generator and discriminator.

        Args:
            real_images: Ground truth images (for discriminator).
            fake_images: Generated images (output of the generator).

        Returns:
            loss_g: Generator loss.
            loss_d: Discriminator loss.
        """
        # Labels for real and fake images
        valid = torch.ones(real_images.size(0), 1).to(real_images.device)
        fake = torch.zeros(fake_images.size(0), 1).to(fake_images.device)

        # Adversarial loss for discriminator
        real_loss = self.adversarial_loss(self.discriminator(real_images), valid)
        fake_loss = self.adversarial_loss(self.discriminator(fake_images.detach()), fake)
        loss_d = (real_loss + fake_loss) / 2

        # Adversarial loss for generator
        loss_g = self.adversarial_loss(self.discriminator(fake_images), valid)

        return loss_g, loss_d

    def train_step(self, real_images, condition_images=None):
        """
        Perform one training step for the generator and discriminator.

        Args:
            real_images: Ground truth images for discriminator training.
            condition_images: Optional conditioning images for generator input (used in style transfer/upscale).

        Returns:
            loss_g: Generator loss.
            loss_d: Discriminator loss.
        """
        # Generate fake images based on mode
        if self.mode == 'SYNTHESIS':
            noise = torch.randn(real_images.size(0), self.latent_dim, 1, 1).to(real_images.device)
            fake_images = self.generator(noise)
        elif self.mode in ['STYLE_TRANSFER', 'UPSCALE']:
            if condition_images is None:
                raise ValueError(f"Condition images required for mode '{self.mode}'")
            fake_images = self.generator(condition_images)

        # --- Train Discriminator ---
        self.opt_d.zero_grad()
        loss_g, loss_d = self.compute_loss(real_images, fake_images)
        loss_d.backward()
        self.opt_d.step()

        # --- Train Generator ---
        self.opt_g.zero_grad()
        loss_g, _ = self.compute_loss(real_images, fake_images)  # Only generator loss needed
        loss_g.backward()
        self.opt_g.step()

        return loss_g.item(), loss_d.item()

    def generate(self, batch_size):
        """
        Generate images using the generator.

        Args:
            batch_size: Number of images to generate.

        Returns:
            Generated images (torch.Tensor).
        """
        if self.mode != 'SYNTHESIS':
            raise NotImplementedError("Image generation is only supported in 'SYNTHESIS' mode")
        noise = torch.randn(batch_size, self.latent_dim, 1, 1).to(next(self.generator.parameters()).device)
        return self.generator(noise)