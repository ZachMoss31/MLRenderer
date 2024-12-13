import os
import shutil
import zipfile
from math import floor
from PIL import Image
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

# Custom AssetParser for folder manipulation
class AssetParser():
    def __init__(self):
        pass
    
    # Function for extracting our zip folders - saving temp directory 'extract_path' to delete later
    def load_textures_from_zip(self, zip_path):
        """
        Load images from a zip file, extract them and return the images as a list.
        """
        images = []
        extract_path = zip_path.replace('.zip', '')  # Create a folder path for extraction
        with zipfile.ZipFile(zip_path, 'r') as archive:
            archive.extractall(extract_path)
        for root, _, files in os.walk(extract_path):
            for file_name in files:
                if file_name.endswith(('.png', '.jpg', '.jpeg')):
                    file_path = os.path.join(root, file_name)
                    img = Image.open(file_path)
                    images.append(img)
        return images, extract_path
        ## Load textures for each category and save their temp directory
        # brick_textures, brick_extract_path = load_textures_from_zip("img/textures/brick.zip")
        # Remove the extracted folders now that we have our image data
        # shutil.rmtree(brick_extract_path)   
    
    def remove_excess_directory(self, zip_path):
        shutil.rmtree(zip_path)
    
    # Function to convert textures to grayscale in preparation for Local Binary extraction
    def convert_to_grayscale(textures):
        grayscale_textures = [np.array(texture.convert('L')) for texture in textures]
        return np.array(grayscale_textures)
        ## Convert to grayscale
        # brick_textures = convert_to_grayscale(brick_textures)
    
    # Function to extract features from grayscale textures
    # def extract_lbp_features(textures, radius=3, n_points=8*3):
        # from skimage.feature import local_binary_pattern
        # lbp_features = [local_binary_pattern(texture, P=n_points, R=radius, method='uniform').flatten() for texture in textures]
        # return np.array(lbp_features)
        ## Extract our features
        #brick_features = extract_lbp_features(brick_textures)

# Custom Dataset, name it TensorTextureSet()?
class TextureDataset(Dataset):
    def __init__(self, images, transform=None):
        """
        Args:
            images (list of PIL Images): List of images in PIL format.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.images = images  # List of images (already loaded in memory)
        self.transform = transform  # Any transformation (like augmentation)

    def __len__(self):
        # Return the number of images in the dataset
        return len(self.images)

    def __getitem__(self, idx):
        # Get image by index
        img = self.images[idx]
        
        # Apply any transformations (e.g., augmentations)
        if self.transform:
            img = self.transform(img)
        
        return img
    
    def display_textures(self, texture_dataloader, num_images=8):
        """
        Displays a specified number of images from the TextureDataset using matplotlib.
        """
        # Count how many images we have displayed
        images_shown = 0
        for data in texture_dataloader:
            # 'data' is now a batch of images (tensors)
            # Convert the tensor back to NumPy and change shape to HWC for plotting
            batch = data.permute(0, 2, 3, 1).numpy()

            # Plot each image in the batch
            fig, ax = plt.subplots(1, min(len(batch), num_images - images_shown), figsize=(15, 15))
            
            for i in range(min(len(batch), num_images - images_shown)):
                ax[i].imshow(batch[i])
                ax[i].axis('off')
            
            plt.show()

            # Update the number of images shown
            images_shown += len(batch)
            
            # Stop if we've displayed enough images
            if images_shown >= num_images:
                break