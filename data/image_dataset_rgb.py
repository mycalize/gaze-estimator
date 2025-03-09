import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import read_file, decode_image, ImageReadMode

class ImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None, grayscale=False):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.grayscale = grayscale  # Flag for grayscale images

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_label = self.img_labels.iloc[idx]  # Ensure correct row indexing
        img_path = os.path.join(self.img_dir, img_label['imagefile'])

        try:
            # Read the image file first
            img_data = read_file(img_path)

            # Decode image
            img_mode = ImageReadMode.GRAY if self.grayscale else ImageReadMode.RGB
            img = decode_image(img_data, mode=img_mode).to(torch.float32)

            # Convert grayscale to 3-channel format if needed
            if self.grayscale:
                img = img.repeat(3, 1, 1)  # Expand 1 channel to 3

        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None, None  # Returning None to avoid crashing

        label = torch.tensor([img_label['gaze_x'], img_label['gaze_y']], dtype=torch.float32)

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)

        return img, label
