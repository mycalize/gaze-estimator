import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import decode_image, ImageReadMode
from torchvision.transforms.functional import convert_image_dtype

class ImageDataset(Dataset):
  def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
    self.img_labels = pd.read_csv(annotations_file)
    self.img_dir = img_dir
    self.transform = transform
    self.target_transform = target_transform

  def __len__(self):
    return len(self.img_labels)
  
  def __getitem__(self, idx):
    img_label = self.img_labels.loc[idx]
    img_path = os.path.join(self.img_dir, img_label['imagefile'])
    img = convert_image_dtype(decode_image(img_path, mode=ImageReadMode.GRAY), dtype=torch.float32)
    label = torch.tensor([img_label['gaze_x'], img_label['gaze_y']], dtype=torch.float32)
    if self.transform:
      img = self.transform(img)
    if self.target_transform:
      label = self.target_transform(label)
    return img, label