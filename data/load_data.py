from torch.utils.data import DataLoader
from data.image_dataset import ImageDataset

def load_data(annotations_file_path, img_dir_path, target_transform, batch_size=64):
    data = ImageDataset(
    annotations_file=annotations_file_path,
    img_dir=img_dir_path,
    transform=None,
    target_transform=target_transform
    )

    loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    return loader