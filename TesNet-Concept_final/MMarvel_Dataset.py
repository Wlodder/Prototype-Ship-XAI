import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import os

class MARVEL_MILITARY(Dataset):
    def __init__(self, base_path, train=True, transform=None):

        self.train = train
        if train:
            self.data_frame = pd.read_csv(os.path.join(base_path, 'train.csv'), delimiter='+')
        else:
            self.data_frame = pd.read_csv(os.path.join(base_path, 'test.csv'), delimiter='+')
        self.transform = transform
        self.base_path = base_path

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # sample = self.self.data_frame.iloc[idx, :]
        image_path = self.data_frame.iloc[idx, 0]  # Assuming the first column is the image file path
        image_path = os.path.join(self.base_path, image_path)
        image = Image.open(image_path)
        image = image.convert('RGB')
        image_tensor = transforms.ToTensor()(image)
        class_no = torch.tensor(self.data_frame.iloc[idx, 1])
        cl = torch

        if self.transform:
            sample = self.transform(sample)

        return sample
