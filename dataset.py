# dataset.py
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from config import IMAGE_SIZE

class RepeatIfGray:
    def __call__(self, x):
        if x.shape[0] == 1:  # 如果是 1 channel，就 repeat 3 次變成 RGB
            return x.repeat(3, 1, 1)
        return x

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomChoice([
        transforms.RandomRotation(degrees=7),
        transforms.RandomAffine(degrees=0, scale=(0.9, 1.1)),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05))
    ]),
    transforms.ToTensor(),
    RepeatIfGray(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    RepeatIfGray(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class NIHDataset(Dataset):
    def __init__(self, df, image_root, label_cols,transform=None):
        self.df = df.reset_index(drop=True)
        self.image_root = image_root
        self.label_cols = label_cols
        self.image_path_dict = self._build_image_path_dict()
        self.transform = transform

    def _build_image_path_dict(self):
        path_dict = {}
        for i in range(1, 13):
            folder = f"images_{i:03d}/images"
            folder_path = os.path.join(self.image_root, folder)
            all_images = glob.glob(os.path.join(folder_path, "*.png"))
            for img_path in all_images:
                img_name = os.path.basename(img_path)
                path_dict[img_name] = img_path
        return path_dict
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_name = row['Image']
        img_path = self.image_path_dict.get(image_name, None)
        if img_path is None:
            raise FileNotFoundError(f"Image {image_name} not found.")
        image = Image.open(img_path).convert('L')
        if self.transform:
            image = self.transform(image)

        labels = torch.tensor(row[self.label_cols].values.astype(np.float32))
        return image, labels