import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image


class XrayDataset(Dataset):
    """Chest x-rays of viral, bacterial, pneumonia and normal lungs"""

    def __init__(self, root_dir, transforms_list):
        self.images = []
        self.tforms = transforms.Compose(transforms_list)
        pneumonia_dir = os.path.join(root_dir, "PNEUMONIA")
        normal_dir = os.path.join(root_dir, "NORMAL")
        for fname in os.listdir(pneumonia_dir):
            pathogen = fname.split('_')[1]  # this gets the middle label, bacterial, viral
            if "bacteria" in pathogen:
                pathogen = 1
            else:
                pathogen = 2

            img = {"image": os.path.join(pneumonia_dir, fname), "label": pathogen}
            self.images.append(img)
        for fname in os.listdir(normal_dir):
            img = {"image": os.path.join(normal_dir, fname), "label": 0}
            self.images.append(img)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]["image"]).convert('RGB')
        img = self.tforms(img)
        label = self.images[idx]["label"]
        return {"image": img, "label": label}
