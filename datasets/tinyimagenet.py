from torch.utils.data import Dataset
import glob
import numpy as np
import os
from torchvision.datasets.folder import pil_loader
from torchvision.datasets.utils import download_and_extract_archive

class TinyImageNet(Dataset):
    def __init__(self, root, train, transforms, download=True):

        self.url = "http://cs231n.stanford.edu/tiny-imagenet-200"
        self.root = root
        if download:
            download_and_extract_archive(self.url, root, filename="tiny-imagenet-200.zip")

        self.root = os.path.join(self.root, "tiny-imagenet-200")
        self.train = train
        self.transforms = transforms
        self.ids_string = np.sort(np.loadtxt(f"{self.root}/wnids.txt", "str"))
        self.ids = {class_string: i for i, class_string in enumerate(self.ids_string)}
        if train:
            self.paths = glob.glob(f"{self.root}/train/*/images/*")
            self.label = [self.ids[path.split("/")[-3]] for path in self.paths]
        else:
            self.val_annotations = np.loadtxt(f"{self.root}/val/val_annotations.txt", "str")
            self.paths = [f"{self.root}/val/images/{sample[0]}" for sample in self.val_annotations]
            self.label = [self.ids[sample[1]] for sample in self.val_annotations]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = pil_loader(self.paths[idx])

        if self.transforms is not None:
            image = self.transforms(image)

        return image, self.label[idx]
        