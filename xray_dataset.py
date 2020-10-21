from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class XRayDataset(Dataset):

    def __init__(self, root, file_list, label_file, transform=None):
        self.labels = []
        self.paths = []
        self.transform = transform

        f = open(file_list, "r")
        lines = f.readlines()
        f.close()

        for line in lines:
            line = line.strip()
            self.paths.append(root+line)
            self.labels.append(label_file[line])
        self.labels = np.asarray(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        path = self.paths[index]
        label = self.labels[index]

        with open(path, 'rb') as f:
            image = Image.open(f).convert('RGB')    # .convert('L') convert to grayscale
        if self.transform is not None:
            image = self.transform(image)
        return image, label