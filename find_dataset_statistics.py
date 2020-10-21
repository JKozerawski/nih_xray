from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import csv
import numpy as np

from xray_dataset import XRayDataset

DATASET_PATH = "/data/jedrzej/medical/nih_xray/images/"


def get_labels_list():
    labels_list = ['Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema', 'Emphysema', 'Fibrosis',
                   'Effusion', 'Pneumonia', 'Pleural_Thickening', 'Cardiomegaly', 'Nodule', 'Mass', 'Hernia']
    return labels_list


def get_labels(csv_file):
    img_to_label = dict()

    labels_list = get_labels_list()

    class_freq = np.zeros(len(labels_list))

    with open(csv_file, newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        next(csvreader)     # skip the header
        for row in csvreader:
            label = np.zeros(len(labels_list))
            for i in range(len(labels_list)):
                if labels_list[i] in row[1]:
                    label[i] = 1
            img_to_label[row[0]] = label
            class_freq += label
    return img_to_label, class_freq


def get_dataset_mean_stddev(batch_size, recalculate=False):
    if recalculate:
        batch_size = batch_size
        transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor()
            ])
        img_to_label, _ = get_labels(csv_file="./Data_Entry_2017_v2020.csv")
        dataset = XRayDataset(root=DATASET_PATH, file_list="./train_val_list.txt", label_file=img_to_label, transform=transform)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        mean = 0.
        std = 0.
        for images, _ in loader:
            batch_samples = images.size(0)  # batch size (the last batch can have smaller size!)
            images = images.view(batch_samples, images.size(1), -1)
            mean += images.mean(2).sum(0)
            std += images.std(2).sum(0)

        mean /= len(loader.dataset)
        std /= len(loader.dataset)
    else:
        mean, std = 0.5319, 0.2001
    return mean, std


