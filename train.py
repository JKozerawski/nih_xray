from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import torch
import torch.nn as nn

from xray_dataset import XRayDataset
from find_dataset_statistics import get_labels
from model import XrayNet


class XrayTrainer():
    def __init__(self):
        self.batch_size = 32
        self.n_epochs = 10
        self.pretrained = False

        device_id = 2  # which gpu to use (None for CPU; 0,1,2 for GPU)
        if device_id is not None:
            self.device = torch.device("cuda:" + str(device_id))
        else:
            self.device = torch.device("cpu")


        self.dataset_path = "/data/jedrzej/medical/nih_xray/images/"
        self.csv_file = "./Data_Entry_2017_v2020.csv"
        self.train_image_list = "./train_val_list.txt"
        self.test_image_list = "./test_list.txt"

        self.init_network()
        self.init_criterions()
        self.init_dataloaders()

    def init_dataloaders(self):
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
            transforms.ToTensor(),
            transforms.Normalize([0.5319, 0.5319, 0.5319], [0.2001, 0.2001, 0.2001])
        ])
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5319, 0.5319, 0.5319], [0.2001, 0.2001, 0.2001])
        ])

        img_to_label, class_freq = get_labels(csv_file=self.csv_file)
        self.train_dataset = XRayDataset(root=self.dataset_path, file_list=self.train_image_list, label_file=img_to_label,
                              transform=train_transform)
        self.test_dataset = XRayDataset(root=self.dataset_path, file_list=self.test_image_list, label_file=img_to_label,
                                    transform=test_transform)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    def init_criterions(self):
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=15, gamma=0.1)

    def init_network(self):
        self.model = XrayNet(pretrained=self.pretrained).to(self.device)

    def train(self):
        print("Training...")
        for epoch in range(self.n_epochs):
            self.model.train()  # set it to train mode
            for batch_idx, (data, target) in enumerate(self.train_loader):
                # training part:
                data, target = data.to(self.device), target.to(self.device).float()
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                # loss = weighted_loss(target.float(), outputs, pos_weights, neg_weights, epsilon=1e-7)
                loss.backward()
                self.optimizer.step()
                acc = self.pred_acc(target, output)
                print("Training batch:", batch_idx, "Accuracy:", acc, "Loss:", round(loss.item(), 4))
            self.scheduler.step()

            self.test()

    def test(self):
            self.model.val()  # set it to train mode
            acc = 0
            for batch_idx, (data, target) in enumerate(self.test_loader):
                # training part:
                batch_size = data.size(0)
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                acc += self.pred_acc(target, output)
            print("Testing:", batch_idx, "Accuracy:", acc)

    def pred_acc(self, original, predicted):
        return torch.round(predicted).eq(original).sum().cpu().numpy() / (original.size(0)*original.size(1))


def compute_class_freqs(labels):
    """
    Compute positive and negative frequences for each class.
    Args:
        labels (np.array): matrix of labels, size (num_examples, num_classes)
    Returns:
        positive_frequencies (np.array): array of positive frequences for each
                                         class, size (num_classes)
        negative_frequencies (np.array): array of negative frequences for each
                                         class, size (num_classes)
    """
    # total number of patients (rows)
    N = len(labels)
    positive_frequencies = np.sum(labels, axis=0) / N
    negative_frequencies = 1 - positive_frequencies
    return positive_frequencies, negative_frequencies

def weighted_loss(y_true, y_pred, pos_weights, neg_weights, epsilon=1e-7):
    """
    Return weighted loss value.
    Args:
        y_true (Tensor): Tensor of true labels, size is (num_examples, num_classes)
        y_pred (Tensor): Tensor of predicted labels, size is (num_examples, num_classes)
    Returns:
        loss (Tensor): overall scalar loss summed across all classes
    """
    # initialize loss to zero
    loss = 0.0
    for i in range(len(pos_weights)):
        # for each class, add average weighted loss for that class
        loss += -(torch.mean(pos_weights[i] * y_true[:, i] * torch.log(y_pred[:, i] + epsilon) + neg_weights[i] * (1 - y_true[:, i]) * torch.log(1 - y_pred[:, i] + epsilon), axis=0))
    return loss
