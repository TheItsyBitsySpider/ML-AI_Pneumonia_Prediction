import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import cv2
import numpy as np
import scipy
import pandas as pd
import sklearn
import cv2 as cv
from tqdm.auto import tqdm
import os
from sklearn.metrics import classification_report

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

training_dataset_dir_normal = "./chest_xray/train/normal"
training_dataset_dir_pneumonia_bacterial = "./chest_xray/train/bacterial"
training_dataset_dir_pneumonia_viral = "./chest_xray/train/viral"

test_dataset_dir_normal = "chest_xray/test/normal"
test_dataset_dir_pneumonia = "./chest_xray/test/PNEUMONIA"

data_transform = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.ToTensor()
])

training_data = datasets.ImageFolder(root="./chest_xray/train/", transform=data_transform, target_transform=None)
testing_data = datasets.ImageFolder(root="./chest_xray/test/", transform=data_transform, target_transform=None)


train_dataloader = DataLoader(dataset=training_data,
                              batch_size=16,
                              num_workers=4,
                              shuffle=True)
test_dataloader = DataLoader(dataset=testing_data,
                              batch_size=16,
                              num_workers=4,
                              shuffle=True)

img, label = next(iter(test_dataloader))


torch.manual_seed(42)

CNN_kernel = 3
CNN_stride = 1
CNN_padding = 1

class CNN(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=CNN_kernel,
                      stride=CNN_stride,
                      padding=CNN_padding),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=CNN_kernel,
                      stride=CNN_stride,
                      padding=CNN_padding),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=CNN_kernel,
                      stride=CNN_stride,
                      padding=CNN_padding),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=CNN_kernel,
                      stride=CNN_stride,
                      padding=CNN_padding),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * 16 * 16,
                      out_features=output_shape)
        )

    def forward(self, x: torch.Tensor):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.classifier(x)
        return x


model = CNN(input_shape=3, hidden_units=16, output_shape=len(training_data.classes)).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(),
                             lr=0.1)

epochs = 5
torch.manual_seed(42)

for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n---------")
    all_true_labels = []
    all_pred_labels = []
    model.to(device)
    for batch, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        all_true_labels.extend(y.cpu().detach().clone().numpy())
        all_pred_labels.extend(torch.argmax(y_pred.cpu().detach(), dim=1).numpy())

    print("Training classification report:")
    print(classification_report(all_true_labels, all_pred_labels))

    model.to(device)
    model.eval()

    all_true_labels = []
    all_pred_labels = []

    with torch.inference_mode():
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)
            test_pred = model(X)
            all_true_labels.extend(y.cpu().detach().clone().numpy())
            all_pred_labels.extend(torch.argmax(test_pred.cpu().detach(), dim=1).numpy())
        print("Test classification report:")
        print(classification_report(all_true_labels, all_pred_labels))
