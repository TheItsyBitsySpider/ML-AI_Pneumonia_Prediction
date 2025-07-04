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
import matplotlib.pyplot as plt
import os
from sklearn.metrics import classification_report, accuracy_score

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

training_dataset_dir_normal = "./chest_xray/train/normal"
training_dataset_dir_pneumonia_bacterial = "./chest_xray/train/bacterial"
training_dataset_dir_pneumonia_viral = "./chest_xray/train/viral"

test_dataset_dir_normal = "chest_xray/test/normal"
test_dataset_dir_pneumonia = "./chest_xray/test/PNEUMONIA"

data_transform = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.RandomRotation(30),
    transforms.GaussianBlur(3, (.1,2)),
    transforms.ToTensor()
])

training_data = datasets.ImageFolder(root="./chest_xray/train/", transform=data_transform, target_transform=None)
testing_data = datasets.ImageFolder(root="./chest_xray/test/", transform=data_transform, target_transform=None)


train_dataloader = DataLoader(dataset=training_data,
                              batch_size=64,
                              num_workers=4,
                              shuffle=True)
test_dataloader = DataLoader(dataset=testing_data,
                              batch_size=64,
                              num_workers=4,
                              shuffle=True)
print(training_data.class_to_idx)
print(testing_data.class_to_idx)
img, label = next(iter(test_dataloader))


torch.manual_seed(42)

CNN_kernel = 3
CNN_stride = 1
CNN_padding = 1

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=64,
                      kernel_size=CNN_kernel,
                      stride=CNN_stride,
                      padding=CNN_padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,
                      out_channels=64,
                      kernel_size=CNN_kernel,
                      stride=CNN_stride,
                      padding=CNN_padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2),
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=CNN_kernel,
                      stride=CNN_stride,
                      padding=CNN_padding),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128,
                      out_channels=128,
                      kernel_size=CNN_kernel,
                      stride=CNN_stride,
                      padding=CNN_padding),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2),
        )
        self.block_3 = nn.Sequential(
            nn.Conv2d(in_channels=128,
                      out_channels=256,
                      kernel_size=CNN_kernel,
                      stride=CNN_stride,
                      padding=CNN_padding),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256,
                      out_channels=256,
                      kernel_size=CNN_kernel,
                      stride=CNN_stride,
                      padding=CNN_padding),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256,
                      out_channels=256,
                      kernel_size=CNN_kernel,
                      stride=CNN_stride,
                      padding=CNN_padding),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2),
        )
        self.block_4 = nn.Sequential(
            nn.Conv2d(in_channels=256,
                      out_channels=512,
                      kernel_size=CNN_kernel,
                      stride=CNN_stride,
                      padding=CNN_padding),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=CNN_kernel,
                      stride=CNN_stride,
                      padding=CNN_padding),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=CNN_kernel,
                      stride=CNN_stride,
                      padding=CNN_padding),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2),
        )
        self.block_5 = nn.Sequential(
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=CNN_kernel,
                      stride=CNN_stride,
                      padding=CNN_padding),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=CNN_kernel,
                      stride=CNN_stride,
                      padding=CNN_padding),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=CNN_kernel,
                      stride=CNN_stride,
                      padding=CNN_padding),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=25088,
                      out_features=4096),
            nn.Linear(in_features=4096,
                      out_features=4096),
            nn.Dropout(.5),
            nn.Linear(in_features=4096,
                      out_features=3),
            nn.Softmax()
        )

    def forward(self, x: torch.Tensor):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        x = self.classifier(x)
        return x


model = CNN().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(),
                             lr=0.00025)

epochs = 35
torch.manual_seed(42)

train_lost_list = []
train_accuracy_list = []
test_lost_list = []
test_accuracy_list = []
epoch_list_for_graph = []

for epoch in tqdm(range(epochs)):
    epoch_list_for_graph.append(epoch)
    total_train_loss = 0
    print(f"Epoch: {epoch}\n---------")
    all_true_labels = []
    all_pred_labels = []
    model.to(device)
    for batch, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        total_train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        all_true_labels.extend(y.cpu().detach().clone().numpy())
        all_pred_labels.extend(torch.argmax(y_pred.cpu().detach(), dim=1).numpy())

    print("Training classification report:")
    print(classification_report(all_true_labels, all_pred_labels))

    train_lost_list.append((total_train_loss / len(train_dataloader)))
    train_accuracy_list.append(accuracy_score(all_true_labels, all_pred_labels))

    model.to(device)
    model.eval()

    total_test_loss = 0
    all_true_labels = []
    all_pred_labels = []

    with torch.inference_mode():
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)
            test_pred = model(X)
            # print(f"raw predictions: {test_pred}")
            loss = loss_fn(test_pred, y)
            total_test_loss += loss.item()
            all_true_labels.extend(y.cpu().detach().clone().numpy())
            _, predicted = torch.max(test_pred.cpu().detach(), dim=1)
            # print(f"predicted: {predicted}")
            # print(f"actually: {y}")
            predicted = predicted.numpy()
            all_pred_labels.extend(predicted)
        print("Test classification report:")
        print(classification_report(all_true_labels, all_pred_labels))
        test_lost_list.append((total_test_loss / len(test_dataloader)))
        test_accuracy_list.append(accuracy_score(all_true_labels, all_pred_labels))


plt.figure(figsize=(15, 7))

# Plot loss
plt.subplot(1, 2, 1)
plt.plot(epoch_list_for_graph, train_lost_list, label='train_loss')
plt.plot(epoch_list_for_graph, test_lost_list, label='test_loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.legend()

# Plot accuracy
plt.subplot(1, 2, 2)
plt.plot(epoch_list_for_graph, train_accuracy_list, label='train_accuracy')
plt.plot(epoch_list_for_graph, test_accuracy_list, label='test_accuracy')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.legend()

plt.savefig("losscurve.png")