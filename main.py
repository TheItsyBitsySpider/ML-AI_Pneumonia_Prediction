import gc
from os.path import isfile

import cv2
import numpy as np
import scipy
import pandas as pd
import sklearn
import cv2 as cv
import os

from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB, CategoricalNB
from sklearn.metrics import classification_report

# Need to preprocess data first
def preprocess(files_to_process, path, label):
    processed_files = []
    labels = []
    for file in files_to_process:
        img = cv2.imread(os.path.join(path, file), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print("Found null image, skipping...")
            continue
        # First, resize image for consistency
        resized = cv2.resize(img, (180, 180), interpolation=cv2.INTER_LINEAR)
        # Reshape to fit with model
        flattened_img = resized.flatten()
        processed_files.append(flattened_img)
        labels.append(label)
    return (processed_files, labels)


def get_batch(batch_size, files, path, label):
    batch = []
    labels = []
    while len(files) > 0:
        file = files.pop()
        img = cv2.imread(os.path.join(path, file), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print("Found null image, skipping...")
            continue
        # First, resize image for consistency
        resized = cv2.resize(img, (1080, 1080), interpolation=cv2.INTER_LANCZOS4)
        # Improve image contrast
        clahe = cv2.createCLAHE(clipLimit=3)
        resized = np.clip(clahe.apply(resized)+15, 0, 255)
        # Reshape to fit with model
        flattened_img = resized.flatten()
        batch.append(flattened_img)
        labels.append(label)
        if len(batch) >= batch_size:
            break
    return (batch, labels)


training_dataset_dir_normal = "./chest_xray/train/NORMAL"
training_dataset_dir_pneumonia = "./chest_xray/train/PNEUMONIA"

test_dataset_dir_normal = "./chest_xray/test/NORMAL"
test_dataset_dir_pneumonia = "./chest_xray/test/PNEUMONIA"

all_files = [file for file in os.listdir(training_dataset_dir_normal) if isfile(os.path.join(training_dataset_dir_normal, file))]
all_pneumonia_virus_files = [file for file in os.listdir(training_dataset_dir_pneumonia) if (isfile(os.path.join(training_dataset_dir_pneumonia, file)) and "virus" in file)]
all_pneumonia_bacterial_files = [file for file in os.listdir(training_dataset_dir_pneumonia) if (isfile(os.path.join(training_dataset_dir_pneumonia, file)) and "bacteria" in file)]

test_normal_files = [file for file in os.listdir(test_dataset_dir_normal) if isfile(os.path.join(test_dataset_dir_normal, file))]
test_pneumonia_virus_files = [file for file in os.listdir(test_dataset_dir_pneumonia) if (isfile(os.path.join(test_dataset_dir_pneumonia, file)) and "virus" in file)]
test_pneumonia_bacterial_files = [file for file in os.listdir(test_dataset_dir_pneumonia) if (isfile(os.path.join(test_dataset_dir_pneumonia, file)) and "bacteria" in file)]

BATCH_SIZE = 16

gnb = GaussianNB()
while len(all_files) > 0:
    batch, labels = get_batch(BATCH_SIZE, all_files, training_dataset_dir_normal, 0)
    gnb.partial_fit(batch, labels, classes=[0, 1, 2])

while len(all_pneumonia_virus_files) > 0:
    batch, labels = get_batch(BATCH_SIZE, all_pneumonia_virus_files, training_dataset_dir_pneumonia, 1)
    gnb.partial_fit(batch, labels)

while len(all_pneumonia_bacterial_files) > 0:
    batch, labels = get_batch(BATCH_SIZE, all_pneumonia_bacterial_files, training_dataset_dir_pneumonia, 2)
    gnb.partial_fit(batch, labels)


all_test_labels = []
y_pred = []

while len(test_normal_files) > 0:
    batch, labels = get_batch(BATCH_SIZE, test_normal_files, test_dataset_dir_normal, 0)
    y_pred.extend(gnb.predict(batch))
    all_test_labels.extend(labels)

while len(test_pneumonia_virus_files) > 0:
    batch, labels = get_batch(BATCH_SIZE, test_pneumonia_virus_files, test_dataset_dir_pneumonia, 1)
    y_pred.extend(gnb.predict(batch))
    all_test_labels.extend(labels)

while len(test_pneumonia_bacterial_files) > 0:
    batch, labels = get_batch(BATCH_SIZE, test_pneumonia_bacterial_files, test_dataset_dir_pneumonia, 2)
    y_pred.extend(gnb.predict(batch))
    all_test_labels.extend(labels)

print(classification_report(all_test_labels, y_pred))
