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
from sklearn.metrics import accuracy_score

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

processed_training_normal, label_normal = preprocess(all_files, training_dataset_dir_normal, 0)
processed_training_pneumonia_viral, label_viral = preprocess(all_pneumonia_virus_files, training_dataset_dir_pneumonia, 1)
processed_training_pneumonia_bacterial, label_bacterial = preprocess(all_pneumonia_bacterial_files, training_dataset_dir_pneumonia, 2)

processed_test_normal, test_label_normal = preprocess(test_normal_files, test_dataset_dir_normal, 0)
processed_test_pneumonia_viral, test_label_viral = preprocess(test_pneumonia_virus_files, test_dataset_dir_pneumonia, 1)
processed_test_pneumonia_bacterial, test_label_bacterial = preprocess(test_pneumonia_bacterial_files, test_dataset_dir_pneumonia, 2)


processed_training_normal.extend(processed_training_pneumonia_viral)
processed_training_normal.extend(processed_training_pneumonia_bacterial)
label_normal.extend(label_viral)
label_normal.extend(label_bacterial)


processed_test_normal.extend(processed_test_pneumonia_viral)
processed_test_normal.extend(processed_test_pneumonia_bacterial)
test_label_normal.extend(test_label_viral)
test_label_normal.extend(test_label_bacterial)

print(processed_training_normal)
print(label_normal)



gnb = GaussianNB()
gnb.fit(processed_training_normal, label_normal)
y_pred = gnb.predict(processed_test_normal)

print(accuracy_score(test_label_normal, y_pred))
