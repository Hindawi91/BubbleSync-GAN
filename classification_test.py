import os
import argparse
import cv2
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from sklearn.metrics import classification_report, confusion_matrix
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, balanced_accuracy_score
from sklearn.metrics import confusion_matrix, roc_auc_score
import timeit
import datetime
import keras

base_DS_OS = "DS2_OS_CNN_grey"
base_DS = "DS2"


All_Models = []
All_Metrics = []
All_CMs = []


model_name = f"./CNN DS2 Binary - Base Model - epoch 70.keras"
print(model_name)


for exp in range(10000, 300001, 10000):

    categories = ['pre_CHF', 'post_CHF']
    images = []
    y_true = []

    print(f"getting data for {exp} ......")

    for j, category in enumerate(categories):
        im_files = glob(f'./Boiling/results_{exp}/{category}/*.j*')

        for i, im_file in enumerate(im_files):

            if category == 'post_CHF':
                y_true.append(0)
            elif category == 'pre_CHF':
                y_true.append(1)

            img1 = image.load_img(im_file)
            img1 = image.img_to_array(img1)
            img1 = np.expand_dims(img1, axis=0)
            img1 /= 255.
            images.append(img1)

    begin_time = datetime.datetime.now()

    print(y_true)

    model = keras.models.load_model(model_name)

    print(f"predicting using model {model_name} on results from GAN{exp}.......")

    imagesNP = np.vstack(images)
    y_pred = model.predict(imagesNP)
    y_pred_prob = y_pred[:, 1]
    y_pred = np.argmax(y_pred, axis=1)

    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    precision_weighted = precision_score(y_true, y_pred, average='weighted')
    recall_weighted = recall_score(y_true, y_pred, average='weighted')

    ROC_AUC_ovr = roc_auc_score(y_true, y_pred_prob)
    ROC_AUC_ovo = roc_auc_score(y_true, y_pred_prob)

    CM = confusion_matrix(y_true, y_pred)

    testing_time = datetime.datetime.now() - begin_time

    metrics = [exp, balanced_acc, f1_weighted, precision_weighted, recall_weighted, ROC_AUC_ovr, ROC_AUC_ovo]
    metrics_names = ["GAN Model", "Balanced Accuracy", "F1_weighted", "Precision_weighted", "Recall_weighted", "ROC_AUC_ovr", "ROC_AUC_ovo"]

    print(CM)
    All_Metrics.append(metrics)
    All_CMs.append(CM)

# Send Metrics To Excel Sheet

df = pd.DataFrame(All_Metrics, columns=metrics_names)

df.to_excel(f'./Val_{base_DS}_Base Model_Metrics.xlsx', index=False, header=True)

print(df.shape)

df = pd.DataFrame([All_CMs])

df.to_excel(f'./Val_{base_DS}_Base Model_CMs.xlsx', index=False, header=True)

print(df.shape)

frames = []

for cm in All_CMs:
    df = pd.DataFrame(cm)
    frames.append(df)

final = pd.concat(frames)

final.to_excel(f'./Val_{base_DS}_Base Model_CMs2.xlsx', index=False, header=True)
