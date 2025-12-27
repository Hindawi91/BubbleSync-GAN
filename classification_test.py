import os
import argparse
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"  # specify which GPU(s) to be used
import keras
import cv2
import tensorflow as tf
import numpy as np
import numpy as np
from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing import image
from glob import glob                                                           
import cv2 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, balanced_accuracy_score
from sklearn.metrics import confusion_matrix,roc_auc_score
import timeit
import datetime

base_DS_OS = "DS2_OS_CNN_grey"
base_DS = "DS2"


All_Models = []
All_Metrics = []
All_CMs = []


model_name = f"./CNN DS2 Binary - Base Model - epoch 70.hdf5"
print(model_name)


for exp in range(10000,300001,10000):

    categories = ['pre_CHF','post_CHF']
    images = []
    y_true = []
    
    print(f"getting data for {exp} ......")
    
    for j,category in enumerate (categories): 
        im_files = glob(f'./Boiling/results_{exp}/{category}/*.j*')

        for i,im_file in enumerate (im_files):
            
            if category == 'post_CHF':
                y_true.append(0)
            elif category == 'pre_CHF':
                y_true.append(1)
            

            # img1 = image.load_img(im_file,target_size=(128, 72))
            img1 = image.load_img(im_file)
            img1 = image.img_to_array(img1)
            img1 = np.expand_dims(img1, axis=0)
            img1 /= 255.
            images.append(img1)
            
        
    begin_time = datetime.datetime.now()

    print(y_true)
    
    model = keras.models.load_model(model_name)
    
    print (f"predicting using model {model_name} on results from GAN{exp}.......")
    
    imagesNP = np.vstack(images)
    y_pred = model.predict(imagesNP)
    # y_pred_prob = model.predict_proba(imagesNP)[:,1]
    y_pred_prob = y_pred[:,1]
    y_pred = np.argmax(y_pred,axis=1)

    # acc = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    
    # f1_none = f1_score(y_true, y_pred, average=None)
    # f1_macro = f1_score(y_true, y_pred, average='macro')
    # f1_micro = f1_score(y_true, y_pred, average='micro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')

    # precision_none = precision_score(y_true, y_pred, average=None)
    # precision_macro = precision_score(y_true, y_pred, average='macro')
    # precision_micro = precision_score(y_true, y_pred, average='micro')
    precision_weighted = precision_score(y_true, y_pred, average='weighted')

    # recall_none = recall_score(y_true, y_pred, average=None)
    # recall_macro = recall_score(y_true, y_pred, average='macro')
    # recall_micro = recall_score(y_true, y_pred, average='micro')
    recall_weighted = recall_score(y_true, y_pred, average='weighted')

    ROC_AUC_ovr = roc_auc_score(y_true, y_pred_prob)
    ROC_AUC_ovo = roc_auc_score(y_true, y_pred_prob)
    
    CM = confusion_matrix(y_true, y_pred)
    
    testing_time = datetime.datetime.now() - begin_time
    
    
    # metrics = [model_name,exp,acc,balanced_acc,f1_none,f1_macro,f1_micro,f1_weighted,precision_none,
    #             precision_macro,precision_micro,precision_weighted,recall_none,
    #             recall_macro,recall_micro,recall_weighted,ROC_AUC_ovr,ROC_AUC_ovo,testing_time]
    # metrics_names = ["Model_name","GAN Model","Accuracy","Balanced Accuracy","F1_none","F1_macro",
    #                   "F1_micro","F1_weighted","Precision_none",
    #                   "Precision_macro","Precision_micro",
    #                   "Precision_weighted","Recall_none",
    #                   "Recall_macro","Recall_micro",
    #                   "Recall_weighted","ROC_AUC_ovr","ROC_AUC_ovo","testing_time"]

    metrics = [exp,balanced_acc,f1_weighted,precision_weighted,recall_weighted,ROC_AUC_ovr,ROC_AUC_ovo]
    metrics_names = ["GAN Model","Balanced Accuracy","F1_weighted","Precision_weighted","Recall_weighted","ROC_AUC_ovr","ROC_AUC_ovo"]

#         ALL_Models.appeand(model_name)
    print (CM)
    All_Metrics.append(metrics)
    All_CMs.append(CM)

#Send Metrics To Excel Sheet

df = pd.DataFrame(All_Metrics,columns=metrics_names)

df.to_excel (f'./Val_{base_DS}_Base Model_Metrics.xlsx', index = False, header=True)

print(df.shape)


# In[30]:


df = pd.DataFrame([All_CMs])

df.to_excel (f'./Val_{base_DS}_Base Model_CMs.xlsx', index = False, header=True)

print(df.shape)


# In[36]:


frames = []

for cm in All_CMs:
    df = pd.DataFrame(cm)
    frames.append(df)

final = pd.concat(frames)
frames = []

for cm in All_CMs:
    df = pd.DataFrame(cm)
    frames.append(df)

final = pd.concat(frames)

final.to_excel (f'./Val_{base_DS}_Base Model_CMs2.xlsx', index = False, header=True)


# In[ ]:





