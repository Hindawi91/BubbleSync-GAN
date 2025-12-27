import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"  # specify which GPU(s) to be used



import numpy as np

from tensorflow import keras
from tensorflow.keras.preprocessing import image
from glob import glob                                                           

import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score, balanced_accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix,roc_auc_score

import datetime

All_Models = []
All_Metrics = []
All_CMs = []
expirements = ["exp1"]


dataset = "DS3"
test_dir = f"../../data/{dataset}/{dataset}_CNN_DATA/test"
best_epoch = "88"

for exp in expirements:   

    models_names = glob(f'*{best_epoch}.hdf5*')

    categories = ['pre_CHF','post_CHF']
    images = []
    y_true = []
    
    print(f"getting data for {exp} ......")
    
    for j,category in enumerate (categories): 
        # im_files = glob(f'./GT_binary_balanced_splitted/test/{category}/*.j*')
        im_files = glob(f'{test_dir}/{category}/*.j*')

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
            
    for model_name in models_names:
        
        begin_time = datetime.datetime.now()
        
        model = keras.models.load_model(f"./{model_name}")
        
        print (f"predicting using model {model_name} .......")
        
        imagesNP = np.vstack(images)
        y_pred = model.predict(imagesNP)
        # y_pred_prob = model.predict_proba(imagesNP)[:,1]
        y_pred_prob = y_pred[:,1]
        y_pred = np.argmax(y_pred,axis=1)
        
        
        acc = accuracy_score(y_true, y_pred)
        balanced_acc = balanced_accuracy_score(y_true, y_pred)

        f1_none = f1_score(y_true, y_pred, average=None)
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_micro = f1_score(y_true, y_pred, average='micro')
        f1_weighted = f1_score(y_true, y_pred, average='weighted')

        precision_none = precision_score(y_true, y_pred, average=None)
        precision_macro = precision_score(y_true, y_pred, average='macro')
        precision_micro = precision_score(y_true, y_pred, average='micro')
        precision_weighted = precision_score(y_true, y_pred, average='weighted')

        recall_none = recall_score(y_true, y_pred, average=None)
        recall_macro = recall_score(y_true, y_pred, average='macro')
        recall_micro = recall_score(y_true, y_pred, average='micro')
        recall_weighted = recall_score(y_true, y_pred, average='weighted')

        ROC_AUC_ovr = roc_auc_score(y_true, y_pred_prob)
        ROC_AUC_ovo = roc_auc_score(y_true, y_pred_prob)
        
        CM = confusion_matrix(y_true, y_pred)
        
        testing_time = datetime.datetime.now() - begin_time

        metrics = [model_name,acc,balanced_acc,f1_none,f1_macro,f1_micro,f1_weighted,precision_none,
                    precision_macro,precision_micro,precision_weighted,recall_none,
                    recall_macro,recall_micro,recall_weighted,ROC_AUC_ovr,ROC_AUC_ovo,testing_time]
        metrics_names = ["Model_name","Accuracy","Balanced Accuracy","F1_none","F1_macro",
                          "F1_micro","F1_weighted","Precision_none",
                          "Precision_macro","Precision_micro",
                          "Precision_weighted","Recall_none",
                          "Recall_macro","Recall_micro",
                          "Recall_weighted","ROC_AUC_ovr","ROC_AUC_ovo","testing_time"]

        print (CM)
        All_Metrics.append(metrics)
        All_CMs.append(CM)

df = pd.DataFrame(All_Metrics,columns=metrics_names)

df.to_excel (f'./CNN {dataset} on {dataset} - Base Model_Metrics.xlsx', index = False, header=True)

print(df.shape)

df = pd.DataFrame([All_CMs])

df.to_excel (f'./CNN {dataset} on {dataset} - Base Model_CMs.xlsx', index = False, header=True)

print(df.shape)

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

final.to_excel (f'./CNN {dataset} on {dataset} - Base Model_CMs2.xlsx', index = False, header=True)

