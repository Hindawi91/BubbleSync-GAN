import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"  # specify which GPU(s) to be used

import timeit
import datetime
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

import numpy as np

dataset = "DS3"
batch_size = 64
epochs = 100



train_data_path = f"../../data/{dataset}/{dataset}_CNN_DATA/train"
test_data_path = f"../../data/{dataset}/{dataset}_CNN_DATA/val"
# test_dir = f"../../data/{dataset}/{dataset}_CNN_DATA/test"


def save_plots(dataset):
    
    fig, axs = plt.subplots(1, 2)

    axs[0].plot(history.history['accuracy'])
    axs[0].plot(history.history['val_accuracy'])
    axs[0].set_title('Accuracy')
    axs[0].set(xlabel='Epoch', ylabel='Accuracy')
    axs[0].legend(['train', 'val'], loc='center right')

    axs[1].plot(history.history['loss'])
    axs[1].plot(history.history['val_loss'])
    axs[1].set_title('Loss')
    axs[1].set(xlabel='Epoch', ylabel='Loss')
    axs[1].legend(['train', 'val'], loc='center right')

    plt.suptitle(f'CNN {dataset} - Base Model')
    plt.savefig(f'CNN {dataset} - Base Model.png',figsize=(20,10))
    
def save_epoch_vs_metrics (dataset):
    
    training_acc = history.history['accuracy']
    validation_acc = history.history['val_accuracy']
    training_loss = history.history['loss']
    validation_loss = history.history['val_loss']
    epochs = []
    for i in range(len(training_acc)):

        epochs.append(i+1) 


    metrics = np.concatenate(([epochs], [training_acc], [validation_acc],[training_loss], [validation_loss]), axis=0)
    metrics = np.transpose(metrics)

    DataFrame = pd.DataFrame(metrics,columns=['epochs','training_acc','validation_acc','training_loss','validation_loss'])

    DataFrame.to_csv(f'CNN {dataset} - Base Model_epoch_vs_metrics.csv',index=False)
    
expirements = [""]
times =[]

base_model = ResNet50(weights= None, include_top=False,)

# base_model.summary()


x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(2, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)



for exp in expirements:

    # train_data_path = f"./HanData_resized_splitted/train"
    # test_data_path = f"./HanData_resized_splitted/val"
    # img_rows = 128
    # img_cols = 72
    # epochs = 100
    # batch_size =16
    
    # train1 = len(os.listdir(f"{train_data_path}/ONB"))
    train1 = len(os.listdir(f"{train_data_path}/pre_CHF"))
    train2 = len(os.listdir(f"{train_data_path}/post_CHF"))
    
    # val1 = len(os.listdir(f"{test_data_path}/ONB"))
    val1 = len(os.listdir(f"{test_data_path}/pre_CHF"))
    val2 = len(os.listdir(f"{test_data_path}/post_CHF"))
    
    num_of_train_samples = train1 + train2  
    num_of_test_samples = val1 + val2 
    
    print(num_of_train_samples)
    print(num_of_test_samples)
    #Image Generator
    train_datagen = ImageDataGenerator(
    rescale = 1./255,
    width_shift_range=0.25,
    height_shift_range=0.25,
    horizontal_flip=True,
    brightness_range=[0.2,1.0],
    zoom_range=[0.5,1.0],
    dtype='float32')

    test_datagen = ImageDataGenerator(
    rescale = 1./255,
    dtype='float32')

    train_generator = train_datagen.flow_from_directory(train_data_path,
                                                        #target_size=(img_rows, img_cols),
                                                        batch_size=batch_size,
                                                        # color_mode = "grayscale",
                                                        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(test_data_path,
                                                            #target_size=(img_rows, img_cols),
                                                            batch_size=batch_size,
                                                            # color_mode = "grayscale",
                                                            class_mode='categorical')
    print(validation_generator.class_indices)
    print(validation_generator.classes)


    model.compile(loss="categorical_crossentropy",
                  optimizer="Adam",
                  metrics=['accuracy'])
    
    
    checkpoint = ModelCheckpoint(filepath=f'./CNN {dataset} Binary - Base Model - ' + 'epoch {epoch:02d}.hdf5', 
                                   monitor = 'val_loss',
                                   verbose=1, 
                                   save_best_only=True)
    callbacks_list = [checkpoint]

#     model.summary()
    
    begin_time = datetime.datetime.now()

    history = model.fit_generator(train_generator,
                        callbacks=callbacks_list,
                        steps_per_epoch=num_of_train_samples // batch_size,
                        epochs=epochs,
                        validation_data=validation_generator,
                        validation_steps=num_of_test_samples // batch_size)

    training_time = datetime.datetime.now() - begin_time
    print(f"CNN training time for {dataset} = {training_time}")
    times.append(str(training_time))
    
    save_plots(dataset)
    save_epoch_vs_metrics(dataset)


df = pd.DataFrame([times])

df.to_excel (f'CNN {dataset} - Base Model.xlsx', index = False, header=True)






























# model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# # train the model on the new data for a few epochs
# # model.fit(...)

# # let's visualize layer names and layer indices to see how many layers
# # we should freeze:
# for i, layer in enumerate(base_model.layers):
#    print(i, layer.name)