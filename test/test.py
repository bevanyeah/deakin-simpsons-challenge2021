import tensorflow as tf
from tensorflow.python.client import device_lib


import sys, os, h5py
from shutil import move, copy

import numpy as np
import tensorflow as tf
from keras.preprocessing import image
from tensorflow.keras import models
from tensorflow.python.keras.saving import hdf5_format
from tensorflow.keras import layers
from tensorflow import keras

if __name__=="__main__":

    input_dir = './/input'
    output_dir = './/output'
    print("Using input_dir: " + input_dir)
    print("Using output_dir: " + output_dir)
    print(sys.version)
    print("Tensorflow version: " + tf.__version__)

    # Loading the model.
    model = 'model.h5'
    with h5py.File(model, mode='r') as f:
        class_names = f.attrs['class_names']
        image_size = f.attrs['image_size']
        model_loaded = hdf5_format.load_model_from_hdf5(f)


    files = []
    images = []
    for file in os.listdir(input_dir):
        # print(file)
        if file.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            img = image.load_img(os.path.join(input_dir, file),
                target_size=image_size)
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            images.append(x)
            files.append(file)
    images = np.vstack(images)

    # Making predictions!
    print("Making Predictions")
    batch_size = 128
    y_proba = model_loaded.predict(images, batch_size=batch_size)
    y_predict = np.argmax(y_proba,axis=1)
    y_score = np.amax(y_proba,axis=1)

    print("Finished Predictions")
    # Writing predictions to file.
    with open(os.path.join(output_dir, 'answer.txt'), 'w') as result_file:
        for i in range(len(files)):
            result_file.write(files[i] + ',' + class_names[y_predict[i]] + '\n')
            move(".\\input\\"+files[i], ".\\output\\"+class_names[y_predict[i]]+"\\"+str(int(y_score[i]*100))+"_"+files[i])

print("Processing complete")