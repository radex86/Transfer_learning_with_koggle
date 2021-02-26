# -*- coding: utf-8 -*-
"""VGGNet.ipynb
Original file is located at
  https://github.com/radex86/Transfer_learning_with_koggle/
Created By:
  Manar Al-Kali
"""

#importing the Important pakages
import tensorflow as tf 
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, GlobalMaxPooling2D
#from tensorflow.keras.layers import Conv2D, MaxPool2D, , BatchNormalization,
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.optimizers import Nadam, Adam, SGD
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# importing the support packages
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sb
import os, sys
from glob import glob

from google.colab import drive
drive.mount('/XXX') #'/XXX' is your google drive account directory # please modify it accodingly 


 #Import os module for navigation and environment setup
import os
# Check current location, '/content' is the Colab virtual machine
os.getcwd()
# Enable the Kaggle environment, use the path to the directory your Kaggle API JSON is stored in
# you must download your Kaggle.jason file and store it in the corresponding location
os.environ['KAGGLE_CONFIG_DIR'] = '/XXX/transferLearning/kaggle' #'/XXX' is your google drive account directory # please modify it accodingly

# do it once, if you didn't install kaggle before
#!pip install kaggle 

#checking the directory to store the dataset
os.chdir('/XXXX/transferLearning/kaggle/') #'/XXX' is your google drive account directory # please modify it accodingly

# You can choose any dataset (in this example it is the fruits-360)
!kaggle datasets download -d moltean/fruits
!unzip -qq -o fruits.zip

# this example extract 3 classes from the fruit-360 dataset, you may choose to take more according to your work
!mkdir /XXX/transferLearning/fruitz  #'/XXX' is your google drive account directory # please modify it accodingly
!mkdir /XXX/transferLearning/fruitz/train
!mkdir /XXX/transferLearning/ruitz/test

# copying the #of classes for training
!cp -r /XXX/transferLearning/kaggle/fruits-360/Training/Banana /XXX/fruitz/train/
!cp -r /XXX/transferLearning/kaggle/fruits-360/Training/Blueberry /XXX/fruitz/train/
!cp -r /XXX/MyDrive/transferLearning/kaggle/fruits-360/Training/Lemon /XXX/fruitz/train/


# copying the #of classes for test
!cp -r /XXX/transferLearning/kaggle/fruits-360/Test/Banana /XXX/fruitz/test/
!cp -r /XXX/MyDrive/transferLearning/kaggle/fruits-360/Test/Blueberry /XXX/fruitz/test/
!cp -r /XXX/MyDrive/transferLearning/kaggle/fruits-360/Test/Lemon /XXX/fruitz/test/

#checking if the processes went well
!ls /XXX/fruitz/train

train_path='/XXX/fruitz/train/'
val_path='/XXX/fruitz/test/'

# checking image demensionallities
dim1=[]
dim2=[]

for img in os.listdir(train_path +'/Banana/'):
  im = plt.imread(train_path +'/Banana/'+img)
  x,y,c = im.shape
  dim1.append(x)
  dim2.append(y)

sb.jointplot(dim1,dim2)

#getting its max and min
x_max, x_min = np.max(x), np.min(x)
y_max, y_min = np.max(y), np.min(y)

print(x_max,'X', x_min, ": ", y_max ,'X', y_min)

# decided to use 100 X 100
IMAGE_SIZE = [100, 100]

# using the glob lib to get the files info
# useful for getting number of files
image_files = glob(train_path + '/*/*.jpg')
valid_image_files = glob(val_path + '/*/*.jpg')

#of samples
N=len(image_files)

#of test samples
N_test= len(valid_image_files)

# of classes
classes=glob(train_path+ '/*')
print(classes)
K= len(classes)
print(K)

# show the image files
#checking using image load in order to check if the two classes has been loaded.
fig = plt.figure(figsize=(20,2))
for i in range(10):
  ax = plt.subplot(1,10, i + 1)
  ax.imshow(image.load_img(np.random.choice(image_files)))

# import the model 
# The Model
VGGNet = VGG16(
    input_shape = IMAGE_SIZE + [3],
    weights = 'imagenet',
    include_top=False
)

# do not re-train the VGGNet weights
VGGNet.trainable=False

# FNN structure
x= GlobalMaxPooling2D()(VGGNet.output)
x=Dense(K, activation='softmax')(x)

model = Model(VGGNet.input, x)

model.summary()

# adding data augmentation 
# create an instance of ImageDataGenerator
gen_train = ImageDataGenerator(
  rotation_range=20,
  width_shift_range=0.1,
  height_shift_range=0.1,
  shear_range=0.1,
  zoom_range=0.2,
  horizontal_flip=True,
  preprocessing_function=preprocess_input
)


gen_test = ImageDataGenerator(
  preprocessing_function=preprocess_input
)

batch_sz = 256
tr_gen = gen_train.flow_from_directory(
    train_path, 
    shuffle=True,
    target_size=IMAGE_SIZE,
    batch_size= batch_sz)

val_gen = gen_test.flow_from_directory(
    val_path,
    target_size=IMAGE_SIZE,
    batch_size=batch_sz)

model.compile(
  loss='categorical_crossentropy',
  optimizer='Nadam',
  metrics=['accuracy']
)

# fit the model
# np.ceil is used to round the devision down not up as np.round()
r = model.fit(
  tr_gen,
  validation_data=val_gen,
  epochs=10,
  steps_per_epoch=int(np.ceil(len(image_files) / batch_sz)),
  validation_steps=int(np.ceil(len(valid_image_files) / batch_sz)),)

# loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()


# accuracies
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
