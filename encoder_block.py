import torchvision.datasets as datasets
from torchvision import transforms
import torch
import cv2
from glob import glob as gb
from keras.layers import Conv2D,BatchNormalization,Activation,MaxPool2D,Conv2DTranspose,Concatenate
data_dir = ''#get normalised dataset
dataset=gb(data_dir)

def convolution_block(image,nfilter):
    x=Conv2D(nfilter,3,padding="same")(image)
    x=BatchNormalization()(x)
    x= Activation('relu')(x)

    x=Conv2D(nfilter,3,padding="same")(x)
    x=BatchNormalization()(x)
    x= Activation('relu')(x)

    return x
def encoder_block(input,nfilter):
    x=convolution_block(input,nfilter)
    p=MaxPool2D(2,2)(x)
    return x,p

for image in dataset:
    img=cv2.imread(image)
    Tensor_img=transforms.Compose([transforms.ToTensor()])
    image_tensor=Tensor_img(img)
