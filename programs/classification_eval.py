import tensorflow as tf
import keras
import keras.utils
from keras import layers
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
import numpy as np
from glob import glob as gb

class_names = ['benign ', 'malignant ', 'normal ']
model=keras.models.load_model('C:\major\Breast-cancer-segmentation-using-Attention-Unet\classification.keras')
img_paths=r'C:\major\Breast-cancer-segmentation-using-Attention-Unet\normalized_for_unet\images\*.png'

send=[]
receive=[]
for img_path in gb(img_paths):
    i=img_path.split('\\')[-1]
    i=i.split('(')[0]
    send.append(i)
    img=keras.utils.load_img(img_path,target_size=(256,256))
    x=keras.utils.img_to_array(img)
    prediction=model.predict(x.reshape(1,256,256,3))
    score = tf.nn.softmax(prediction[0])
    receive.append(class_names[np.argmax(score)])
b_b,b_m,b_n,m_b,m_m,m_n,n_b,n_m,n_n=0,0,0,0,0,0,0,0,0
for i in range(len(send)):
    if send[i]=='benign ':
        if receive[i]=='benign ':
            b_b=b_b+1
        elif receive[i]=='malignant ':
            b_m=b_m+1
        elif receive[i]=='normal ':
            b_n=b_n+1
    elif send[i]=='malignant ':
        if receive[i]=='benign ':
            m_b=m_b+1
        elif receive[i]=='malignant ':
            m_m=m_m+1
        elif receive[i]=='normal ':
            m_n=m_n+1
    elif send[i]=='normal ':
        if receive[i]=='benign ':
            n_b=n_b+1
        elif receive[i]=='malignant ':
            n_m=n_m+1
        elif receive[i]=='normal ':
            n_n=n_n+1
print(b_b,b_m,b_n,)
print(m_b,m_m,m_n,)
print(n_b,n_m,n_n)