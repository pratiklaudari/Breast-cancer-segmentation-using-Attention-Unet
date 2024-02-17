import cv2
from glob import glob as gb
from tensorflow import image as tfi
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image as ii
from keras.layers import Input,Conv2D
from keras.models import Model
from keras.metrics import MeanIoU
import encoder_block
import attentation
import decoder
from keras.callbacks import EarlyStopping,ModelCheckpoint,Callback

imagepath=r'C:\major\Breast-cancer-segmentation-using-Attention-Unet\normalized_for_unet\images\*.png'
maskpath=r'C:\major\Breast-cancer-segmentation-using-Attention-Unet\normalized_for_unet\mask\*.png'
images=gb(imagepath)
masks=gb(maskpath)

SIZE=256
#loading and resizing images
def load_image(image, SIZE):
    return np.round(tfi.resize(ii.img_to_array(ii.load_img(image)) / 255., (SIZE, SIZE)), 4)


def load_images(image_paths, SIZE, mask=False, trim=None):
    if mask:
        images = np.zeros(shape=(len(image_paths), SIZE, SIZE, 1))
    else:
        images = np.zeros(shape=(len(image_paths), SIZE, SIZE, 3))

    for i, image in enumerate(image_paths):
        img = load_image(image, SIZE)
        if mask:
            images[i] = img[:, :, :1]
        else:
            images[i] = img

    return images

#showing images and masks
def show_image(image, title=None, cmap=None, alpha=1):
    plt.imshow(image, cmap=cmap, alpha=alpha)
    if title is not None:
        plt.title(title)
    plt.axis('off')
    plt.show()

def show_mask(image, mask, cmap=None, alpha=0.4):
    plt.imshow(image)
    plt.imshow(mask, cmap=cmap, alpha=alpha)
    plt.axis('off')
    plt.show()
ims = load_images(images,SIZE=SIZE)
msk = load_images(masks,mask=True,SIZE=SIZE)

class ShowProgress(Callback):
    def on_epoch_end(self, epochs, logs=None):
        image = ims
        mask = msk
        y_true = mask
        y_pred = model.predict(image)
        # Calculate Dice coefficient
        intersection = np.sum(y_true * y_pred)
        dice = (2. * intersection) / (np.sum(y_true) + np.sum(y_pred))
        print('Dice coefficient: ', dice)

        # Calculate Jaccard coefficient
        union = np.sum(y_true) + np.sum(y_pred) - intersection
        jaccard = intersection / union
        print('Jaccard coefficient: ', jaccard)

        pred_mask = self.model.predict(image[np.newaxis,...])
#attention unet
input_layer= Input(shape=(256, 256, 3))
#encoder
p1,c1=encoder_block.encoder_block(input_layer,num_filters=32)
p2,c2=encoder_block.encoder_block(p1,num_filters=64)
p3,c3=encoder_block.encoder_block(p2,num_filters=128)
p4,c4=encoder_block.encoder_block(p3,num_filters=256)

encoding=Conv2D(filters=512, kernel_size=3, padding="same", activation="relu")(p4)
#decoder
a1=attentation.att(encoding,c4,num_filters=256)
d1=decoder.decoder_block(encoding,a1,num_filters=256)

a2=attentation.att(d1,c3,num_filters=128)
d2=decoder.decoder_block(d1,a2,num_filters=128)

a3=attentation.att(d2,c2,num_filters=64)
d3=decoder.decoder_block(d2,a3,num_filters=64)

a4=attentation.att(d3,c1,num_filters=32)
d4=decoder.decoder_block(d3,a4,num_filters=32)

output_layer = Conv2D(1, kernel_size=1, activation='sigmoid', padding='same')(d4)
model = Model(inputs=[input_layer], outputs=[output_layer])
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy', MeanIoU(num_classes=2, name='IoU')]
)
#callback
cb=[
    ModelCheckpoint('C:\major\Breast-cancer-segmentation-using-Attention-Unet\dataset\Aunetresults1.keras',save_best_only=True),
    ShowProgress()]
#training
BATCH_SIZE = 8
SPE = len(images)//BATCH_SIZE

# Training
results = model.fit(
    ims, msk,
    validation_split=0.2,
    epochs=20,
    steps_per_epoch=SPE,
    batch_size=BATCH_SIZE,
    callbacks=cb
)







