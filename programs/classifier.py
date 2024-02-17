import tensorflow as tf
import keras
import keras.utils
from keras import layers
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
filepath=r'C:\major\Breast-cancer-segmentation-using-Attention-Unet\normalized'
training_datasets=keras.utils.image_dataset_from_directory(filepath,validation_split=0.2,subset='training',image_size=(256,256),seed=123,batch_size=10)
validation_dataset=keras.utils.image_dataset_from_directory(filepath,validation_split=0.2,subset='validation',image_size=(256,256),seed=123,batch_size=10)

class_names = training_datasets.class_names
num_classes = len(class_names)

data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal",
                      input_shape=(256,256,3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
  ]
)

model = Sequential([
  data_augmentation,
  layers.Rescaling(1./255),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(128, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(256, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(512, activation='relu'),
  layers.Dense(num_classes, name="outputs")
])
cb=[
    ModelCheckpoint('C:\major\Breast-cancer-segmentation-using-Attention-Unet\dataset\classification.keras',save_best_only=True)]

model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
epochs=10
history = model.fit(
  training_datasets,
  validation_data=validation_dataset,
  epochs=20,
  callbacks=cb
)
