from keras.preprocessing import image
import numpy as np

def normalize_image(image_path):
  # Load image
  img = image.load_img(image_path)
  img_array = image.img_to_array(img)

  # Normalize by subtracting mean and dividing by standard deviation
  mean = np.average(img_array)
  sd= np.std(img_array)
  img_array -= mean
  img_array /= sd

  return img_array

def save_image(image_array, output_path):
  # Convert the image array back to PIL image format
  img = image.array_to_img(image_array)

  # Save the image to the desired path
  img.save(output_path)
#usage
def normalize(image_path, output_path):
  n=normalize_image(image_path)
  save_image(n,output_path)