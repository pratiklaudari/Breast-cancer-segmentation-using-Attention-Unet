from flask import Flask, request, jsonify, send_file
from keras.preprocessing import image
import numpy as np
from keras.models import load_model, Model
from PIL import Image


def normalize(image_path):
    # Load image
    img = image.load_img(image_path)
    img_array = image.img_to_array(img)

    # Normalize by subtracting mean and dividing by standard deviation
    mean = np.average(img_array)
    sd = np.std(img_array)
    img_array -= mean
    img_array /= sd

    return img_array

def save_image(image_array, output_path):
    # Convert the image array back to PIL image format
    img = image.array_to_img(image_array)

    # Save the image to the desired path
    img.save(output_path)

app = Flask(__name__)


segmentation_model = load_model(r'C:\Users\shrij\OneDrive\Desktop\Breastcancer\Flask.h5')

classification_model = load_model('C:\major\Breast-cancer-segmentation-using-Attention-Unet\dataset\classification.keras')

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    image_path = f'upload_image/{image_file.filename}'
    image_file.save(image_path)

    # Normalize the uploaded image (assuming the 'normalize' function is defined in 'utils')
    normalized_image_path = f'normalized_image/{image_file.filename}'
    normalize(image_path, normalized_image_path)

    # Process the normalized image
    processed_data = process_image(normalized_image_path)

    return jsonify({'data': processed_data})

def process_image(image_path):
    # Load the normalized image
    img = image.load_img(image_path, target_size=(256, 256))
    img_array = image.img_to_array(img)

    return img_array


@app.route('/classify', methods=['POST'])
def classify_image():
    if 'image_path' not in request.form:
        return jsonify({'error': 'No image path provided'}), 400

    # Load the image from the given path
    img = image.load_img(request.form['image_path'], target_size=(256, 256))
    img_array = image.img_to_array(img)

    # Preprocess the image for classification
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Make the classification prediction
    classification_prediction = classification_model.predict(img_array)

    # Extract the predicted class label
    predicted_class = np.argmax(classification_prediction)

    return jsonify({'predicted_class': predicted_class})

@app.route('/segment', methods=['POST'])
def segment_image():
    if 'image_path' not in request.form:
        return jsonify({'error': 'No image path provided'}), 400

    # Load the image from the given path
    img = image.load_img(request.form['image_path'], target_size=(256, 256))
    img_array = image.img_to_array(img)

    # Preprocess the image for segmentation
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Make the segmentation prediction
    segmentation_prediction = segmentation_model.predict(img_array)

    # Postprocess the segmentation prediction
    segmentation_prediction = np.round(segmentation_prediction)
    segmentation_prediction = segmentation_prediction.squeeze()
    segmentation_prediction = np.where(segmentation_prediction > 0.5, 1, 0)

    # Save the segmentation result
    segmentation_result_path = 'segmentation_result.png'
    segmentation_result = Image.fromarray(segmentation_prediction * 255).convert('RGB')
    segmentation_result.save(segmentation_result_path)


if __name__ == '__main__':
    app.run()
