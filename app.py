from flask import Flask, request, jsonify, render_template, send_file
from keras.preprocessing import image
import numpy as np
from utils import normalize
from keras.models import load_model, Model
from PIL import Image

app = Flask(__name__, template_folder='website')

# Load the saved segmentation model
segmentation_model = Model(inputs=[input_layer], outputs=[output_layer])
segmentation_model.load_weights('C:\major\Breast-cancer-segmentation-using-Attention-Unet\dataset\Aunetresults1.keras')

# Load the saved classification model
classification_model = load_model('C:\major\Breast-cancer-segmentation-using-Attention-Unet\dataset\classification.keras')

@app.route('/')
def home():
   return render_template('index.html')

app = Flask(__name__)
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    image_path = f'upload_image/{image_file.filename}'
    image_file.save(image_path)

    # Normalize the uploaded image
    normalized_image_path = f'normalized_image/{image_file.filename}'
    normalize(image_path, normalized_image_path)

    # Process the normalized image
    processed_data = process_image(normalized_image_path)

    return jsonify({'data': processed_data})

def process_image(image_path):
    # Load the normalized image
    img = image.load_img(image_path, target_size=(256, 256))
    img_array = image.img_to_array(img)

    # Add additional processing steps here

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