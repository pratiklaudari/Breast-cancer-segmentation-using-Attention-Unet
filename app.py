from flask import Flask, request, jsonify, render_template
from keras.preprocessing import image
import numpy as np
from utils import normalize
from keras.models import load_model

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

    # Load and preprocess the normalized image
    image = cv2.imread(normalized_image_path)
    image = cv2.resize(image, (256, 256))
    image = np.expand_dims(image, axis=0) / 255.0

    # Make the segmentation prediction
    segmentation_prediction = segmentation_model.predict(image)
    segmentation_prediction = np.round(segmentation_prediction)

    # Save the segmentation result
    segmentation_result_path = 'segmentation_result.png'
    cv2.imwrite(segmentation_result_path, segmentation_prediction[0] * 255)

    # Make the classification prediction
    classification_prediction = classification_model.predict(image)

    # Extract the predicted class label
    predicted_class = np.argmax(classification_prediction)

    return jsonify({'segmentation_result_path': segmentation_result_path, 'predicted_class': predicted_class})

if __name__ == '__main__':
   app.run()