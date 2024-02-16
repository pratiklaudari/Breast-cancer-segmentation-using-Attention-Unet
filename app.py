from flask import Flask, request, jsonify, render_template
from keras.preprocessing import image
import numpy as np

app = Flask(__name__, template_folder='website')
 
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

    return jsonify({'data': processed_data})  



if __name__ == '__main__':
   app.run()