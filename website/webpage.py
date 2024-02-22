from flask import Flask, render_template, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf

model = load_model('C:\major\Breast-cancer-segmentation-using-Attention-Unet\dataset\Aunetresults_re.keras')
classify=load_model('C:\major\Breast-cancer-segmentation-using-Attention-Unet\dataset\classification_re.keras')

def l_image(t):
    test_image = image.load_img(t, target_size=(256,256))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    return test_image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = r'C:\major\Breast-cancer-segmentation-using-Attention-Unet\test'

@app.route('/')
def upload_form():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_image():
    file = request.files['file']
    filename = secure_filename(file.filename)
    t=r'C:\major\Breast-cancer-segmentation-using-Attention-Unet\website\static\uploads'
    t=os.path.join(t, filename)
    file.save(t)

    class_names = ['benign ', 'malignant ', 'normal ']
    img = l_image(t)
    mask = model.predict(img)
    image = cv2.imread(t)
    image = cv2.resize(image, dsize=(255, 255))
    plt.imshow(image)
    plt.imshow(tf.squeeze(mask[0]), alpha=0.4)
    plt.axis('off')
    plt.savefig('C:\major\Breast-cancer-segmentation-using-Attention-Unet\website\static\output\output.png')
    c=classify.predict(img)
    score = tf.nn.softmax(c[0])
    data=str(class_names[np.argmax(score)])
    acc=str(100 * np.max(score))

    return render_template('results.html',cla=data,per=acc)

if __name__ == '__main__':
    app.run(debug=False)