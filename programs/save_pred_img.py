
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import cv2
from glob import glob as gb
model = load_model('C:\major\Breast-cancer-segmentation-using-Attention-Unet\dataset\Aunetresults_re.keras')
image_path=r'C:\major\Breast-cancer-segmentation-using-Attention-Unet\for_evaluation\images\*.png'
images=gb(image_path)
def l_image(t):
    test_image = image.load_img(t, target_size=(256,256))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    return test_image
for i in images:
    img=l_image(i)
    pred_image=model.predict(img)
    pred_image=pred_image/225
    pred=image.array_to_img(pred_image[0])
    p=i.replace("images","predicted_image")
    pred.save(p)