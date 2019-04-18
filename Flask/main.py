from flask import request, url_for
from flask_api import FlaskAPI, status, exceptions
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import numpy as np
import tensorflow.keras.backend as K
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"   
app = FlaskAPI(__name__)

HUB_URL = "https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1"
PATH='/home/joao/repos/TensorFlowTest/TransferLearning/experimental/Inceptionv3-1_epochs-numclasses_32'
IMAGE_SIZE = (299, 299)
global original_labels
original_labels = np.array([
  'apple_pie',
  'guacamole',
  'tacos',
  'pizza',
  'waffles'])

def process_image(image):
    if image.mode != 'RGB':
        image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE)
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)

    return image

global loaded_model
loaded_model = tf.keras.models.load_model(PATH + '/saved_model.h5')
with tf.Session() as sess:
   # sess = K.get_session()
    saver = tf.train.Saver()
    saver.restore(sess, PATH + '/checkpoint.ckpt')
    init = tf.global_variables_initializer()
    sess.run(init)      

@app.route("/predict", methods=['GET', 'POST'])
def notes_list():
    with tf.Session() as sess:

        if request.method == 'POST':
            # check if the post request has the file part
            if 'file' not in request.files:
                return 'No file part'
            file = request.files['file']
            # if user does not select file, browser also
            # submit a empty part without filename
            if file.filename == '':
                return 'No selected file'
            image = Image.open(file)
            processed_image = process_image(image)

            prediction = loaded_model.predict(processed_image)
            return 'success ' + str(prediction)

    return ''


if __name__ == "__main__":
    app.run(debug=True)