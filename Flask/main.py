from flask import request, url_for
from flask_api import FlaskAPI, status, exceptions
import tensorflow as tf
import tensorflow_hub as hub
import PIL
from tensorflow.keras.preprocessing import image as Image
import numpy as np
import tensorflow.keras.backend as K
import os
from grpc.beta import implementations
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from tensorflow.python.framework import tensor_util

app = FlaskAPI(__name__)

IMAGE_SIZE = (299, 299)

global original_labels
original_labels = np.array([
  'apple_pie',
  'chocolate_cake',
  'churros',
  'guacamole',
  'hamburger',
  'pancakes',
  'pizza',
  'risotto',
  'spaghetti_bolognese',
  'tacos',
  'waffles'])

def process_image(image):
    if image.mode != 'RGB':
        image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE)
    image = tf.keras.preprocessing.image.img_to_array(image) / 255
    #image = np.expand_dims(image, axis=0)

    return image


@app.route("/predict", methods=['GET', 'POST'])
def notes_list():
    if request.method != 'POST':
        response = {}
        response["error"] = "Method not allowed"
        return str(response)

    if 'file' not in request.files:
        response = {}
        response["error"] = "No file part " + str(request)
        return str(response)

    file = request.files['file']
    if file.filename == '':
        response = {}
        response["error"] = "No selected file"
        return str(response)
        
    image = PIL.Image.open(file)
    processed_image = process_image(image)

    channel = implementations.insecure_channel('localhost',9000) 
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    
    _request = predict_pb2.PredictRequest()
    _request.model_spec.name = 'inception'
    _request.inputs['input_image'].CopyFrom(
        tf.contrib.util.make_tensor_proto(processed_image, shape=[1, 299, 299, 3]))

    result = stub.Predict(_request, 60.0) 
    result_array = tensor_util.MakeNdarray(result.outputs['dense/Softmax:0'])

    argmax =  np.argmax(result_array, axis=-1)
    response = {}
    response['class_index'] = str(argmax[0])
    response['confidence'] = str(result_array[0][argmax][0])
    response['result'] = str(original_labels[argmax][0])
    return str(response)



if __name__ == "__main__":
    app.run(debug=True)