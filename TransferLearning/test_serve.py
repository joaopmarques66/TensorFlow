
import argparse
import json

import numpy as np
import requests
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from grpc.beta import implementations
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from tensorflow.python.framework import tensor_util

channel = implementations.insecure_channel('localhost',9000) 

stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

# Argument parser for giving input image_path from command line
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path of the image")
args = vars(ap.parse_args())

image_path = args['image']
# Preprocessing our input image
img = image.img_to_array(image.load_img(image_path, target_size=(299, 299))) / 255.

# this line is added because of a bug in tf_serving(1.10.0-dev)
img = img.astype('float32')

request = predict_pb2.PredictRequest()
 
request.model_spec.name = 'inception'

request.inputs['input_image'].CopyFrom(
  tf.contrib.util.make_tensor_proto(img, shape=[1, 299, 299, 3]))

result = stub.Predict(request, 60.0) 
print(result)