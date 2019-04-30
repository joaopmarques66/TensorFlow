import tensorflow as tf
VERSION=1
PATH = './export_model/{}'.format(VERSION)
converter = tf.lite.TFLiteConverter.from_saved_model(PATH)
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)