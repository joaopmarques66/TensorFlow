import tensorflow as tf
VERSION=1
PATH = './export_model/{}'.format(VERSION)
converter = tf.lite.TFLiteConverter.from_saved_model(PATH)
converter.post_training_quantize = True
tflite_model = converter.convert()
open("converted_model1.tflite", "wb").write(tflite_model)