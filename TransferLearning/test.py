
import tensorflow as tf
import tensorflow_hub as hub
from random import shuffle
import cv2 
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pylab as plt
import tensorflow.keras.backend as K

original_labels = np.array([
  'apple_pie',
  #'baby_back_ribs',
  #'baklava',
  #'beef_carpaccio',
  #'beef_tartare',
  #'beet_salad',
  #'beignets',
  #'bibimbap',
  #'bread_pudding',
  #'breakfast_burrito',
  #'bruschetta',
  #'caesar_salad',
  #'cannoli',
  #'caprese_salad',
  #'carrot_cake',
  #'ceviche',
  #'cheesecake',
  #'cheese_plate',
  #'chicken_curry',
  #'chicken_quesadilla',
  #'chicken_wings',
  #'chocolate_cake',
  #'chocolate_mousse',
  #'churros',
  #'clam_chowder',
  #'club_sandwich',
  #'crab_cakes',
  #'creme_brulee',
  #'croque_madame',
  #'cup_cakes',
  #'deviled_eggs',
  #'donuts',
  #'dumplings',
  #'edamame',
  #'eggs_benedict',
  #'escargots',
  #'falafel',
  #'filet_mignon',
  #'fish_and_chips',
  #'foie_gras',
  #'french_fries',
  #'french_onion_soup',
  #'french_toast',
  #'fried_calamari',
  #'fried_rice',
  #'frozen_yogurt',
  #'garlic_bread',
  #'gnocchi',
  #'greek_salad',
  #'grilled_cheese_sandwich',
  #'grilled_salmon',
  'guacamole',
  #'gyoza',
  #'hamburger',
  #'hot_and_sour_soup',
  #'hot_dog',
  #'huevos_rancheros',
  #'hummus',
  #'ice_cream',
  #'lasagna',
  #'lobster_bisque',
  #'lobster_roll_sandwich',
  #'macaroni_and_cheese',
  #'macarons',
  #'miso_soup',
  #'mussels',
  #'nachos',
  #'omelette',
  #'onion_rings',
  #'oysters',
  #'pad_thai',
  #'paella',
  #'pancakes',
  #'panna_cotta',
  #'peking_duck',
  #'pho',
  'pizza',
  #'pork_chop',
  #'poutine',
  #'prime_rib',
  #'pulled_pork_sandwich',
  #'ramen',
  #'ravioli',
  #'red_velvet_cake',
  #'risotto',
  #'samosa',
  #'sashimi',
  #'scallops',
  #'seaweed_salad',
  #'shrimp_and_grits',
  #'spaghetti_bolognese',
  #'spaghetti_carbonara',
  #'spring_rolls',
  #'steak',
  #'strawberry_shortcake',
  #'sushi',
  'tacos',
  #'takoyaki',
  #'tiramisu',
  #'tuna_tartare',
  'waffles'])

def load_model():
  return tf.keras.models.load_model('./{}/saved_model.h5'.format(MODEL_NAME))
  """
    json_file = open('{}.json'.format(MODEL_NAME), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = tf.keras.models.model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("{}.h5".format(MODEL_NAME))
    print("Loaded model from disk")
    return loaded_model
  """  

HUB_URL = "https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1"

#MODEL_NAME='1024-units_1-epochs_0.0001-LR_249-frozen-layers'
MODEL_NAME='Inceptionv3-1_epochs-numclasses_32'
TESTING_ROOT='/home/joao/repos/FoodDataset/testing'
IMAGE_SIZE = (299, 299)

image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)

image_data = image_generator.flow_from_directory(str(TESTING_ROOT), target_size=IMAGE_SIZE, shuffle=True)

for image_batch,label_batch in image_data:
  print("Image batch shape: ", image_batch.shape)
  print("Labe batch shape: ", label_batch.shape)
  break
label_names = sorted(image_data.class_indices.items(), key=lambda pair:pair[1])
label_names = np.array([key.title() for key, value in label_names])


loaded_model  = load_model()
sess = K.get_session()
init = tf.global_variables_initializer()
sess.run(init)

saver = tf.train.Saver()
sess = tf.keras.backend.get_session()
saver.restore(sess, './{}/checkpoint.ckpt'.format(MODEL_NAME))

result_batch = loaded_model.predict(image_batch)

#print(result_batch)
#print(original_labels)
#print(label_names)
labels_batch = original_labels[np.argmax(result_batch, axis=-1)]

plt.figure(figsize=(10,9))
for n in range(21):
  plt.subplot(6,5,n+1)
  plt.imshow(image_batch[n])
  prob = result_batch[n][np.argmax(result_batch[n])] * 100
  plt.title(labels_batch[n] + " " + str(round(prob)) + "%")
  plt.axis('off')
_ = plt.suptitle("Model predictions")
plt.show()
