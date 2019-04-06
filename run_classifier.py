import tensorflow as tf
import tensorflow_hub as hub

from tensorflow.keras import layers
import os
import numpy as np
import PIL as pil
import numpy as np

#----------------------Create session
import tensorflow.keras.backend as K
sess = K.get_session()
init = tf.global_variables_initializer()

#-------------------------------------------------------------------------------
'''
feature_extractor_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/2" #@param {type:"string"}

def feature_extractor(x):
    feature_extractor_module = hub.Module(feature_extractor_url)
    return feature_extractor_module(x)

IMAGE_SIZE = hub.get_expected_image_size(hub.Module(feature_extractor_url))
'''

classifier_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/2" #@param {type:"string"}
def classifier(x):
    classifier_module = hub.Module(classifier_url)
    return classifier_module(x)

IMAGE_SIZE = hub.get_expected_image_size(hub.Module(classifier_url))


img_path  = "dummy_data/Fish/19.jpeg"
img       = pil.Image.open(img_path)
img       = img.resize(IMAGE_SIZE)
img       = np.array(img) / 255.0
print(img.shape)
#model_path = os.path.join("saved_models","1554354743")
model_path = b'./saved_models/1554396194'
#------------------Load Model
# Load the saved keras model back.
model = tf.contrib.saved_model.load_keras_model(model_path)
model.summary()


#------------------Predict
predictions = model.predict(img[np.newaxis, ...])

print(np.argmax(predictions[0]))
