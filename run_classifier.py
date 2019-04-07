import tensorflow as tf
import tensorflow_hub as hub

from tensorflow.keras import layers

#----------------------Create session
import tensorflow.keras.backend as K
import PIL as pil
import os
import pandas as pd
import numpy as np
import argparse



classifier_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/2" #@param {type:"string"}
def classifier(x):
    classifier_module = hub.Module(classifier_url)
    return classifier_module(x)

IMAGE_SIZE = hub.get_expected_image_size(hub.Module(classifier_url))




#===============================================================================
#                  Prepare argument parsing and model testing
#===============================================================================
parser = argparse.ArgumentParser(description='Picnic Image Classifier (Runner)')
parser.add_argument('-m', '--model_path', type=str, default=None)
parser.add_argument('-d', '--dataset',type=str,default='dataset',help='Dataset root')
parser.add_argument('-l', '--labels',type=str,default='labels.txt')
args = parser.parse_args()
if args.model_path is None:
    with open('export_path.txt','rb') as f:
        export_path = f.read()

else:
    export_path = args.model_path



export_path = str(export_path,'UTF-8')
print("****Model PATH to use: ", export_path)
#===============================================================================



#------------------Load Model
# Load the saved keras model back.
model = tf.contrib.saved_model.load_keras_model(export_path)
model.summary()

#------------------Prepare data to read
#data_root = "dataset"
DATA_ROOT = os.path.join(".",args.dataset)
test_root = os.path.join(DATA_ROOT, "test")

testing_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
testing_data      = testing_generator.flow_from_directory(directory=str(test_root),
                                                           target_size=IMAGE_SIZE,
                                                           class_mode=None,
                                                           batch_size=1,
                                                           shuffle=False,
                                                           color_mode='rgb')

#------------------Initialize variables
sess = K.get_session()
init = tf.global_variables_initializer()
sess.run(init)


#------------------Predict
testing_data.reset()               #reset generator
predictions = model.predict_generator(testing_data,verbose=1)


#------------------Process results and save
predicted_class_indices = np.argmax(predictions,axis=1)

print("Predictions shape:", predictions.shape,'\n')
print("Most likely classes:\n",predicted_class_indices)

labels = (training_data.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]


filenames = testing_data.filenames
results   = pd.DataFrame({"File":filenames,
                          "Label":predictions})
results.to_csv("results.tsv",index=False,sep='\t')
