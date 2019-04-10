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
from classifier import remove_path

#===============================================================================
#                  Prepare argument parsing and model testing
#===============================================================================
parser = argparse.ArgumentParser(description='Picnic Image Classifier')
#parser.add_argument('-config', type=str, help='Name of configuration file')
#parser.add_argument('-broker', type=str, help="Name of the broker or counterparty to connect to. tier1 for Tier1FX and pig for FXPig")
parser.add_argument('-d', '--dataset',type=str,default='dataset',help='Dataset root')
parser.add_argument('-p', '--predict', action='store_true')
parser.add_argument('-e','--epochs', type=int, default=1)
parser.add_argument('-b','--bach_size',type=int,default=50)
parser.add_argument('--model_url',type=str,default="https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/2")
args = parser.parse_args()



#-------------------------------------------------------------------------------

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
#classifier_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/2" #@param {type:"string"}
classifier_url = args.model_url
def classifier(x):
    classifier_module = hub.Module(classifier_url)
    return classifier_module(x)

IMAGE_SIZE = hub.get_expected_image_size(hub.Module(classifier_url))


#-------------------------------------------------------------------------------
BATCH_SIZE = args.batch_size
DATA_ROOT = args.dataset
#ata_root = "dataset"
train_root = os.path.join(DATA_ROOT,'train')

training_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
training_data = training_generator.flow_from_directory(directory=str(train_root),
                                                       color_mode='rgb',
                                                       target_size=IMAGE_SIZE,
                                                       batch_size=BATCH_SIZE,
                                                       class_mode='categorical',
                                                       shuffle=True)


#-------------------------------------------------------------------------------
#image_data = image_generator.flow_from_directory(str(data_root), target_size=IMAGE_SIZE)
for image_batch,label_batch in training_data:
  print("Image batch shape: ", image_batch.shape)
  print("Label batch shape: ", label_batch.shape)
  break


#-----------------Use Keras layer--------------------------------------------------------------
features_extractor_layer = layers.Lambda(classifier, input_shape=IMAGE_SIZE+[3])

#-------------------Freeze variables in extractor layer
features_extractor_layer.trainable = False


#----------------Attach classification head
model = tf.keras.Sequential([
  features_extractor_layer,
  layers.Dense(training_data.num_classes, activation='softmax')
])
model.summary()

#----------------Initialize tf
init = tf.global_variables_initializer()
sess.run(init)


#----------------test run single batch
result = model.predict(image_batch)
result.shape


#-----------------Prepare to train
model.compile(
  optimizer=tf.train.AdamOptimizer(),
  loss='categorical_crossentropy',
  metrics=['accuracy'])

#-------------------Setup log objects
class CollectBatchStats(tf.keras.callbacks.Callback):
    def __init__(self):
        self.batch_losses = []
        self.batch_acc = []

    def on_batch_end(self, batch, logs=None):
        self.batch_losses.append(logs['loss'])
        self.batch_acc.append(logs['acc'])


#------------------Start training
steps_per_epoch = training_data.samples//training_data.batch_size
batch_stats = CollectBatchStats()
model.fit((item for item in training_data), epochs=args.epochs,
                    steps_per_epoch=steps_per_epoch,
                    callbacks = [batch_stats])

#-----------------Do some plotting here


#-----------------Save model
export_path = tf.contrib.saved_model.save_keras_model(model, "saved_models")

print("*"*20)
print("EXPORT PATH: ",export_path)
print("*"*20)


print("Saving export_path to file...")
with open("export_path.txt",'wb') as f:
    f.write(export_path)
print("Done saving")

print("Saving labels in file 'labels.txt'...")
labels = (training_data.class_indices)
with open("labels.txt",'w') as f:
    print(labels)
    #f.write(labels)
print("Done saving")

#===============================================================================
#   Load Model
if args.predict:

    '''
    img_path  = "dummy_data/Fish/19.jpeg"
    img       = pil.Image.open(img_path)
    img       = img.resize(IMAGE_SIZE)
    img       = np.array(img) / 255.0
    print(img.shape)
    '''
    #model_path = os.path.join("saved_models","1554354743")
    #model_path = b'./saved_models/1554396194'
    #------------------Load Model
    # Load the saved keras model back.
    model = tf.contrib.saved_model.load_keras_model(export_path)
    model.summary()

    #------------------Prepare data to read
    #data_root = "dataset"
    test_root = os.path.join(DATA_ROOT, "test")

    testing_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
    testing_data      = testing_generator.flow_from_directory(directory=str(test_root),
                                                               target_size=IMAGE_SIZE,
                                                               class_mode=None,
                                                               batch_size=1,
                                                               shuffle=False,
                                                               color_mode='rgb')

    #------------------Initialize variables
    init = tf.global_variables_initializer()
    sess.run(init)

    #------------------Predict
    testing_data.reset()               #reset generator
    predictions = model.predict_generator(testing_data,verbose=1)


    #------------------Process results and save
    predicted_class_indices = np.argmax(predictions,axis=1)

    print("Predictions shape:", predictions.shape,'\n')
    print("Most likely classes:\n",predicted_class_indices)

    labels = dict((v,k) for k,v in labels.items())
    predictions = [labels[k] for k in predicted_class_indices]


    filenames = testing_data.filenames
    filenames = remove_path(filenames)
    results   = pd.DataFrame({"file":filenames,
                              "label":predictions})
    results.to_csv("results.tsv",index=False,sep='\t')
