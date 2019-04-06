import tensorflow as tf
import tensorflow_hub as hub

from tensorflow.keras import layers

#----------------------Create session
import tensorflow.keras.backend as K
import PIL as pil
import numpy as np
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


#-------------------------------------------------------------------------------
data_root = "dummy_data"

image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
image_data = image_generator.flow_from_directory(str(data_root))


#-------------------------------------------------------------------------------
image_data = image_generator.flow_from_directory(str(data_root), target_size=IMAGE_SIZE)
for image_batch,label_batch in image_data:
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
  layers.Dense(image_data.num_classes, activation='softmax')
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
steps_per_epoch = image_data.samples//image_data.batch_size
batch_stats = CollectBatchStats()
model.fit((item for item in image_data), epochs=1,
                    steps_per_epoch=steps_per_epoch,
                    callbacks = [batch_stats])

#-----------------Do some plotting here


#-----------------Save model
export_path = tf.contrib.saved_model.save_keras_model(model, "./saved_models")

print("*"*20)
print("EXPORT PATH: ",export_path)
print("*"*20)

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
data_root = "dummy_data"

image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
training_data   = image_generator.flow_from_directory(str(data_root), target_size=IMAGE_SIZE)



#------------------Initialize variables
init = tf.global_variables_initializer()
sess.run(init)

#------------------Predict
predictions = model.predict(training_data)

print("Predictions shape:", predictions.shape,'\n')
print("Most likely class: ",np.argmax(predictions[0]))
