import tensorflow as tf
import tensorflow_hub as hub

from tensorflow.keras import layers

#----------------------Create session
import tensorflow.keras.backend as K
import PIL as pil
import numpy as np
import os


class PicnicClassifier(object):
    #IMAGE_SIZE = (224,224)
    #DATA_ROOT  = 'dataset'
    def __init__(self,image_size=(224,224),
                      data_root='dataset',
                      classifier_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/2"):

        self.IMAGE_SIZE     = image_size
        self.DATA_ROOT      = data_root
        self.classifier_url = classifier_url
        self.sess           = K.get_session()
        self.model          = None
        self.predictions    = None
        self.export_path    = None

        #-------Initialize Data generators
        self.testing_data   = None
        self.training_data  = None
        #self._initialize_generators()

    def _classifier(self, x):
        classifier_module = hub.Module(self.classifier_url)
        return classifier_module(x)

    def _set_expected_image_size(self):
        self.IMAGE_SIZE = hub.get_expected_image_size(hub.Module(self.classifier_url))
        return self.IMAGE_SIZE

    def train_model(self):
        pass

    def _initialize_generators(self):
        self.testing_data  = self._test_generator()
        self.training_data = self._train_generator()

    def _train_generator(self,directory='train'):
        train_root = os.path.join(self.DATA_ROOT,directory)
        training_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
        training_data      = training_generator.flow_from_directory(directory=str(train_root),
                                                                    color_mode='rgb',
                                                                    target_size=self.IMAGE_SIZE,
                                                                    batch_size=100,
                                                                    class_mode='categorical',
                                                                    shuffle=True)

        return training_data

    def _test_generator(self,directory='test'):
        test_root = os.path.join(self.DATA_ROOT,directory)
        testing_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
        testing_data      = testing_generator.flow_from_directory(directory=str(test_root),
                                                                  target_size=self.IMAGE_SIZE,
                                                                  class_mode=None,
                                                                  batch_size=1,
                                                                  shuffle=False,
                                                                  color_mode='rgb')
        return testing_data

    def _validation_generator(self):
        raise NotImplementedError

    def stat_session(self):
        init = tf.global_variables_initializer()
        self.sess.run(init)
        return self.sess

    #===========================
    #   Predict, train, load, save, and all that

    def predict(self,use_generator=1):
        self.start_session()
        self.testing_data = self._test_generator()
        '''
        self.testing_data = self._test_generator()
        self.testing_data.reset()

        if use_generator:
            self.predictions = self.model.predict_generator(self.testing_data,verbose=1)
        else:
            self.predictions = self.model.predict(self.testing_data,verbose=1)
        '''
    def train(self):
        ...

    def load_model(self,directory=None,verbose=0):
        # Load the saved keras model back.
        if directory is None:
            model_path = self.export_path
        else:
            model_path = directory

        self.model = tf.contrib.saved_model.load_keras_model(model_path)
        if verbose:
            print("\nModel loaded from: {}\n".format(model_path))
            print("Loaded model summary:\n")
            model.summary()

    def save_model(self,directory='saved_models',verbose=0):
        #-----------------Save model
        self.export_path = tf.contrib.saved_model.save_keras_model(self.model, directory)
        if verbose:
            print("\n***\tModel data saved to: {}\n".format(self.export_path))
            print("Model summary:\n")
            self.model.summary()

        return self.export_path

    def get_export_path(self):
        return self.export_path

    def save_predictions(self,output_file='results.tsv'):
        predicted_class_indices = np.argmax(self.predictions,axis=1)

        print("Predictions shape:", predictions.shape,'\n')
        print("Most likely classes:\n",predicted_class_indices)

        labels = (self.training_data.class_indices)
        labels = dict((v,k) for k,v in labels.items())
        predictions = [labels[k] for k in predicted_class_indices]

        filenames = self.testing_data.filenames
        filenames = remove_path(filenames)
        results   = pd.DataFrame({"File":filenames,
                                  "Label":predictions})
        results.to_csv(output_file,index=False,sep='\t')

def remove_path(filenames):
    only_files = [filepath.split('/')[-1] for filepath in filenames]
    return only_files
class ImageRecognition(object):
        FILTER_MODE=0
        def __init__(self, path_to_labels, path_to_model):
                self.image_path = None
                self.retrained_labels = path_to_labels
                self.retrained_graph = path_to_model
                self.label_lines = self.read_labels()
                self._filter_threshold = 0.0015

        def read_image(self,image_path):
                # Read in the image_data
                self.image_data = tf.gfile.FastGFile(image_path, 'rb').read()

        def read_labels(self):
                # Loads label file, strips off carriage return
                label_lines = [line.rstrip() for line in tf.gfile.GFile(self.retrained_labels)]
                return label_lines

        def load_graph(self,model_file):
            '''Example taken from label_image.py from tensorflow examples repository'''
            graph = tf.Graph()
            graph_def = tf.GraphDef()

            with open(model_file, "rb") as f:
                graph_def.ParseFromString(f.read())
            with graph.as_default():
                tf.import_graph_def(graph_def)

            return graph

        def set_filter_mode(self,mode):
                '''
                Changes the mode of the filter applied during the classification process
                   (is it simple filter or not?)
                mode: a bool (True/False) or int (1 or 0); True means that simple filter will be used. False indicates the opposite
                '''
                if isinstance(mode,(bool,int)):
                        assert(mode==1 or mode==0)
                        self.FILTER_MODE = mode

        def print_rank(self, predictions):
                for node_id in predictions:
                        human_string = self.label_lines[node_id]
                        score = predictions[0][node_id]
                        print('%s (score = %.5f)' % (human_string, score))

        def predict_OLD(self):
                # Unpersists graph from file
                with tf.gfile.FastGFile(self.retrained_graph, 'rb') as f:
                        graph_def = tf.GraphDef()
                        graph_def.ParseFromString(f.read())
                        _ = tf.import_graph_def(graph_def, name='')
                with tf.Session() as sess:
                        # Feed the image_data as input to the graph and get first prediction
                        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

                        predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': self.image_data})

                        # Sort to show labels of first prediction in order of confidence
                        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
                print(type(top_k),top_k)   #DEBUGGING LINE
                label = self.label_lines[top_k[0]]
                #certainty = predictions[0][top_k[0]]

                return label

        def predict(self):
                # Unpersists graph from file
                #'''
                with open(self.retrained_graph, 'rb') as f:
                        graph_def = tf.GraphDef()
                        proto_b   = f.read()

                        text_format.Merge(proto_b,graph_def)

                        #graph_def.ParseFromString(f.read())
                        _ = tf.import_graph_def(graph_def, name='')
                #'''
                #graph = self.load_graph(self.retrained_graph)
                with tf.Session(graph=graph) as sess:
                        # Feed the image_data as input to the graph and get first prediction
                        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

                        predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': self.image_data})

                        # Sort to show labels of first prediction in order of confidence
                        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
                print(type(top_k),top_k)   #DEBUGGING LINE
                label = self.label_lines[top_k[0]]
                #certainty = predictions[0][top_k[0]]

                return label


        def _prediction_filter(self,labels,scores,simple_filter=True):
                '''
                Will help filter out the items that are not likely to be in the image
                A simple_filter = True means that the probabilities below a predifined threshold will be removed.
                If it is False, then some other calculation will be used (to be determined)
                '''
                filtered_scores = []
                filtered_labels = []
                if simple_filter:
                        for i in range(len(scores)):
                                if scores[i]>self._filter_threshold:
                                        filtered_scores.append(scores[i])
                                        filtered_labels.append(labels[i])
                else:
                     #this is still a very simplistic filtering
                     avg = sum(scores)/len(scores)
                     max_score = scores[0]

                     for i in range(1,len(scores)):
                             sub_avg = sum(scores[i:])/len(scores[i:])
                             print("Avg:",avg,"sub_avg:",sub_avg,"Diff:",avg-sub_avg)
                             #if the difference between the new average and the actual average is too big, then we exclude all the items from this point in the loop forward
                             if (avg-sub_avg) > 0.002:     #the threshold here is completely arbitrary for now
                                     filtered_scores = scores[:i]
                                     filtered_labels = labels[:i]
                                     break

                return filtered_labels,filtered_scores

        def predict_multilabel(self):
                '''Placeholder. It should find several types of objects present in one image'''
                with tf.gfile.FastGFile(self.retrained_graph, 'rb') as f:
                        graph_def = tf.GraphDef()
                        graph_def.ParseFromString(f.read())
                        _ = tf.import_graph_def(graph_def, name = '')
                with tf.Session() as sess:
                        sigmoid_tensor = sess.graph.get_tensor_by_name('final_result:0')
                        predictions = sess.run(sigmoid_tensor, {'DecodeJpeg/contents:0': self.image_data})

                        #Sort labels
                        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
                print(type(top_k),'is ', top_k)  #REMOVE AFTER DEBUGGING
                labels = [self.label_lines[ID] for ID in top_k]  #a list of sorted labels
                scores = [predictions[0][node_id] for node_id in top_k]

                #after the initial predictions, we filter the less likely items
                labels_filtered,scores_filtered = self._prediction_filter(labels=labels,
                                                                          scores=scores,
                                                                          simple_filter=self.FILTER_MODE)


                #tf.nn.sigmoid_cross_entropy_with_logits(logits,targets,name=None)
                #labels = None
                return labels_filtered,scores_filtered

if __name__ == '__main__':

    #Set up argument parsing
    pass

if __name__ == "__main2__":
        #cfg = load_config()
        img_path    = sys.argv[1]
        labels_path = sys.argv[2]
        model_path  = sys.argv[3]
        imgRec=ImageRecognition(labels_path,model_path)
        #imgRec.set_image_path("brocolli.jpg")
        #imgRec.read_image('brocolli.jpg')
        img_path = sys.argv[1]
        imgRec.read_image(img_path)
        #print(imgRec.predict())
        labels,scores = imgRec.predict()
        added_scores = sum(scores)
        print("Total sum of scores is:",added_scores)
        print(list(zip(labels,scores)))
