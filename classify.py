import tensorflow as tf
import sys


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

        def predict(self):
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

if __name__ == "__main__":
        cfg = load_config()
        imgRec=ImageRecognition(cfg)
        #imgRec.set_image_path("brocolli.jpg")
        #imgRec.read_image('brocolli.jpg')
        img_path = sys.argv[1]
        imgRec.read_image(img_path)
        #print(imgRec.predict())
        labels,scores = imgRec.predict_multilabel()
        added_scores = sum(scores)
        print("Total sum of scores is:",added_scores)
        print(list(zip(labels,scores)))
