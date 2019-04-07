import os
import argparse
from classifier import PicnicClassifier


def main(args):

    #-------Create and set up the classifier
    clapi = PicnicClassifier(data_root=args.dataset)
    clapi._set_expected_image_size()

    #-------Check if a model should be trained or just tested
    if args.train:
        clapi.train()
        clapi.save_model(directory=args.output_model,verbose=1)

    if args.predict:
        clapi.load_model(directory=args.saved_model,verbose=1)
        clapi.predict()
        clapi.save_predictions(output_file=args.output_predictions)


if __name__ == '__main__':
    default_model = os.listdir('saved_models')[-1]
    #---------Set up argument parsing
    parser = argparse.ArgumentParser(description='Picnic Image Classifier')
    parser.add_argument('-d', '--dataset',type=str,default='dataset',help='Dataset root')
    parser.add_argument('-l', '--labels',type=str,default='labels.txt',help="File with all the classes or categories")
    parser.add_argument('--saved_model',type=str,default=os.path.join('saved_models',default_model))
    parser.add_argument('--output_predictions',type=str,default='results.tsv')
    parser.add_argument('--output_model',type=str,default='saved_models')
    parser.add_argument('-p', '--predict', action='store_true')
    parser.add_argument('-t', '--train',action='store_true')
    args = parser.parse_args()

    main(args)
