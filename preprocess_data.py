import pandas as pd

def read_train_data(filename):
    """
    Data for the challenge comes in .tsv format. Two header columns.
    The first column contains filenames. Second column contains the respective labels.
    There can be more than one label per product.
    """
    
    all_data = pd.read_csv(filename,sep='\t')
    return all_data
    

def main():
    read_train_data('test')

if __name__ == '__main__':
    main()