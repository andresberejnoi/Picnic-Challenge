import pandas as pd
import os
import shutil      #to copy files 

def get_train_files_labels(filename):
    """
    Data for the challenge comes in .tsv format. Two header columns.
    The first column contains filenames. Second column contains the respective labels.
    There can be more than one label per product.
    return a dictionary mapping filenames to their associated labels 
    """
    
    all_data = pd.read_csv(filename,sep='\t')
    
    filenames = all_data['file']        #filenames are separated
    labels    = all_data['label']       #labels are separeted but several labels for one image are still grouped in one string

    #-----Preprocess labels 
    split_labels = separate_labels(labels)
    
    
    #-----Turn it all into a dictionary
    d = {file:tags for file,tags in zip(filenames,split_labels)}
    
    return d

def separate_labels(labels):
    """
    Each row corresponds to a file and can have many layers. 
    Separate the layers into a list per row.
    This replaces the '&' character with ',' and then splits the string with ',' as separator
    """
    split_labels = [ [item.strip() for item in l.replace('&',',').split(',')] for l in labels]
    
    return split_labels

def separate_object_classes(filename):
    #----Read train file
    data = pd.read_csv(filename,sep='\t')
    
    #----Make dictioary of labels and populate it
    categories_dic = {label:[] for label in data['label']}
    
    for category in categories_dic:
        filtered = data.loc[data['label']==category]
        files = filtered['file']
        categories_dic[category] = files
    
    return categories_dic

def create_new_dataset_folder(output_folder="New_Dataset"):
    ''''''
    
def create_dirs(dir_list, root='.'):
    for dirname in dir_list:
        #---Create folder
        filepath = os.path.join(root,dirname)
        os.mkdir(filepath)

def rearrange_files(dest,source,filenames):
    
    for name in filenames:
        source_file = os.path.join(source,name)
        shutil.copy(source_file,dest)

def make_new_dataset(filename):
    new_dataset_folder = "train_dataset_arranged"
    try:
        os.mkdir(new_dataset_folder)
    except FileExistsError:
        pass
    
    #---Get the categories in the dataset
    categories_dic = separate_object_classes(filename)
    
    #---Create folders for each category
    create_dirs(categories_dic.keys(),new_dataset_folder)
    
    #---Copy file into their respective category folders
    root_source      = os.path.join("The Picnic Hackathon 2019","train")
    #root_destination = new_dataset_folder 
    for cat in categories_dic:
        print('*'*80)
        print("Processing Images for category: {}".format(cat))
        destination = os.path.join(new_dataset_folder,cat)
        files = categories_dic[cat]
        rearrange_files(destination,root_source,files)
    

def main(filename):
    make_new_dataset(filename)
    
if __name__ == '__main__':
    #name = "The Picnic Hackathon 2019/train.tsv"
    import sys
    main(sys.argv[1])