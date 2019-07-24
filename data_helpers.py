import numpy as np
import re


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r", encoding='utf-8').readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r", encoding='utf-8').readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]
def cargar():
    x_text=[]
    tam=[]
    for i in range(10):
        nombre=str(i)+".txt"
        archivo_to_read=list(open(nombre, "r", encoding='utf-8').readlines())
        archivo_to_read=[s.strip() for s in archivo_to_read]
        tam.append(len(archivo_to_read))
        x_text+=archivo_to_read
    x_text=[clean_str(sent) for sent in x_text]
    #Generate Labels
    label_0=[[1,0,0,0,0,0,0,0,0,0] for j in range(tam[0])]
    label_1=[[0,1,0,0,0,0,0,0,0,0] for j in range(tam[1])]
    label_2=[[0,0,1,0,0,0,0,0,0,0] for j in range(tam[2])]
    label_3=[[0,0,0,1,0,0,0,0,0,0] for j in range(tam[3])]
    label_4=[[0,0,0,0,1,0,0,0,0,0] for j in range(tam[4])]
    label_5=[[0,0,0,0,0,1,0,0,0,0] for j in range(tam[5])]
    label_6=[[0,0,0,0,0,0,1,0,0,0] for j in range(tam[6])]
    label_7=[[0,0,0,0,0,0,0,1,0,0] for j in range(tam[7])]
    label_8=[[0,0,0,0,0,0,0,0,1,0] for j in range(tam[8])]
    label_9=[[0,0,0,0,0,0,0,0,0,1] for j in range(tam[9])]
    y=np.concatenate([label_0,label_1,label_2,label_3,label_4,label_5,label_6,label_7,label_8,label_9],0)
    return [x_text,y]    
def cargar_test(archivo):
    data_=list(open(archivo, "r", encoding='utf-8').readlines())    
    x_text=[sent.rstrip("\n") for sent in data_]
    label_=[[1,0,0,0,0,0,0,0,0,0] for j in range(len(x_text))]
    y=np.concatenate([label_],0)
    return [x_text,y]
def cargar_pruebas():
    x_text=[]
    tam=[]
    for i in range(9):
        nombre="t_"+str(i)+".txt"
        archivo_to_read=list(open(nombre, "r", encoding='utf-8').readlines())
        archivo_to_read=[s.strip() for s in archivo_to_read]
        tam.append(len(archivo_to_read))
        x_text+=archivo_to_read
    x_text=[clean_str(sent) for sent in x_text]
    #Generate Labels
    label_0=[[1,0,0,0,0,0,0,0,0] for j in range(tam[0])]
    label_1=[[0,1,0,0,0,0,0,0,0] for j in range(tam[1])]
    label_2=[[0,0,1,0,0,0,0,0,0] for j in range(tam[2])]
    label_3=[[0,0,0,1,0,0,0,0,0] for j in range(tam[3])]
    label_4=[[0,0,0,0,1,0,0,0,0] for j in range(tam[4])]
    label_5=[[0,0,0,0,0,1,0,0,0] for j in range(tam[5])]
    label_6=[[0,0,0,0,0,0,1,0,0] for j in range(tam[6])]
    label_7=[[0,0,0,0,0,0,0,1,0] for j in range(tam[7])]
    label_8=[[0,0,0,0,0,0,0,0,1] for j in range(tam[8])]
    y=np.concatenate([label_0,label_1,label_2,label_3,label_4,label_5,label_6,label_7,label_8],0)
    return [x_text,y]
def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
