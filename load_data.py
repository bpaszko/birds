import pickle
import numpy

def load_data(file):
    f = open(file, 'rb') 
    data = pickle.load(f)
    f.close()
    return data

def split_data(data):
    p = numpy.random.permutation(len(data['labels']))
    dataset = data['data'][p]
    labels = data['labels'][p]
    train_data = dataset[:52000]
    train_labels = labels[:52000]
    valid_data = dataset[52000:65000]
    valid_labels= labels[52000:65000]
    test_data = dataset[65000:]
    test_labels = labels[65000:]
    return train_data, train_labels, valid_data, valid_labels, test_data, test_labels

train_data, train_labels, valid_data, valid_labels, test_data, test_labels = split_data(load_data('full_data.pkl'))


