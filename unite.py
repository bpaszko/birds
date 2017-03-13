import pickle
import numpy as np

def wrap(pixels):
    colors_offset = len(pixels)//3
    row, col = 0, 0
    data = np.zeros((32,32,3), dtype=np.float32)
    for i in range(colors_offset):
        #print(row, col, i)
        data[row][col][0] = pixels[i]
        data[row][col][1] = pixels[i+colors_offset]
        data[row][col][2] = pixels[i+colors_offset*2]
        col = col+1
        if col % 32 ==0: 
            col = 0
            row = row + 1
    return data

def unpack(file):
    f = open(file, "rb")
    dc = pickle.load(f, encoding='bytes')
    newdict = {'data':[], 'labels':[]}
    for i in range(len(dc[b'data'])):
        newdict['data'].append(wrap(dc[b'data'][i]))
        newdict['labels'].append(np.array([1, 0]) if dc[b'labels'][i] == 2 else np.array([0, 1]))
    f.close()
    return newdict 

my_dict = {'data':[], 'labels':[]}
for i in range(1,6):
    name = 'cifar-10-batches-py/data_batch_' + str(i)
    part = unpack(name)
    my_dict['data'] += part['data']
    my_dict['labels'] += part['labels']

part = unpack('cifar-10-batches-py/test_batch')
my_dict['data'] += part['data']
my_dict['labels'] += part['labels']
del part
#with open('full_data.pkl', 'wb') as f:
#    pickle.dump(my_dict, f)
#    f.close()
