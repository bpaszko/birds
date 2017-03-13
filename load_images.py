import numpy as np
import pickle, os
from PIL import Image
def load_image(f_name):
    im = Image.open(f_name)
    out = im.resize((32, 32))
    pixels = list(out.getdata())
    try:
        if not isinstance(pixels[0], tuple):
            return 0
    except: 
        return 0
    data = np.zeros(shape = (32,32,3), dtype=np.float32)
    for i in range(32):
        for j in range(32):
            data[i][j][0] = pixels[i*32+j][0]
            data[i][j][1] = pixels[i*32+j][1]
            data[i][j][2] = pixels[i*32+j][2]
    return data    


def read_all():
    data = {'data':[], 'labels':[]}
    for dir in os.listdir('images'):
        for file in os.listdir('images/'+dir):
            dt = load_image('images/'+dir+'/'+file)
            if isinstance(dt, int):
                continue
            data['data'].append(dt)
            data['labels'].append(np.array([1,0]))
    return data
