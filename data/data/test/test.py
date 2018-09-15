import os
import pickle

path = './title'

for i in os.listdir(path):
    tmp = pickle.load(open(os.path.join(path, i), 'rb'))
    print(len(tmp), len(tmp[0]), len(tmp[0][0]), len(tmp[0][0][0]))
