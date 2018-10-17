import os

current_dir = root_dir = '/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:-1])
data_list = os.listdir(os.path.join(current_dir, 'data', 'data', 'test', 'index'))

for i in data_list:
    print(i.split('.')[0])
    os.system("python3 main.py '{}'".format(i.split('.')[0]))
