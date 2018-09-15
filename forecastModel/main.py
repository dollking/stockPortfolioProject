import os
import tensorflow as tf

from model.model import Model
from preprocess.preprocess import Preprocessing


if __name__ == '__main__':
    root_dir = '/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:-1])
    # preprocessing = Preprocessing(root_dir)
    # preprocessing.preprocessing()

    learning = Model(tf.Session(), root_dir, [2, 3, 1], [2, 3, 1])
    learning.training()

    learning.get_predict()
