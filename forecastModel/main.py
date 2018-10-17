import os
import sys
from numpy import ndarray
import tensorflow as tf


def preprocess_data(root_data_path):
    from .preprocess.preprocess import Preprocessing
    preprocessing = Preprocessing(root_data_path)
    preprocessing.preprocessing()


def train(root_data_path, company_name, model_size):
    from .model.model import Model

    tf.reset_default_graph()
    with tf.Session() as session:
        learning = Model(session, root_data_path, model_size[0], model_size[1])
        learning.training(company_name)
        session.close()


def forecast(root_data_path, root_model_path, company_name, model_size):
    from .model.model import Model

    tf.reset_default_graph()
    with tf.Session() as session:
        learning = Model(session, root_data_path, model_size[0], model_size[1])
        hypothesis, acc, recall, precision = learning.forecast(root_data_path, root_model_path, company_name)
        session.close()

        return hypothesis.tolist(), acc, recall, precision


if __name__ == '__main__':
    from model.model import Model
    from preprocess.preprocess import Preprocessing

    root_dir = '/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:-1])
    # preprocessing = Preprocessing(root_dir)
    # preprocessing.preprocessing()

    arg = sys.argv
    learning = Model(tf.Session(), root_dir, [4, 8, 5], [1, 3, 2])
    learning.training(arg[1])

    # print(learning.forecast(arg[1], arg[2], arg[3]))
