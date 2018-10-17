import os
import pickle
import configparser
from numpy import ndarray

from forecastModel import main as forecast


def train_model(root_data_path, company_list):
    for company in company_list:
        forecast.train(os.path.join(root_data_path, 'data', 'train'), company,
                       [[int(i) for i in config['MODEL']['INDEX_MODEL_SIZE'].split(',')],
                        [int(i) for i in config['MODEL']['INDEX_MODEL_SIZE'].split(',')]])


def save_accuracy(root_data_path, root_model_path, company_list):
    for company in company_list:
        data = forecast.forecast(os.path.join(root_data_path, 'data', 'train'), root_model_path, company.strip(),
                                 [[int(i) for i in config['MODEL']['INDEX_MODEL_SIZE'].split(',')],
                                  [int(i) for i in config['MODEL']['TITLE_MODEL_SIZE'].split(',')]])

        fw_test = open(os.path.join(root_data_path, 'forecast_data', 'train', company.strip() + '.pkl'), 'wb')
        pickle.dump(data, fw_test)
        fw_test.close()

        data = forecast.forecast(os.path.join(root_data_path, 'data', 'test'), root_model_path, company.strip(),
                                 [[int(i) for i in config['MODEL']['INDEX_MODEL_SIZE'].split(',')],
                                  [int(i) for i in config['MODEL']['TITLE_MODEL_SIZE'].split(',')]])

        fw_test = open(os.path.join(root_data_path, 'forecast_data', 'test', company.strip() + '.pkl'), 'wb')
        pickle.dump(data, fw_test)
        fw_test.close()


def fitness_func(chromosome):
    root_path = '/'.join(os.path.dirname(os.path.abspath(__file__)).split('/'))
    root_data_path = os.path.join(root_path, 'data')
    company_list = open(os.path.join(root_data_path, 'data', 'company_list.txt')).readlines()

    data = []
    for company in company_list:
        fp = open(os.path.join(root_data_path, 'forecast_data', 'train', company.strip() + '.pkl'), 'rb')
        data.append(pickle.load(fp))
        fp.close()

    forecast_size = len(data[-1][0])
    c = 0
    for i in range(forecast_size):
        a, b = 0, 0
        temp = [d[0][i] for d in data]
        for j in temp:
            if j[0] > j[1]:
                a += 1
            else:
                b += 1

        if b < 5:
            c += 1
        print(a, b)
    print(c, forecast_size)


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('config.ini')

    root_path = '/'.join(os.path.dirname(os.path.abspath(__file__)).split('/'))

    # forecast.preprocess_data(root_path)

    root_data_path = os.path.join(root_path, 'data')
    root_model_path = os.path.join(root_path, 'trainedModel')
    company_list = open(os.path.join(root_data_path, 'data', 'company_list.txt')).readlines()

    # train_model(root_data_path, company_list)
    # save_accuracy(root_data_path, root_model_path, company_list)
    fitness_func([])
