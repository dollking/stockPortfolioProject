import os
import pickle
import configparser
from ga.ga import GeneticAlgorithm
from ga.chromosome import Chromosome

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
    up_down_rate = {}
    for company in company_list:
        fp = open(os.path.join(root_data_path, 'forecast_data', 'train', company.strip() + '.pkl'), 'rb')
        data.append([company.strip()] + list(pickle.load(fp)))
        fp.close()
        fp = open(os.path.join(root_data_path, 'genetic_data', 'train', company.strip() + '.pkl'), 'rb')
        up_down_rate[company.strip()] = pickle.load(fp)
        fp.close()

    investment_cnt = 0
    investment_yield = 0.0
    profit_investment_cnt = 0
    correct_cnt = 0
    ttt = []
    forecast_size = len(data[-1][1])
    for i in range(forecast_size):
        up_list = []
        for j in data:
            if j[1][i][1] > j[1][i][0]:
                up_list.append([j[0], j[1][i][1], j[2], j[3], j[4]])
        if len(up_list) < 5:
            continue
        up_list.sort(key=lambda x: chromosome[0]*x[1] + chromosome[1]*x[2] + chromosome[2]*x[3] + chromosome[3]*x[4],
                     reverse=True)
        investment_cnt += 1

        tmp_investment_yield = 0.0
        for j in up_list[:4]:
            tmp_rate = up_down_rate[j[0]][i]
            if tmp_rate >= 0.38:
                correct_cnt += 1

            tmp_investment_yield += tmp_rate

        investment_yield += tmp_investment_yield
        ttt.append(tmp_investment_yield)
        if tmp_investment_yield >= 0.38:
            profit_investment_cnt += 1

    investment_yield /= investment_cnt
    correct_cnt /= investment_cnt
    profit_investment_cnt /= investment_cnt
    # print(investment_yield, correct_cnt, profit_investment_cnt)
    # print(min(ttt), max(ttt))
    return investment_yield + (correct_cnt * profit_investment_cnt * 1.2)


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

    # chromosome = Chromosome()
    # chromosome.add_gene('float', [0.5, 10])
    # chromosome.add_gene('float', [0.5, 10])
    # chromosome.add_gene('float', [0.5, 10])
    # chromosome.add_gene('float', [0.5, 10])
    #
    # # make object for genetic algorithm
    # genetic = GeneticAlgorithm(fitness_func, 15, chromosome, 100, thread_count=3)
    #
    # # set operation using genetic algorithm
    # genetic.add_method('selection', 'ranking')
    # genetic.add_method('crossover', 'one_point')
    # genetic.add_method('mutation', 'elitist')
    # genetic.add_method('survivor', 'elitist', {'exception_rate': 0.1})
    #
    # genetic.run()
    # print(genetic.population[0])

    fitness_func([1.7397293962968536, 1.4323126547408116, 0.7695660545768713, 6.257239205775543])

    # fitness_func([8.465540576047815, 7.324827600077752, 4.339734397951959, 0.8731353472924104])