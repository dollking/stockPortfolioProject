import os
import csv
import sys
import time
import pickle

import numpy as np


class Preprocessing(object):
    def __init__(self, root_dir):
        self.root_path = root_dir
        self.predictRange = 1
        self.predictStartPos = 51  # +1

        self.vec_path = os.path.join(self.root_path, 'data', 'doc2vec_pkl')
        self.train_path = os.path.join(self.root_path, 'data', 'raw_data_pkl', 'train')
        self.test_path = os.path.join(self.root_path, 'data', 'raw_data_pkl', 'test')
        self.save_path = os.path.join(self.root_path, 'data', 'data')

        self.train_data_path = os.path.join(self.root_path, 'data', 'raw_data_train')
        self.test_data_path = os.path.join(self.root_path, 'data', 'raw_data_test')
        self.savePath = os.path.join(self.root_path, 'data', 'pklData')

        self.train_date_list = list()
        self.test_date_list = list()

        self.linear_x = [i for i in range(1, 6)]

    def progress(self, rate, name):
        cnt = int(rate * 30)
        char = '#' * cnt + '-' * (30 - cnt)
        print('>>> make {} data : |{}|'.format(name, char), end='\r', flush=True)

    def rate(self, d1, d2):
        if d2 == 0:
            return 0 if d1 is 0 else d1 * 100

        return ((d1 - d2) / d2) * 100

    # 0: start, 1: high, 2: low, 3: close, 4: volume
    def Exp_MA(self, data, pos, size):
        weight = 2 / (size + 1)
        result = 0.0

        for i in range(size - 1, -1, -1):
            result += weight * data[i][pos]
            weight *= weight

        return result

    def MACD(self, raw_data, isTrain=True):
        data = []
        date_list = self.train_date_list if isTrain else self.test_date_list

        for i in range(self.predictStartPos, len(date_list)):
            temp = []
            for j in [1, 3, 5, 20, 25]:
                end_pos = i - j + 1
                temp.insert(0, self.Exp_MA([raw_data[date_list[j]] for j in range(end_pos - 12, end_pos)], 3, 12) -
                            self.Exp_MA([raw_data[date_list[j]] for j in range(end_pos - 26, end_pos)], 3, 26))

            data.append(temp)
        return data

    def DMI(self, raw_data, isTrain=True):
        data = []
        date_list = self.train_date_list if isTrain else self.test_date_list

        for i in range(self.predictStartPos, len(date_list)):
            temp = []
            for j in [1, 3, 5, 20, 25]:
                end_pos = i - j
                dmn_plus = 0.0
                dmn_minus = 0.0
                tr = 0.0
                for k in range(10):
                    high_data = (raw_data[date_list[end_pos - k]][1] - raw_data[date_list[end_pos - k - 1]][1])
                    low_data = (raw_data[date_list[end_pos - k - 1]][2] - raw_data[date_list[end_pos - k]][2])
                    dmn_plus += high_data if (high_data > 0 and high_data > low_data) else 0.0
                    dmn_minus += low_data if (low_data > 0 and high_data < low_data) else 0.0

                di_plus = dmn_plus / 10
                di_minus = dmn_minus / 10
                value = abs(di_plus - di_minus) * 100 / (di_plus + di_minus) if di_plus + di_minus else 0.
                temp.insert(0, value)

            data.append(temp)
        return data

    def volume_ratio(self, raw_data, isTrain=True):
        data = []
        date_list = self.train_date_list if isTrain else self.test_date_list

        for i in range(self.predictStartPos, len(date_list)):
            temp = []
            for j in [1, 3, 5, 20, 25]:
                end_pos = i - j
                sum_upper = 0.0
                sum_lower = 0.0
                sum_else = 0.0
                for k in range(20):
                    if raw_data[date_list[end_pos - k]][3] - raw_data[date_list[end_pos - k - 1]][3] > 0:
                        sum_upper += raw_data[date_list[end_pos - k]][4]
                    elif raw_data[date_list[end_pos - k]][3] - raw_data[date_list[end_pos - k - 1]][3] < 0:
                        sum_lower += raw_data[date_list[end_pos - k]][4]
                    else:
                        sum_else += raw_data[date_list[end_pos - k]][4]
                temp.insert(0, (sum_upper + (sum_else / 2)) * 100 / (0.01 + sum_lower + (sum_else / 2)))

            data.append(temp)
        return data

    def volume_moving_average(self, raw_data, isTrain=True):
        data = []
        date_list = self.train_date_list if isTrain else self.test_date_list

        for i in range(self.predictStartPos, len(date_list)):
            temp = []
            for j in [1, 3, 5, 20, 25]:
                end_pos = i - j
                temp.insert(0, sum([raw_data[date_list[end_pos - k]][4] for k in range(10)]) / 10)

            data.append(temp)
        return data

    def psy_line(self, raw_data, isTrain=True):
        data = []
        date_list = self.train_date_list if isTrain else self.test_date_list

        for i in range(self.predictStartPos, len(date_list)):
            temp = []
            for j in [1, 3, 5, 20, 25]:
                end_pos = i - j
                cnt = 1.0
                for k in range(10):
                    if raw_data[date_list[end_pos - k]][3] - raw_data[date_list[end_pos - k - 1]][3] > 0:
                        cnt += 1

                temp.insert(0, cnt * 100 / 10)

            data.append(temp)
        return data

    def RSI(self, raw_data, isTrain=True):
        data = []
        date_list = self.train_date_list if isTrain else self.test_date_list

        for i in range(self.predictStartPos, len(date_list)):
            temp = []
            for j in [1, 3, 5, 20, 25]:
                end_pos = i - j
                amount_upper = 0.0
                amount_lower = 0.0
                for k in range(9):
                    tmp_amount = raw_data[date_list[end_pos - k]][3] - raw_data[date_list[end_pos - k - 1]][3]
                    if tmp_amount > 0:
                        amount_upper += tmp_amount
                    elif tmp_amount < 0:
                        amount_lower -= tmp_amount

                value = amount_upper / (amount_upper + amount_lower) if amount_upper else 0.0
                temp.insert(0, value)

            data.append(temp)
        return data

    def ATR(self, raw_data, isTrain=True):
        data = []
        date_list = self.train_date_list if isTrain else self.test_date_list

        for i in range(self.predictStartPos, len(date_list)):
            temp = []
            for j in [1, 3, 5, 20, 25]:
                end_pos = i - j
                tr = 0.0
                for k in range(10):
                    tr += max(abs(raw_data[date_list[end_pos - k]][1] - raw_data[date_list[end_pos - k]][2]),
                              abs(raw_data[date_list[end_pos - k - 1]][3] - raw_data[date_list[end_pos - k]][1]),
                              abs(raw_data[date_list[end_pos - k - 1]][3] - raw_data[date_list[end_pos - k]][2]))

                temp.insert(0, tr / 10)

            data.append(temp)
        return data

    def envelope_upper(self, raw_data, isTrain=True):
        data = []
        date_list = self.train_date_list if isTrain else self.test_date_list

        for i in range(self.predictStartPos, len(date_list)):
            temp = []
            for j in [1, 3, 5, 20, 25]:
                end_pos = i - j
                middle = sum([raw_data[date_list[end_pos - k]][3] for k in range(10)]) / 10.

                temp.insert(0, middle * 1.1)    # alpha = 0.1

            data.append(temp)
        return data

    def make_target_data(self):
        cnt = 1
        for com in os.listdir(self.vec_path):
            train = pickle.load(open(os.path.join(self.train_path, com), 'rb'))
            test = pickle.load(open(os.path.join(self.test_path, com), 'rb'))

            data = list()
            for i in range(self.predictStartPos, len(train)):
                r = self.rate(train[self.train_date_list[i]][3], train[self.train_date_list[i - 1]][3])
                if r < 0.4:
                    data.append([1.0, 0.0])
                else:
                    data.append([0.0, 1.0])

            fw_test = open(os.path.join(self.save_path, 'train', 'target', com), 'wb')
            pickle.dump(data, fw_test)
            fw_test.close()

            data = list()
            for i in range(self.predictStartPos, len(test)):
                r = self.rate(test[self.test_date_list[i]][3], test[self.test_date_list[i - 1]][3])
                if r < 0.5:
                    data.append([1.0, 0.0])
                else:
                    data.append([0.0, 1.0])

            fw_test = open(os.path.join(self.save_path, 'test', 'target', com), 'wb')
            pickle.dump(data, fw_test)
            fw_test.close()

            self.progress(cnt / len(os.listdir(self.vec_path)), 'target')
            cnt += 1
        print()

    def make_title_data(self):
        cnt = 1
        for com in os.listdir(self.vec_path):
            vec = pickle.load(open(os.path.join(self.vec_path, com), 'rb'))

            data = [[], [], [], [], [], []]
            for i in range(self.predictStartPos, len(self.train_date_list)):
                for j in range(6):
                    data[j].append(vec[self.train_date_list[i - 6 + j]])

            fw_train = open(os.path.join(self.save_path, 'train', 'title', com), 'wb')
            pickle.dump(data, fw_train)
            fw_train.close()

            data = [[], [], [], [], [], []]
            for i in range(self.predictStartPos, len(self.test_date_list)):
                for j in range(6):
                    data[j].append(vec[self.test_date_list[i - 6 + j]])

            fw_test = open(os.path.join(self.save_path, 'test', 'title', com), 'wb')
            pickle.dump(data, fw_test)
            fw_test.close()

            self.progress(cnt / len(os.listdir(self.vec_path)), 'title')
            cnt += 1
        print()

    def make_index_data(self):
        cnt = 1
        for com in os.listdir(self.vec_path):
            train = pickle.load(open(os.path.join(self.train_path, com), 'rb'))
            test = pickle.load(open(os.path.join(self.test_path, com), 'rb'))

            temp = []
            temp.append(self.MACD(train))
            temp.append(self.DMI(train))
            temp.append(self.volume_ratio(train))
            temp.append(self.volume_moving_average(train))
            temp.append(self.psy_line(train))
            temp.append(self.RSI(train))
            temp.append(self.ATR(train))
            temp.append(self.envelope_upper(train))

            fw_train = open(os.path.join(self.save_path, 'train', 'index', com), 'wb')
            pickle.dump(temp, fw_train)
            fw_train.close()

            temp = []
            temp.append(self.MACD(test, False))
            temp.append(self.DMI(test, False))
            temp.append(self.volume_ratio(test, False))
            temp.append(self.volume_moving_average(test, False))
            temp.append(self.psy_line(test, False))
            temp.append(self.RSI(test, False))
            temp.append(self.ATR(test, False))
            temp.append(self.envelope_upper(test, False))

            fw_test = open(os.path.join(self.save_path, 'test', 'index', com), 'wb')
            pickle.dump(temp, fw_test)
            fw_test.close()

            self.progress(cnt / len(os.listdir(self.vec_path)), 'index')
            cnt += 1
        print()

    def preprocessing(self):
        tmp = pickle.load(open(os.path.join(self.train_path, 'LG.pkl'), 'rb'))
        self.train_date_list = list(tmp.keys())
        self.train_date_list.sort()

        tmp = pickle.load(open(os.path.join(self.test_path, 'LG.pkl'), 'rb'))
        self.test_date_list = list(tmp.keys())
        self.test_date_list.sort()

        self.make_index_data()
        self.make_title_data()
        self.make_target_data()
