import os
import sys
import time
import csv
import pickle
import numpy as np


class Preprocessing(object):
    def __init__(self):
        self.current_dir = os.getcwd()
        self.predictRange = 1
        self.predictStartPos = 51  # +1

        self.vec_path = './data/doc2vec_pkl'
        self.train_path = './data/raw_data_pkl/train'
        self.test_path = './data/raw_data_pkl/test'
        self.save_path = './data/data'

        self.train_data_path = os.path.join(self.current_dir, 'data', 'raw_data_train')
        self.test_data_path = os.path.join(self.current_dir, 'data', 'raw_data_test')
        self.savePath = os.path.join(self.current_dir, 'data', 'pklData')

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
    def MA(self, data, pos):
        temp = [i[pos] for i in data]
        return sum(temp) / len(temp)

    def Exp_MA(self, data, pos, size):
        weight = 2 / (size + 1)
        result = 0.0

        for i in range(size - 1, -1, -1):
            result += weight * data[i][pos]
            weight *= weight

        return result

    def WMA(self, data, pos):
        temp = [(i + 1) * data[i][pos] for i in range(len(data))]
        return sum(temp) / len(temp)

    def linear(self, data):
        y = [data[i][3] for i in range(5)]
        return np.polyfit(self.linear_x, y, 1)[0]

    def MACD(self, raw_data, isTrain=True):
        data = []
        date_list = self.train_date_list if isTrain else self.test_date_list

        for i in range(self.predictStartPos, len(date_list)):
            temp = []
            for j in range([0, 2, 4, 19, 24]):
                end_pos = i - j
                tmp_date1 = date_list[i - j]
                tmp_date2 = date_list[i - j - 1]
                self.Exp_MA([raw_data[date_list[i]] for i in range(end_pos - 12, end_pos)], 3, 12)
                self.Exp_MA([raw_data[date_list[i]] for i in range(end_pos - 26, end_pos)], 3, 26)
                temp.insert(0, [self.rate(raw_data[tmp_date1][0], raw_data[tmp_date2][0]),
                                self.rate(raw_data[tmp_date1][1], raw_data[tmp_date2][1]),
                                self.rate(raw_data[tmp_date1][2], raw_data[tmp_date2][3]),
                                self.rate(raw_data[tmp_date1][3], raw_data[tmp_date2][3]),
                                self.rate(raw_data[tmp_date1][4], raw_data[tmp_date2][4])]
                            )

            data.append(temp)
        return data

    def makeMVData(self, raw_data, isTrain=True):
        data1 = []
        data2 = []
        date_list = self.train_date_list if isTrain else self.test_date_list

        for i in range(self.predictStartPos, len(date_list)):
            temp1 = []
            temp2 = []
            for j in range(26):
                pos = i - j
                temp1.insert(0, [self.rate(self.MA([raw_data[date_list[i]] for i in range(pos - 5, pos)], 3),
                                           self.MA([raw_data[date_list[i]] for i in range(pos - 6, pos - 1)], 3)),
                                 self.rate(self.MA([raw_data[date_list[i]] for i in range(pos - 10, pos)], 3),
                                           self.MA([raw_data[date_list[i]] for i in range(pos - 11, pos - 1)], 3)),
                                 self.rate(self.MA([raw_data[date_list[i]] for i in range(pos - 15, pos)], 3),
                                           self.MA([raw_data[date_list[i]] for i in range(pos - 16, pos - 1)], 3)),
                                 self.rate(self.MA([raw_data[date_list[i]] for i in range(pos - 20, pos)], 3),
                                           self.MA([raw_data[date_list[i]] for i in range(pos - 21, pos - 1)], 3))]
                             )

                temp2.insert(0, [self.rate(self.MA([raw_data[date_list[i]] for i in range(pos - 5, pos)], 4),
                                           self.MA([raw_data[date_list[i]] for i in range(pos - 6, pos - 1)], 4)),
                                 self.rate(self.MA([raw_data[date_list[i]] for i in range(pos - 10, pos)], 4),
                                           self.MA([raw_data[date_list[i]] for i in range(pos - 11, pos - 1)], 4)),
                                 self.rate(self.MA([raw_data[date_list[i]] for i in range(pos - 15, pos)], 4),
                                           self.MA([raw_data[date_list[i]] for i in range(pos - 16, pos - 1)], 4)),
                                 self.rate(self.MA([raw_data[date_list[i]] for i in range(pos - 20, pos)], 4),
                                           self.MA([raw_data[date_list[i]] for i in range(pos - 21, pos - 1)], 4))]
                             )

            data1.append(temp1)
            data2.append(temp2)

        return [data1, data2]  # close, volume

    def makeWMVData(self, raw_data, isTrain=True):
        data1 = []
        data2 = []
        date_list = self.train_date_list if isTrain else self.test_date_list

        for i in range(self.predictStartPos, len(raw_data)):
            temp1 = []
            temp2 = []
            for j in range(26):
                pos = i - j

                temp1.insert(0, [self.rate(self.WMA([raw_data[date_list[i]] for i in range(pos - 5, pos)], 3),
                                           self.WMA([raw_data[date_list[i]] for i in range(pos - 6, pos - 1)], 3)),
                                 self.rate(self.WMA([raw_data[date_list[i]] for i in range(pos - 10, pos)], 3),
                                           self.WMA([raw_data[date_list[i]] for i in range(pos - 11, pos - 1)], 3)),
                                 self.rate(self.WMA([raw_data[date_list[i]] for i in range(pos - 15, pos)], 3),
                                           self.WMA([raw_data[date_list[i]] for i in range(pos - 16, pos - 1)], 3)),
                                 self.rate(self.WMA([raw_data[date_list[i]] for i in range(pos - 20, pos)], 3),
                                           self.WMA([raw_data[date_list[i]] for i in range(pos - 21, pos - 1)], 3))]
                             )

                temp2.insert(0, [self.rate(self.WMA([raw_data[date_list[i]] for i in range(pos - 5, pos)], 4),
                                           self.WMA([raw_data[date_list[i]] for i in range(pos - 6, pos - 1)], 4)),
                                 self.rate(self.WMA([raw_data[date_list[i]] for i in range(pos - 10, pos)], 4),
                                           self.WMA([raw_data[date_list[i]] for i in range(pos - 11, pos - 1)], 4)),
                                 self.rate(self.WMA([raw_data[date_list[i]] for i in range(pos - 15, pos)], 4),
                                           self.WMA([raw_data[date_list[i]] for i in range(pos - 16, pos - 1)], 4)),
                                 self.rate(self.WMA([raw_data[date_list[i]] for i in range(pos - 20, pos)], 4),
                                           self.WMA([raw_data[date_list[i]] for i in range(pos - 21, pos - 1)], 4))]
                             )

            data1.append(temp1)
            data2.append(temp2)

        return [data1, data2]  # close, volume

    def makeLinearData(self, raw_data, isTrain=True):
        data = []
        date_list = self.train_date_list if isTrain else self.test_date_list

        for i in range(self.predictStartPos, len(raw_data)):
            temp = list()
            for j in range(26):
                pos = i - j

                temp.insert(0, [self.rate(self.linear([raw_data[date_list[i]] for i in range(pos - 5, pos)]),
                                          self.linear([raw_data[date_list[i]] for i in range(pos - 8, pos - 3)])),
                                self.rate(self.linear([raw_data[date_list[i]] for i in range(pos - 5, pos)]),
                                          self.linear([raw_data[date_list[i]] for i in range(pos - 10, pos - 5)])),
                                self.rate(self.linear([raw_data[date_list[i]] for i in range(pos - 5, pos)]),
                                          self.linear([raw_data[date_list[i]] for i in range(pos - 13, pos - 8)])),
                                self.rate(self.linear([raw_data[date_list[i]] for i in range(pos - 5, pos)]),
                                          self.linear([raw_data[date_list[i]] for i in range(pos - 18, pos - 3)]))]
                            )

            data.append(temp)
        return data     # close

    def make_target_data(self):
        cnt = 1
        for com in os.listdir(self.vec_path):
            train = pickle.load(open(os.path.join(self.train_path, com), 'rb'))
            test = pickle.load(open(os.path.join(self.test_path, com), 'rb'))

            data = list()
            for i in range(self.predictStartPos, len(train)):
                r = self.rate(train[self.train_date_list[i]][3], train[self.train_date_list[i - 1]][3])
                if r < 0.5:
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
            train = pickle.load(open(os.path.join(self.train_path, com), 'rb'))
            test = pickle.load(open(os.path.join(self.test_path, com), 'rb'))

            data = [[], [], [], [], [], []]
            target = []
            for i in range(self.predictStartPos, len(self.train_date_list)):
                for j in range(6):
                    data[j].append(vec[self.train_date_list[i - 6 + j]])

                target.append(
                    [1.0, 0.0] if (train[self.train_date_list[i]][3] - train[self.train_date_list[i - 1]][3]) /
                                  train[self.train_date_list[i - 1]][3] < 0.005 else [0.0, 1.0]
                )

            fw_train = open(os.path.join(self.save_path, 'train', 'title', com), 'wb')
            pickle.dump(data, fw_train)
            fw_train.close()

            data = [[], [], [], [], [], []]
            target = []
            for i in range(self.predictStartPos, len(self.test_date_list)):
                for j in range(6):
                    data[j].append(vec[self.test_date_list[i - 6 + j]])

                target.append(
                    [1.0, 0.0] if (test[self.test_date_list[i]][3] - test[self.test_date_list[i - 1]][3]) /
                                  test[self.test_date_list[i - 1]][3] < 0.005 else [0.0, 1.0]
                )

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
            temp.extend(self.makeMVData(train))
            temp.extend(self.makeWMVData(train))
            temp.append(self.makeLinearData(train))

            fw_train = open(os.path.join(self.save_path, 'train', 'index', com), 'wb')
            pickle.dump(temp, fw_train)
            fw_train.close()

            temp = []
            temp.append(self.MACD(test, False))
            temp.extend(self.makeMVData(test, False))
            temp.extend(self.makeWMVData(test, False))
            temp.append(self.makeLinearData(test, False))

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

        self.make_title_data()
        self.make_index_data()
        self.make_target_data()