import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import  MinMaxScaler
import datetime

from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')





class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val',"whole"]
        type_map = {'train': 0, 'val': 1, 'test': 2,"whole":3}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.flag = flag
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        self.scaler2 = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        border1s = [0, 12 * 30 * 24 * 4- self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len,12 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24  * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 12 * 30 * 24 * 4]
        print(border1s,border2s)
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
            self.scaler2.fit(df_raw[[self.target]].values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]

        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        self.data_stamp_row = df_stamp.date.apply(self.data_stamp_trans, 1)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(labels=['date'], axis = 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp


    def __getitem__(self, index):
        s_begin = index * 96
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        seq_x_t = self.data_stamp_row[s_begin:s_end]
        seq_y_t = self.data_stamp_row[r_begin:r_end]


        if self.flag == "whole":
            return seq_x, seq_y, seq_x_mark, seq_y_mark,seq_x_t,seq_y_t
        else:
            return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return int((self.data_x.shape[0] - self.seq_len - self.pred_len + 1)/96)

    def inverse_transform(self, data):
        return self.scaler2.inverse_transform(data.reshape([-1,1]))

    def data_stamp_trans(self,row):
        return datetime.datetime.timestamp(row)

    def data_stamp_reverse(self,rows):
        new_list = []
        for i in rows:
            new_list.append(datetime.datetime.fromtimestamp(i))
        return new_list

