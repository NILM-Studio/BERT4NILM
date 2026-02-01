from abc import *
from config import *
from dataloader import *

import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import torch.utils.data as data_utils


class AbstractDataset(metaclass=ABCMeta):
    def __init__(self, args, stats=None):
        self.house_indicies = args.house_indicies
        self.appliance_names = args.appliance_names
        self.normalize = args.normalize
        self.sampling = args.sampling
        self.cutoff = [args.cutoff[i]
                       for i in ['aggregate'] + self.appliance_names]

        self.threshold = [args.threshold[i] for i in self.appliance_names]
        self.min_on = [args.min_on[i] for i in self.appliance_names]
        self.min_off = [args.min_off[i] for i in self.appliance_names]

        self.val_size = args.validation_size
        self.window_size = args.window_size
        self.window_stride = args.window_stride

        self.x, self.y = self.load_data()
        self.status = self.compute_status(self.y)
        print('Appliance:', self.appliance_names)
        print('Sum of ons:', np.sum(self.status, axis=0))
        print('Total length:', self.status.shape[0])

        if stats is None:
            self.x_mean = np.mean(self.x, axis=0)
            self.x_std = np.std(self.x, axis=0)
        else: 
            self.x_mean, self.x_std = stats

        self.x = (self.x - self.x_mean) / self.x_std

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @classmethod
    def raw_code(cls):
        return cls.code()

    @abstractmethod
    def load_data(self):
        pass

    def get_data(self):
        return self.x, self.y, self.status

    def get_original_data(self):
        x_org = self.x * self.x_std + self.x_mean
        return x_org, self.y, self.status

    def get_mean_std(self):
        return self.x_mean, self.x_std

    def compute_status(self, data):
        status = np.zeros(data.shape)
        if len(data.squeeze().shape) == 1:
            columns = 1
        else:
            columns = data.squeeze().shape[-1]

        if not self.threshold:
            self.threshold = [10 for i in range(columns)]
        if not self.min_on:
            self.min_on = [1 for i in range(columns)]
        if not self.min_off:
            self.min_off = [1 for i in range(columns)]

        for i in range(columns):
            initial_status = data[:, i] >= self.threshold[i]
            status_diff = np.diff(initial_status)
            events_idx = status_diff.nonzero()

            events_idx = np.array(events_idx).squeeze()
            events_idx += 1

            if initial_status[0]:
                events_idx = np.insert(events_idx, 0, 0)

            if initial_status[-1]:
                events_idx = np.insert(
                    events_idx, events_idx.size, initial_status.size)

            events_idx = events_idx.reshape((-1, 2))
            on_events = events_idx[:, 0].copy()
            off_events = events_idx[:, 1].copy()
            assert len(on_events) == len(off_events)

            if len(on_events) > 0:
                off_duration = on_events[1:] - off_events[:-1]
                off_duration = np.insert(off_duration, 0, 1000)
                on_events = on_events[off_duration > self.min_off[i]]
                off_events = off_events[np.roll(
                    off_duration, -1) > self.min_off[i]]

                on_duration = off_events - on_events
                on_events = on_events[on_duration >= self.min_on[i]]
                off_events = off_events[on_duration >= self.min_on[i]]
                assert len(on_events) == len(off_events)

            temp_status = data[:, i].copy()
            temp_status[:] = 0
            for on, off in zip(on_events, off_events):
                temp_status[on: off] = 1
            status[:, i] = temp_status

        return status

    def get_status(self):
        return self.status

    def get_datasets(self):
        val_end = int(self.val_size * len(self.x))
        val = NILMDataset(self.x[:val_end], self.y[:val_end], self.status[:val_end],
                          self.window_size, self.window_size)
        train = NILMDataset(self.x[val_end:], self.y[val_end:], self.status[val_end:],
                            self.window_size, self.window_stride)
        return train, val

    def get_bert_datasets(self, mask_prob=0.25):
        val_end = int(self.val_size * len(self.x))
        val = NILMDataset(self.x[:val_end], self.y[:val_end], self.status[:val_end],
                          self.window_size, self.window_size)
        train = BERTDataset(self.x[val_end:], self.y[val_end:], self.status[val_end:],
                            self.window_size, self.window_stride, mask_prob=mask_prob)
        return train, val

    def _get_rawdata_root_path(self):
        return Path(RAW_DATASET_ROOT_FOLDER)

    def _get_folder_path(self):
        root = self._get_rawdata_root_path()
        return root.joinpath(self.raw_code())


class CSV_Dataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'csv_dataset'

    def load_data(self):
        csv_root = Path(RAW_DATASET_ROOT_FOLDER)
        
        train_csv = csv_root.joinpath('train.csv')
        val_csv = csv_root.joinpath('val.csv')
        test_csv = csv_root.joinpath('test.csv')
        
        if not train_csv.exists():
            raise FileNotFoundError(f'Training CSV not found: {train_csv}')
        
        print(f'Loading training data from {train_csv}...')
        train_data = pd.read_csv(train_csv)
        
        if val_csv.exists():
            print(f'Loading validation data from {val_csv}...')
            val_data = pd.read_csv(val_csv)
        else:
            print('Validation CSV not found, using portion of training data for validation')
            val_size = int(len(train_data) * self.val_size)
            val_data = train_data.iloc[:val_size].copy()
            train_data = train_data.iloc[val_size:].copy()
        
        if test_csv.exists():
            print(f'Loading test data from {test_csv}...')
            test_data = pd.read_csv(test_csv)
        else:
            print('Test CSV not found, using validation data for testing')
            test_data = val_data.copy()
        
        print(f'Train data shape: {train_data.shape}')
        print(f'Val data shape: {val_data.shape}')
        print(f'Test data shape: {test_data.shape}')
        
        print(f'Columns in CSV: {train_data.columns.tolist()}')
        
        timestamp_col = 'timestamp' if 'timestamp' in train_data.columns else train_data.columns[0]
        aggregate_col = 'aggregate' if 'aggregate' in train_data.columns else train_data.columns[1]
        
        print(f'Timestamp column: {timestamp_col}')
        print(f'Aggregate column: {aggregate_col}')
        print(f'Appliance columns: {self.appliance_names}')
        
        for appliance in self.appliance_names:
            if appliance not in train_data.columns:
                raise ValueError(f'Appliance {appliance} not found in CSV columns')
        
        train_data = self._preprocess_csv_data(train_data, timestamp_col, aggregate_col)
        val_data = self._preprocess_csv_data(val_data, timestamp_col, aggregate_col)
        test_data = self._preprocess_csv_data(test_data, timestamp_col, aggregate_col)
        
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        
        x_train = train_data[aggregate_col].values
        y_train = train_data[self.appliance_names].values
        
        return x_train, y_train
    
    def _preprocess_csv_data(self, data, timestamp_col, aggregate_col):
        data = data.copy()
        
        if timestamp_col != 'time':
            data = data.rename(columns={timestamp_col: 'time'})
        
        if aggregate_col != 'aggregate':
            data = data.rename(columns={aggregate_col: 'aggregate'})
        
        if 'time' in data.columns:
            try:
                data['time'] = pd.to_datetime(data['time'])
                data = data.set_index('time')
            except:
                print('Warning: Could not parse timestamp column, using index instead')
        
        data = data.dropna()
        data = data[data['aggregate'] > 0]
        data[data < 5] = 0
        
        cutoff_dict = {col: self.cutoff[i] for i, col in enumerate(['aggregate'] + self.appliance_names)}
        for col in data.columns:
            if col in cutoff_dict:
                data[col] = data[col].clip(0, cutoff_dict[col])
        
        return data
    
    def get_test_data(self):
        if not hasattr(self, 'test_data'):
            raise ValueError('Test data not loaded')
        
        aggregate_col = 'aggregate' if 'aggregate' in self.test_data.columns else self.test_data.columns[1]
        x_test = self.test_data[aggregate_col].values
        y_test = self.test_data[self.appliance_names].values
        
        x_test = (x_test - self.x_mean) / self.x_std
        
        test_dataset = NILMDataset(x_test, y_test, np.zeros_like(y_test),
                                   self.window_size, self.window_size)
        
        return test_dataset


class REDD_LF_Dataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'redd_lf'

    @classmethod
    def _if_data_exists(self):
        folder = Path(RAW_DATASET_ROOT_FOLDER).joinpath(self.code())
        first_file = folder.joinpath('house_1', 'channel_1.dat')
        if first_file.is_file():
            return True
        return False

    def load_data(self):
        for appliance in self.appliance_names:
            assert appliance in ['dishwasher',
                                 'refrigerator', 'microwave', 'washer_dryer']

        for house_id in self.house_indicies:
            assert house_id in [1, 2, 3, 4, 5, 6]

        if not self.cutoff:
            self.cutoff = [6000] * (len(self.appliance_names) + 1)

        if not self._if_data_exists():
            print('Please download, unzip and move data into',
                  self._get_folder_path())
            raise FileNotFoundError

        else:
            directory = self._get_folder_path()

            for house_id in self.house_indicies:
                house_folder = directory.joinpath('house_' + str(house_id))
                house_label = pd.read_csv(house_folder.joinpath(
                    'labels.dat'), sep=' ', header=None)

                main_1 = pd.read_csv(house_folder.joinpath(
                    'channel_1.dat'), sep=' ', header=None)
                main_2 = pd.read_csv(house_folder.joinpath(
                    'channel_2.dat'), sep=' ', header=None)
                house_data = pd.merge(main_1, main_2, how='inner', on=0)
                house_data.iloc[:, 1] = house_data.iloc[:,
                                                        1] + house_data.iloc[:, 2]
                house_data = house_data.iloc[:, 0: 2]

                appliance_list = house_label.iloc[:, 1].values
                app_index_dict = defaultdict(list)

                for appliance in self.appliance_names:
                    data_found = False
                    for i in range(len(appliance_list)):
                        if appliance_list[i] == appliance:
                            app_index_dict[appliance].append(i + 1)
                            data_found = True

                    if not data_found:
                        app_index_dict[appliance].append(-1)

                if np.sum(list(app_index_dict.values())) == -len(self.appliance_names):
                    self.house_indicies.remove(house_id)
                    continue

                for appliance in self.appliance_names:
                    if app_index_dict[appliance][0] == -1:
                        temp_values = house_data.copy().iloc[:, 1]
                        temp_values[:] = 0
                        temp_data = house_data.copy().iloc[:, :2]
                        temp_data.iloc[:, 1] = temp_values
                    else:
                        temp_data = pd.read_csv(house_folder.joinpath(
                            'channel_' + str(app_index_dict[appliance][0]) + '.dat'), sep=' ', header=None)

                    if len(app_index_dict[appliance]) > 1:
                        for idx in app_index_dict[appliance][1:]:
                            temp_data_ = pd.read_csv(house_folder.joinpath(
                                'channel_' + str(idx) + '.dat'), sep=' ', header=None)
                            temp_data = pd.merge(
                                temp_data, temp_data_, how='inner', on=0)
                            temp_data.iloc[:, 1] = temp_data.iloc[:,
                                                                  1] + temp_data.iloc[:, 2]
                            temp_data = temp_data.iloc[:, 0: 2]

                    house_data = pd.merge(
                        house_data, temp_data, how='inner', on=0)

                house_data.iloc[:, 0] = pd.to_datetime(
                    house_data.iloc[:, 0], unit='s')
                house_data.columns = ['time', 'aggregate'] + \
                    [i for i in self.appliance_names]
                house_data = house_data.set_index('time')
                house_data = house_data.resample(self.sampling).mean().fillna(
                    method='ffill', limit=30)

                if house_id == self.house_indicies[0]:
                    entire_data = house_data
                else:
                    entire_data = entire_data.append(
                        house_data, ignore_index=True)

                entire_data = entire_data.dropna().copy()
                entire_data = entire_data[entire_data['aggregate'] > 0]
                entire_data[entire_data < 5] = 0
                entire_data = entire_data.clip(
                    [0] * len(entire_data.columns), self.cutoff, axis=1)

            return entire_data.values[:, 0], entire_data.values[:, 1:]


class UK_DALE_Dataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'uk_dale'

    @classmethod
    def _if_data_exists(self):
        folder = Path(RAW_DATASET_ROOT_FOLDER).joinpath(self.code())
        first_file = folder.joinpath('house_1', 'channel_1.dat')
        if first_file.is_file():
            return True
        return False

    def load_data(self):
        for appliance in self.appliance_names:
            assert appliance in ['dishwasher', 'fridge',
                                 'microwave', 'washing_machine', 'kettle', 'air-condition']

        for house_id in self.house_indicies:
            assert house_id in [1, 2, 3, 4, 5]

        if not self.cutoff:
            self.cutoff = [6000] * (len(self.appliance_names) + 1)

        if not self._if_data_exists():
            print('Please download, unzip and move data into',
                  self._get_folder_path())
            raise FileNotFoundError

        else:
            directory = self._get_folder_path()

            for house_id in self.house_indicies:
                print(f'loading house_' + str(house_id) + '...')
                house_folder = directory.joinpath('house_' + str(house_id))
                house_label = pd.read_csv(house_folder.joinpath(
                    'labels.dat'), sep=' ', header=None)

                house_data = pd.read_csv(house_folder.joinpath(
                    'channel_1.dat'), sep=' ', header=None)
                house_data.iloc[:, 0] = pd.to_datetime(
                    house_data.iloc[:, 0], unit='s')
                house_data.columns = ['time', 'aggregate']
                house_data = house_data.set_index('time')
                house_data = house_data.resample(self.sampling).mean().fillna(
                    method='ffill', limit=30)

                appliance_list = house_label.iloc[:, 1].values
                app_index_dict = defaultdict(list)

                for appliance in self.appliance_names:
                    data_found = False
                    for i in range(len(appliance_list)):
                        if appliance_list[i] == appliance:
                            app_index_dict[appliance].append(i + 1)
                            data_found = True

                    if not data_found:
                        app_index_dict[appliance].append(-1)

                if np.sum(list(app_index_dict.values())) == -len(self.appliance_names):
                    self.house_indicies.remove(house_id)
                    continue

                for appliance in self.appliance_names:
                    if app_index_dict[appliance][0] == -1:
                        house_data.insert(len(house_data.columns), appliance, np.zeros(len(house_data)))
                    else:
                        file_path = house_folder.joinpath(
                            'channel_' + str(app_index_dict[appliance][0]) + '.dat')
                        print(f'---------loading data from file {file_path}...----------\n')
                        temp_data = pd.read_csv(house_folder.joinpath(
                            'channel_' + str(app_index_dict[appliance][0]) + '.dat'), sep=' ', header=None)
                        temp_data.iloc[:, 0] = pd.to_datetime(
                            temp_data.iloc[:, 0], unit='s')
                        temp_data.columns = ['time', appliance]
                        temp_data = temp_data.set_index('time')
                        temp_data = temp_data.resample(self.sampling).mean().fillna(
                            method='ffill', limit=30)
                        house_data = pd.merge(
                            house_data, temp_data, how='inner', on='time')

                if house_id == self.house_indicies[0]:
                    entire_data = house_data
                else:
                    entire_data = entire_data.append(
                        house_data, ignore_index=True)

            entire_data = entire_data.dropna().copy()
            entire_data = entire_data[entire_data['aggregate'] > 0]
            entire_data[entire_data < 5] = 0
            entire_data = entire_data.clip(
                [0] * len(entire_data.columns), self.cutoff, axis=1)

        return entire_data.values[:, 0], entire_data.values[:, 1:]
