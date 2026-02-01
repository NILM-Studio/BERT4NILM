import torch
import numpy as np
from sklearn.metrics import confusion_matrix


def get_user_input(args):
    if torch.cuda.is_available():
        args.device = 'cuda:' + input('Input GPU ID: ')
    else:
        args.device = 'cpu'

    dataset_code = {'r': 'redd_lf', 'u': 'uk_dale', 'c': 'csv_dataset'}
    args.dataset_code = dataset_code[input(
        'Input r for REDD, u for UK_DALE, c for CSV: ')]

    if args.dataset_code == 'redd_lf':
        app_dict = {
            'r': ['refrigerator'],
            'w': ['washer_dryer'],
            'm': ['microwave'],
            'd': ['dishwasher'],
        }
        args.appliance_names = app_dict[input(
            'Input r, w, m or d for target appliance: ')]

    elif args.dataset_code == 'uk_dale':
        app_dict = {
            'k': ['kettle'],
            'f': ['fridge'],
            'w': ['washing_machine'],
            'm': ['microwave'],
            'd': ['dishwasher'],
            'a': ['air-condition'],
        }
        args.appliance_names = app_dict[input(
            'Input k, f, w, m ,d or a for target appliance: ')]

    elif args.dataset_code == 'csv_dataset':
        # 从 CSV 文件中自动读取电器名称
        from config import RAW_DATASET_ROOT_FOLDER
        import pandas as pd
        from pathlib import Path
        
        csv_root = Path(RAW_DATASET_ROOT_FOLDER)
        train_csv = csv_root.joinpath('train.csv')
        
        if not train_csv.exists():
            raise FileNotFoundError(f'Training CSV not found: {train_csv}')
        
        print(f'Loading training data from {train_csv} to detect appliance names...')
        train_data = pd.read_csv(train_csv)
        
        # 获取所有列名
        all_columns = train_data.columns.tolist()
        # 过滤掉非电器列
        non_appliance_columns = ['timestamp', 'time', 'aggregate']
        args.appliance_names = [col for col in all_columns if col not in non_appliance_columns]
        
        print(f'Automatically detected appliance names: {args.appliance_names}')
        print('You can press Enter to confirm or input custom appliance names (comma separated): ')
        
        # 允许用户确认或输入自定义电器名称
        appliance_input = input()
        if appliance_input.strip():
            args.appliance_names = [name.strip() for name in appliance_input.split(',')]
            print(f'Using custom appliance names: {args.appliance_names}')

    args.num_epochs = int(input('Input training epochs: '))


def set_template(args):
    args.output_size = len(args.appliance_names)
    if args.dataset_code == 'redd_lf':
        args.window_stride = 120
        args.house_indicies = [1, 2, 3, 4, 5, 6]

        args.cutoff = {
            'aggregate': 6000,
            'refrigerator': 400,
            'washer_dryer': 3500,
            'microwave': 1800,
            'dishwasher': 1200,
        }

        args.threshold = {
            'refrigerator': 50,
            'washer_dryer': 20,
            'microwave': 200,
            'dishwasher': 10
        }

        args.min_on = {
            'refrigerator': 10,
            'washer_dryer': 300,
            'microwave': 2,
            'dishwasher': 300
        }

        args.min_off = {
            'refrigerator': 2,
            'washer_dryer': 26,
            'microwave': 5,
            'dishwasher': 300
        }

        args.c0 = {
            'refrigerator': 1e-6,
            'washer_dryer': 0.001,
            'microwave': 1.,
            'dishwasher': 1.
        }

    elif args.dataset_code == 'uk_dale':
        args.window_stride = 240
        args.house_indicies = [1, 2, 3, 4, 5]
        
        args.cutoff = {
            'aggregate': 6000,
            'kettle': 3100,
            'fridge': 300,
            'washing_machine': 2500,
            'microwave': 3000,
            'dishwasher': 2500,
            'air-condition': 2000,
        }

        args.threshold = {
            'kettle': 2000,
            'fridge': 50,
            'washing_machine': 20,
            'microwave': 200,
            'dishwasher': 10,
            'air-condition': 20,
        }

        args.min_on = {
            'kettle': 2,
            'fridge': 10,
            'washing_machine': 300,
            'microwave': 2,
            'dishwasher': 300,
            'air-condition': 20,
        }

        args.min_off = {
            'kettle': 0,
            'fridge': 2,
            'washing_machine': 26,
            'microwave': 5,
            'dishwasher': 300,
            'air-condition': 0,
        }

        args.c0 = {
            'kettle': 1.,
            'fridge': 1e-6,
            'washing_machine': 0.01,
            'microwave': 1.,
            'dishwasher': 1.,
            'air-condition': 0.001,
        }

    elif args.dataset_code == 'csv_dataset':
        args.window_stride = 240
        args.house_indicies = [1]
        
        args.cutoff = {
            'aggregate': 6000,
            'kettle': 3100,
            'fridge': 300,
            'washing_machine': 2500,
            'microwave': 3000,
            'dishwasher': 2500,
            'air-condition': 2000,
        }

        args.threshold = {
            'kettle': 2000,
            'fridge': 50,
            'washing_machine': 20,
            'microwave': 200,
            'dishwasher': 10,
            'air-condition': 20,
        }

        args.min_on = {
            'kettle': 2,
            'fridge': 10,
            'washing_machine': 300,
            'microwave': 2,
            'dishwasher': 300,
            'air-condition': 20,
        }

        args.min_off = {
            'kettle': 0,
            'fridge': 2,
            'washing_machine': 26,
            'microwave': 5,
            'dishwasher': 300,
            'air-condition': 0,
        }

        args.c0 = {
            'kettle': 1.,
            'fridge': 1e-6,
            'washing_machine': 0.01,
            'microwave': 1.,
            'dishwasher': 1.,
            'air-condition': 0.001,
        }
        
        # 为自动检测到的电器添加默认参数
        for appliance in args.appliance_names:
            if appliance not in args.cutoff:
                args.cutoff[appliance] = 6000
                args.threshold[appliance] = 50
                args.min_on[appliance] = 10
                args.min_off[appliance] = 2
                args.c0[appliance] = 1.0

    args.optimizer = 'adam'
    args.lr = 1e-4
    args.enable_lr_schedule = False
    args.batch_size = 128


def acc_precision_recall_f1_score(pred, status):
    assert pred.shape == status.shape

    pred = pred.reshape(-1, pred.shape[-1])
    status = status.reshape(-1, status.shape[-1])
    accs, precisions, recalls, f1_scores = [], [], [], []

    for i in range(status.shape[-1]):
        tn, fp, fn, tp = confusion_matrix(status[:, i], pred[:, i], labels=[
                                          0, 1]).ravel()
        acc = (tn + tp) / (tn + fp + fn + tp)
        precision = tp / np.max((tp + fp, 1e-9))
        recall = tp / np.max((tp + fn, 1e-9))
        f1_score = 2 * (precision * recall) / \
            np.max((precision + recall, 1e-9))

        accs.append(acc)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)

    return np.array(accs), np.array(precisions), np.array(recalls), np.array(f1_scores)


def relative_absolute_error(pred, label):
    assert pred.shape == label.shape

    pred = pred.reshape(-1, pred.shape[-1])
    label = label.reshape(-1, label.shape[-1])
    temp = np.full(label.shape, 1e-9)
    relative, absolute, sum_err = [], [], []

    for i in range(label.shape[-1]):
        relative_error = np.mean(np.nan_to_num(np.abs(label[:, i] - pred[:, i]) / np.max(
            (label[:, i], pred[:, i], temp[:, i]), axis=0)))
        absolute_error = np.mean(np.abs(label[:, i] - pred[:, i]))

        relative.append(relative_error)
        absolute.append(absolute_error)

    return np.array(relative), np.array(absolute)
