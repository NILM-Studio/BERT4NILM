import torch
from dataset import *
from dataloader import *
from trainer import *
from config import *
from utils import *
from model import BERT4NILM


def test_model(test_house_id, model_path, args):
    args.validation_size = 1.0
    args.house_indicies = [test_house_id]
    dataloader = NILMDataloader(args, None, bert=True)
    # 加载训练时的统计数据
    # 这里需要加载训练时计算的stats
    dataset = dataloader.get_dataset(args, stats)  # 需要根据数据集类型选择

    dataloader = NILMDataloader(args, dataset)
    _, test_loader = dataloader.get_dataloaders()

    model = BERT4NILM(args)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))

    trainer = Trainer(args, model, None, None, stats, None)
    rel_err, abs_err, acc, prec, recall, f1 = trainer.test(test_loader)

    print('Mean Accuracy:', acc)
    print('Mean F1-Score:', f1)
    print('Mean Relative Error:', rel_err)
    print('Mean Absolute Error:', abs_err)
