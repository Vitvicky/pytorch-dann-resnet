import os
import sys
import datetime

import torch
sys.path.append(os.path.abspath('.'))
from models.model import SVHNmodel
from core.train import train_dann
from utils.utils import get_data_loader, init_model, init_random_seed
from utils.altutils import setLogger


class Config(object):
    # params for path
    currentDir = os.path.dirname(os.path.realpath(__file__))
    dataset_root = os.environ["DATASETDIR"]
    model_root = os.path.join(currentDir, 'checkpoints')
    config = os.path.join(model_root, 'config.txt')
    finetune_flag = False
    lr_adjust_flag = 'simple'
    src_only_flag = False

    # params for datasets and data loader
    batch_size = 128

    # params for source dataset
    src_dataset = "svhn"
    src_model_trained = True
    src_classifier_restore = os.path.join(model_root, src_dataset + '-source-classifier-final.pt')

    # params for target dataset
    tgt_dataset = "mnist"
    tgt_model_trained = True
    dann_restore = os.path.join(model_root, src_dataset + '-' + tgt_dataset + '-dann-final.pt')

    # params for training dann
    gpu_id = '0'

    ## for digit
    num_epochs = 200
    log_step = 50
    save_step = 100
    eval_step = 1

    ## for office
    # num_epochs = 1000
    # log_step = 10  # iters
    # save_step = 500
    # eval_step = 5  # epochs

    manual_seed = None
    alpha = 0

    # params for optimizing models
    lr = 0.01
    momentum = 0.9
    weight_decay = 1e-6

params = Config()

currentDir = os.path.dirname(os.path.realpath(__file__))
logFile = os.path.join(currentDir+'/../', 'dann-{}-{}.log'.format(params.src_dataset, params.tgt_dataset))
loggi = setLogger(logFile)

device = torch.device("cuda:" + params.gpu_id if torch.cuda.is_available() else "cpu")

# init random seed
init_random_seed(params.manual_seed)

# load dataset
src_data_loader = get_data_loader(params.src_dataset, params.dataset_root, params.batch_size, train=True)
src_data_loader_eval = get_data_loader(params.src_dataset, params.dataset_root, params.batch_size, train=False)
tgt_data_loader = get_data_loader(params.tgt_dataset, params.dataset_root, params.batch_size, train=True)
tgt_data_loader_eval = get_data_loader(params.tgt_dataset, params.dataset_root, params.batch_size, train=False)

# load dann model
dann = init_model(net=SVHNmodel(), restore=None)

# train dann model
print("Training dann model")
if not (dann.restored and params.dann_restore):
    dann = train_dann(dann, params, src_data_loader, tgt_data_loader, tgt_data_loader_eval, device, loggi)
