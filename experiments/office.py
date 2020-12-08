import os
import sys
import torch
sys.path.append(os.path.abspath('.'))
from core.train import train_dann
from core.test import test
from models.model import AlexModel
from models.model import ResNet50
from utils.utils import get_data_loader, init_model, init_random_seed
from utils.altutils import setLogger

# To avoid proxy issues while downloading pretrained model
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


class Config(object):
    # params for path
    currentDir = os.path.dirname(os.path.realpath(__file__))
    dataset_root = os.environ["DATASETDIR"]
    model_root = os.path.join(currentDir, 'checkpoints')

    finetune_flag = True
    lr_adjust_flag = 'non-simple'
    src_only_flag = False

    # params for datasets and data loader
    batch_size = 32

    # params for source dataset
    # src_dataset = "amazon31"
    # src_dataset = "dslr31"
    src_dataset = "webcam31"
    src_model_trained = True
    src_classifier_restore = os.path.join(model_root, src_dataset + '-source-classifier-final.pt')

    # params for target dataset
    # tgt_dataset = "webcam31"
    # tgt_dataset = "dslr31"
    tgt_dataset = "amazon31"
    tgt_model_trained = True
    dann_restore = os.path.join(model_root, src_dataset + '-' + tgt_dataset + '-dann-final.pt')

    # params for pretrain
    num_epochs_src = 100
    log_step_src = 5
    save_step_src = 50
    eval_step_src = 10

    # params for training dann
    gpu_id = '0'

    ## for office
    num_epochs = 1000
    log_step = 10  # iters
    save_step = 500
    eval_step = 10  # epochs

    manual_seed = 8888
    alpha = 0

    # params for optimizing models
    lr = 2e-4


params = Config()

currentDir = os.path.dirname(os.path.realpath(__file__))
logFile = os.path.join(currentDir+'/../', 'dann-{}-{}.log'.format(params.src_dataset, params.tgt_dataset))
loggi = setLogger(logFile)

# init random seed
init_random_seed(params.manual_seed)

# init device
device = torch.device("cuda:" + params.gpu_id if torch.cuda.is_available() else "cpu")

# load dataset
src_data_loader = get_data_loader(params.src_dataset, params.dataset_root, params.batch_size)
tgt_data_loader = get_data_loader(params.tgt_dataset, params.dataset_root, params.batch_size)

# load dann model
# dann = init_model(net=AlexModel(), restore=None)
dann = init_model(net=ResNet50(), restore=None)

# train dann model
print("Start training dann model.")

# if not (dann.restored and params.dann_restore):
dann = train_dann(dann, params, src_data_loader, tgt_data_loader, tgt_data_loader, device, loggi)

print('done')