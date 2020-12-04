import os
import random

import torch
import torch.backends.cudnn as cudnn

from datasets import get_mnist, get_mnistm, get_svhn
from datasets.office import get_office
from datasets.officecaltech import get_officecaltech
from datasets.syndigits import get_syndigits
from datasets.synsigns import get_synsigns
from datasets.gtsrb import get_gtsrb

def make_cuda(tensor):
    """Use CUDA if it's available."""
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor


def denormalize(x, std, mean):
    """Invert normalization, and then convert array into image."""
    out = x * std + mean
    return out.clamp(0, 1)


def init_weights(layer):
    """Init weights for layers w.r.t. the original paper."""
    layer_name = layer.__class__.__name__
    if layer_name.find("Conv") != -1:
        layer.weight.data.normal_(0.0, 0.02)
        layer.bias.data.fill(0.0)
    elif layer_name.find("BatchNorm") != -1:
        layer.weight.data.normal_(1.0, 0.02)
        layer.bias.data.fill_(0)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        size = m.weight.size()
        m.weight.data.normal_(0.0, 0.1)
        m.bias.data.fill_(0)


def init_random_seed(manual_seed):
    """Init random seed."""
    seed = None
    if manual_seed is None:
        seed = random.randint(1, 10000)
    else:
        seed = manual_seed
    print("use random seed: {}".format(seed))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_data_loader(name, dataset_root, batch_size, train=True):
    """Get data loader by name."""
    if name == "mnist":
        return get_mnist(dataset_root, batch_size, train)
    elif name == "mnistm":
        return get_mnistm(dataset_root, batch_size, train)
    elif name == "svhn":
        return get_svhn(dataset_root, batch_size, train)
    elif name == "amazon31":
        return get_office(dataset_root, batch_size, 'amazon')
    elif name == "webcam31":
        return get_office(dataset_root, batch_size, 'webcam')
    elif name == "dslr31":
        return get_office(dataset_root, batch_size, 'dslr')
    elif name == "webcam10":
        return get_officecaltech(dataset_root, batch_size, 'webcam')
    elif name == "syndigits":
        return get_syndigits(dataset_root, batch_size, train)
    elif name == "synsigns":
        return get_synsigns(dataset_root, batch_size, train)
    elif name == "gtsrb":
        return get_gtsrb(dataset_root, batch_size, train)


def init_model(net, restore):
    """Init models with cuda and weights."""
    # init weights of model
    # net.apply(init_weights)

    # restore model weights
    if restore is not None and os.path.exists(restore):
        net.load_state_dict(torch.load(restore))
        net.restored = True
        print("Restore model from: {}".format(os.path.abspath(restore)))
    else:
        print("No trained model, train from scratch.")

    # check if cuda is available
    if torch.cuda.is_available():
        cudnn.benchmark = True
        net.cuda()

    return net


def save_model(net, model_root, filename):
    """Save trained model."""
    if not os.path.exists(model_root):
        os.makedirs(model_root)
    torch.save(net.state_dict(), os.path.join(model_root, filename))
    print("save pretrained model to: {}".format(os.path.join(model_root, filename)))