from .resnet import get_resnet_model
from .vgg import *

def get_model(model, method, num_classes):
    """Returns the requested model, ready for training/pruning with the specified method.

    :param model: str, either wrn or r50
    :param method: full or prune
    :param num_classes: int, num classes in the dataset
    :return: A prunable ResNet model
    """

    if model in ['wrn', 'r32', 'r50', 'r101', 'r152']:
        net = get_resnet_model(model, method, num_classes)
    else if model in ['r164']:
        net = get_network_slimming_model(method, num_classes)
    else:
        net = get_vgg_model(model, method, num_classes)
    return net