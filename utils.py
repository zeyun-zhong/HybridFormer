import torch
import sys
import numpy as np
import random
import traceback
import argparse
from torchlight import DictAction
import torch.nn as nn
import torch.distributed as dist
import pickle


def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = True
    # training speed is too slow if set to True
    torch.backends.cudnn.deterministic = False

    # on cuda 11 cudnn8, the default algorithm is very slow
    # unlike on cuda 10, the default works well
    torch.backends.cudnn.benchmark = True


def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def has_batchnorms(model):
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    for name, module in model.named_modules():
        if isinstance(module, bn_types):
            return True
    return False


def list_all_reduce(list_to_reduce):
    list_to_reduce_sum = sum(list_to_reduce)
    local_count = len(list_to_reduce)

    # Convert to tensors for reduction
    list_to_reduce_sum_tensor = torch.tensor(list_to_reduce_sum).cuda()
    local_count_tensor = torch.tensor(local_count).cuda()

    # Use all_reduce to sum these tensors across all GPUs
    dist.all_reduce(list_to_reduce_sum_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(local_count_tensor, op=dist.ReduceOp.SUM)

    # Calculate the global mean loss
    global_mean = list_to_reduce_sum_tensor.item() / local_count_tensor.item()
    return global_mean


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(
        description='GRA Transformer')

    parser.add_argument(
        '--work-dir',
        default='./work_dir/temp',
        help='the work folder for storing results')

    parser.add_argument('-model_saved_name', default='')
    parser.add_argument(
        '--config',
        default='./config/nturgbd-cross-view/test_bone.yaml',
        help='path to the configuration file')

    # processor
    parser.add_argument(
        '--phase', default='train', help='must be train or test')
    parser.add_argument(
        '--save-score',
        type=str2bool,
        default=False,
        help='if ture, the classification score will be stored')

    # gra
    parser.add_argument(
        '--joint-label',
        type=list,
        default=[],
        help='tells which group each joint belongs to')

    # visulize and debug
    parser.add_argument(
        '--seed', type=int, default=42, help='random seed for pytorch')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=100,
        help='the interval for printing messages (#iteration)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=1,
        help='the interval for storing models (#iteration)')
    parser.add_argument(
        '--save-epoch',
        type=int,
        default=10,
        help='the start epoch to save model (#iteration)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=5,
        help='the interval for evaluating models (#iteration)')
    parser.add_argument(
        '--ema',
        action="store_true",
        default=False,
        help='ema weight for eval')

    parser.add_argument('--lambda_1', type=float, default=1e-4)
    parser.add_argument('--lambda_2', type=float, default=1e-1)

    parser.add_argument(
        '--print-log',
        type=str2bool,
        default=True,
        help='print logging or not')
    parser.add_argument(
        '--show-topk',
        type=int,
        default=[1, 5],
        nargs='+',
        help='which Top K accuracy will be shown')

    # feeder
    parser.add_argument(
        '--feeder', default='feeder.feeder', help='data loader will be used')
    parser.add_argument(
        '--num-worker',
        type=int,
        default=4,
        help='the number of worker for data loader')
    parser.add_argument(
        '--train-feeder-args',
        action=DictAction,
        default=dict(),
        help='the arguments of data loader for training')
    parser.add_argument(
        '--test-feeder-args',
        action=DictAction,
        default=dict(),
        help='the arguments of data loader for test')

    # model
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument(
        '--model-args',
        action=DictAction,
        default=dict(),
        help='the arguments of model')
    parser.add_argument(
        '--weights',
        default=None,
        help='the weights for network initialization')
    parser.add_argument(
        '--ignore-weights',
        type=str,
        default=[],
        nargs='+',
        help='the name of weights which will be ignored in the initialization')

    # optim
    parser.add_argument(
        '--base-lr', type=float, default=0.025, help='initial learning rate')
    parser.add_argument(
        '--lrscheduler', default='step', choices=['step', 'cosine'])
    parser.add_argument(
        '--step',
        type=int,
        default=[110, 120],
        nargs='+',
        help='the epoch where optimizer reduce the learning rate')
    parser.add_argument(
        '--device',
        type=int,
        default=[0,1],
        nargs='+',
        help='the indexes of GPUs for training or testing')
    parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
    parser.add_argument(
        '--nesterov', type=str2bool, default=False, help='use nesterov or not')
    parser.add_argument(
        '--momentum', type=float, default=0.9, help='nesterov momentum')
    parser.add_argument(
        '--batch-size', type=int, default=256, help='training batch size')
    parser.add_argument(
        '--test-batch-size', type=int, default=256, help='test batch size')
    parser.add_argument(
        '--start-epoch',
        type=int,
        default=0,
        help='start training from which epoch')
    parser.add_argument(
        '--num-epoch',
        type=int,
        default=80,
        help='stop training in which epoch')
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.0005,
        help='weight decay for optimizer')
    parser.add_argument(
        '--lr-decay-rate',
        type=float,
        default=0.1,
        help='decay rate for learning rate')
    parser.add_argument('--warm_up_epoch', type=int, default=0)
    parser.add_argument('--max_norm', type=float, default=0.)
    parser.add_argument('--use_mixup', type=str2bool, default=False)
    parser.add_argument('--mixup_prob', type=float, default=0.)
    parser.add_argument('--start_eval', type=int, default=0)

    # DDP related
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--port', default="12355")
    parser.add_argument('--sync_bn', type=str2bool, default=False)

    return parser