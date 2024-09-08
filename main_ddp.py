#!/usr/bin/env python
from __future__ import print_function
import inspect
import os
import pickle
import shutil
import sys
import time
from collections import OrderedDict
from copy import deepcopy
import random

import numpy as np
import glob
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.optim as optim
import yaml
from tensorboardX import SummaryWriter
from tqdm import tqdm
import wandb
from mixup import MixUp
from utils import import_class, get_parser, init_seed, has_batchnorms, list_all_reduce, get_world_size


def ema_update(source, target, decay=0.9999, start_itr=20, itr=None):
    # If an iteration counter is provided and itr is less than the start itr,
    # peg the ema weights to the underlying weights.
    if itr and itr<start_itr:
        decay = 0.0
    # source = copy.deepcopy(source)
    with torch.no_grad():
        for key, value in source.module.state_dict().items():
            target.state_dict()[key].copy_(target.state_dict()[key] * decay + value * (1 - decay))


class Processor():
    """
        Processor for Skeleton-based Action Recgnition
    """

    def __init__(self, arg):
        self.arg = arg
        self.save_arg()
        # self.setup_writer()  # deprecated
        self.load_model()
        self.load_optimizer()
        self.create_lrscheduler()
        self.setup_dataloader()
        self.lr = self.arg.base_lr
        self.best_acc = 0
        self.best_acc_epoch = 0
        self.global_step = 0
        self.num_classes = self.model.num_class
        self.arg.model_saved_name = os.path.join(self.arg.work_dir, 'runs')
        if os.path.isdir(self.arg.model_saved_name):
            print('log_dir: ', self.arg.model_saved_name, 'already exist')

        if arg.sync_bn and has_batchnorms(self.model):
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

        self.model = self.model.cuda(self.arg.local_rank)
        self.model = DDP(self.model, device_ids=[self.arg.local_rank])

        if self.arg.ema:
            self.best_acc_ema = 0
            self.best_acc_epoch_ema = 0
            Model = import_class(self.arg.model)
            self.model_ema = Model(**self.arg.model_args).cuda(self.arg.local_rank)
            self.model_ema.eval()
            for p in self.model_ema.parameters():
                p.requires_grad = False
            ema_update(self.model, self.model_ema, itr=0)

    def is_main_process(self):
        return self.arg.local_rank == 0

    def setup_writer(self):
        if not self.is_main_process():
            return

        if self.arg.phase == 'train':
            if not self.arg.train_feeder_args['debug']:
                self.train_writer = SummaryWriter(os.path.join(self.arg.model_saved_name, 'train'), 'train')
                self.val_writer = SummaryWriter(os.path.join(self.arg.model_saved_name, 'val'), 'val')
            else:
                self.train_writer = self.val_writer = SummaryWriter(os.path.join(self.arg.model_saved_name, 'test'), 'test')

    def setup_dataloader(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()
        if self.arg.phase == 'train':
            train_dataset = Feeder(**self.arg.train_feeder_args)
            self.train_sampler = DistributedSampler(train_dataset, shuffle=True)
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=train_dataset,
                batch_size=self.arg.batch_size,
                pin_memory=True,
                num_workers=self.arg.num_worker,
                drop_last=True,
                sampler=self.train_sampler,
                persistent_workers=True,
            )
        test_dataset = Feeder(**self.arg.test_feeder_args)
        self.test_sampler = DistributedSampler(test_dataset, shuffle=False)
        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=self.arg.test_batch_size,
            pin_memory=True,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            sampler=self.test_sampler,
            persistent_workers=True,
        )

    def load_model(self):
        output_device = self.arg.local_rank
        Model = import_class(self.arg.model)
        shutil.copy2(inspect.getfile(Model), self.arg.work_dir)
        print(Model)
        self.model = Model(**self.arg.model_args)
        print(self.model)

        # If torch version is 2. then use torch compile
        # if int(torch.__version__[0]) >= 2:
        #     self.model = torch.compile(self.model, mode='default')

        self.loss = nn.CrossEntropyLoss().cuda(output_device)

        if self.arg.weights:
            self.global_step = int(self.arg.weights[:-3].split('-')[-1])
            self.print_log('Load weights from {}.'.format(self.arg.weights))
            if '.pkl' in self.arg.weights:
                with open(self.arg.weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.weights)

            weights = OrderedDict([[k.split('module.')[-1], v.cuda(output_device)] for k, v in weights.items()])

            keys = list(weights.keys())
            for w in self.arg.ignore_weights:
                for key in keys:
                    if w in key:
                        if weights.pop(key, None) is not None:
                            self.print_log('Sucessfully Remove Weights: {}.'.format(key))
                        else:
                            self.print_log('Can Not Remove Weights: {}.'.format(key))

            try:
                self.model.load_state_dict(weights)
            except:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print('Can not find these weights:')
                for d in diff:
                    print('  ' + d)
                state.update(weights)
                self.model.load_state_dict(state)

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=self.arg.momentum,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)

        elif self.arg.optimizer == 'NAdam':
            self.optimizer = optim.NAdam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)

        elif self.arg.optimizer == "AdamW":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)

        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

        self.print_log('using warm up, epoch: {}'.format(self.arg.warm_up_epoch))

    def save_arg(self):
        # save arg
        if not self.is_main_process():
            return
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            f.write(f"# command line: {' '.join(sys.argv)}\n\n")
            yaml.dump(arg_dict, f)

    def create_lrscheduler(self):
        scheduler_name = self.arg.lrscheduler
        self.scheduler = None

        if scheduler_name == 'cosine':
            from lr_scheduler import WarmUpCosineAnnealingLR
            self.scheduler = WarmUpCosineAnnealingLR(
                self.optimizer,
                T_max=self.arg.num_epoch,
                warmup_epochs=self.arg.warm_up_epoch,
                eta_min=1e-6
            )

    def adjust_learning_rate(self, epoch):
        if self.arg.optimizer == 'SGD' or self.arg.optimizer == 'Adam' or self.arg.optimizer == 'NAdam':
            if epoch < self.arg.warm_up_epoch:
                lr = self.arg.base_lr * (epoch + 1) / self.arg.warm_up_epoch
            else:
                lr = self.arg.base_lr * (
                        self.arg.lr_decay_rate ** np.sum(epoch >= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        else:
            raise ValueError()

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, str, print_time=True):
        if not self.is_main_process():
            return

        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(str, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def train(self, epoch, save_model=False):
        self.model.train()
        self.print_log('Training epoch: {}'.format(epoch + 1))
        self.train_sampler.set_epoch(epoch)
        loader = self.data_loader['train']

        if self.arg.lrscheduler == "step":
            assert self.scheduler is None
            self.adjust_learning_rate(epoch)

        loss_value = torch.tensor(0.).to(self.arg.local_rank)
        acc_value = torch.tensor(0.).to(self.arg.local_rank)
        count = torch.tensor(0.).to(self.arg.local_rank)

        if hasattr(self, 'train_writer'):
            self.train_writer.add_scalar('epoch', epoch, self.global_step)
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)
        process = tqdm(loader, ncols=40, disable=not self.is_main_process())

        use_amp = True # True
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None

        # Mixup
        if self.arg.use_mixup:
            mixup_fn = MixUp(num_classes=self.num_classes, mix_prob=self.arg.mixup_prob)

        for batch_idx, (data, label, index) in enumerate(process):
            self.global_step += 1
            data = data.float().cuda(self.arg.local_rank, non_blocking=True)
            label = label.long().cuda(self.arg.local_rank, non_blocking=True)
            timer['dataloader'] += self.split_time()

            if self.arg.use_mixup:
                data, label = mixup_fn(data, label)

            with torch.cuda.amp.autocast(enabled=use_amp):
                output = self.model(data)
                loss = self.loss(output, label)

            # Perform backward and optimizer steps
            # If using mixed precision, scale loss and call step on the scaler
            if use_amp:
                self.optimizer.zero_grad()
                scaler.scale(loss).backward()

                if self.arg.max_norm > 0:
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.arg.max_norm)

                scaler.step(self.optimizer)
                scaler.update()
            # If not using mixed precision, perform the usual backward and optimizer steps
            else:
                self.optimizer.zero_grad()
                loss.backward()

                if self.arg.max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.arg.max_norm)

                self.optimizer.step()

            bs = label.size(0)
            loss_value += bs * loss
            count += bs
            timer['model'] += self.split_time()

            if self.arg.use_mixup:
                _top_max_k_vals, top_max_k_inds = torch.topk(
                    label, 2, dim=1, largest=True, sorted=True
                )
                idx_top1 = torch.arange(label.shape[0]), top_max_k_inds[:, 0]
                idx_top2 = torch.arange(label.shape[0]), top_max_k_inds[:, 1]
                output[idx_top1] += output[idx_top2]
                output[idx_top2] = 0.0
                label = top_max_k_inds[:, 0]

            value, predict_label = torch.max(output.data, 1)
            acc = torch.mean((predict_label == label.data).float())
            acc_value += bs * acc
            self.lr = self.optimizer.param_groups[0]['lr']
            timer['statistics'] += self.split_time()

            if hasattr(self, 'train_writer'):
                self.train_writer.add_scalar('acc', acc, self.global_step)
                self.train_writer.add_scalar('loss', loss.data.item(), self.global_step)
                self.train_writer.add_scalar('lr', self.lr, self.global_step)

        if self.scheduler is not None:
            self.scheduler.step()

        # statistics of time consumption and loss
        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }

        # Now perform the all_reduce operation to sum these values across all GPUs
        dist.all_reduce(acc_value, op=dist.ReduceOp.SUM)
        dist.all_reduce(loss_value, op=dist.ReduceOp.SUM)
        dist.all_reduce(count, op=dist.ReduceOp.SUM)

        global_loss = loss_value.item() / count.item()
        global_acc = acc_value.item() / count.item()

        self.print_log(
            '\tMean training loss: {:.4f}. Mean training acc: {:.2f}%.'.format(global_loss, global_acc*100))
        self.print_log('\tTime consumption: [Data]{dataloader}, [Network]{model}'.format(**proportion))

        if save_model and self.is_main_process():
            state_dict = self.model.state_dict()
            weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])
            torch.save(weights, self.arg.model_saved_name + '-' + str(epoch+1) + '-' + str(int(self.global_step)) + '.pt')

        self.log_dict.update({
            'lr': self.lr,
            'train/loss': global_loss,
            'train/acc': global_acc,
        })

    def eval(self, epoch, save_score=False, loader_name=['test'], wrong_file=None, result_file=None):
        if save_score:
            assert get_world_size() == 1, "Save score function currently only supports single gpu."

        if wrong_file is not None and self.is_main_process():
            f_w = open(wrong_file, 'w')
        if result_file is not None and self.is_main_process():
            f_r = open(result_file, 'w')
        self.model.eval()
        self.print_log('Eval epoch: {}'.format(epoch + 1))
        for ln in loader_name:
            score_frag = []
            loss_value = torch.tensor(0.).to(self.arg.local_rank)
            acc_value = torch.tensor(0.).to(self.arg.local_rank)
            count = torch.tensor(0.).to(self.arg.local_rank)

            if self.arg.ema:
                score_frag_ema = []
                loss_value_ema = torch.tensor(0.).to(self.arg.local_rank)
                acc_value_ema = torch.tensor(0.).to(self.arg.local_rank)

            process = tqdm(self.data_loader[ln], ncols=40, disable=not self.is_main_process())

            for batch_idx, (data, label, index) in enumerate(process):
                with torch.no_grad():
                    data = data.float().cuda(self.arg.local_rank, non_blocking=True)
                    label = label.long().cuda(self.arg.local_rank, non_blocking=True)
                    output = self.model(data)

                    loss = self.loss(output, label)

                    value, predict_label = torch.max(output.data, 1)
                    acc = torch.mean((predict_label == label.data).float())

                    bs = label.size(0)
                    acc_value += bs * acc
                    loss_value += bs * loss
                    count += bs

                    if save_score:
                        score_frag.append(output.data.cpu().numpy())

                    if self.arg.ema:
                        output_ema = self.model_ema(data)
                        loss_ema = self.loss(output_ema, label)

                        value_ema, predict_label_ema = torch.max(output_ema.data, 1)
                        acc_ema = torch.mean((predict_label_ema == label.data).float())

                        acc_value_ema += bs * acc_ema
                        loss_value_ema += bs * loss_ema

                        if save_score:
                            score_frag_ema.append(output_ema.data.cpu().numpy())

                if (wrong_file is not None or result_file is not None) and self.is_main_process():
                    predict = list(predict_label.cpu().numpy())
                    true = list(label.data.cpu().numpy())
                    for i, x in enumerate(predict):
                        if result_file is not None:
                            f_r.write(str(x) + ',' + str(true[i]) + '\n')
                        if x != true[i] and wrong_file is not None:
                            f_w.write(str(index[i]) + ',' + str(x) + ',' + str(true[i]) + '\n')

            # Now perform the all_reduce operation to sum these values across all GPUs
            dist.all_reduce(acc_value, op=dist.ReduceOp.SUM)
            dist.all_reduce(loss_value, op=dist.ReduceOp.SUM)
            dist.all_reduce(count, op=dist.ReduceOp.SUM)

            global_loss = loss_value.item() / count.item()
            global_acc = acc_value.item() / count.item()

            if global_acc > self.best_acc:
                self.best_acc = global_acc
                self.best_acc_epoch = epoch + 1

            if self.arg.ema:
                dist.all_reduce(acc_value_ema, op=dist.ReduceOp.SUM)
                dist.all_reduce(loss_value_ema, op=dist.ReduceOp.SUM)

                global_loss_ema = loss_value_ema.item() / count.item()
                global_acc_ema = acc_value_ema.item() / count.item()

                if global_acc_ema > self.best_acc_ema:
                    self.best_acc_ema = global_acc_ema
                    self.best_acc_epoch_ema = epoch + 1

            # print('Accuracy: ', global_acc, ' model: ', self.arg.model_saved_name)
            if self.arg.phase == 'train':
                if hasattr(self, 'val_writer'):
                    self.val_writer.add_scalar('loss', global_loss, self.global_step)
                    self.val_writer.add_scalar('acc', global_acc, self.global_step)
                    if self.arg.ema:
                        self.val_writer.add_scalar('loss_ema', global_loss_ema, self.global_step)
                        self.val_writer.add_scalar('acc_ema', global_acc_ema, self.global_step)

                self.log_dict.update({
                    'val/loss': global_loss,
                    'val/acc': global_acc,
                })

                if self.arg.ema:
                    self.log_dict.update({
                        'val/loss_ema': global_loss_ema,
                        'val/acc_ema': global_acc_ema,
                    })

            self.print_log('\tMean {} loss of {} batches: {}.'.format(
                ln, len(self.data_loader[ln]), global_loss))
            self.print_log('\tTop1: {:.2f}%'.format(global_acc * 100))

            if self.arg.ema:
                self.print_log('\tMean {} loss_ema of {} batches: {}.'.format(
                    ln, len(self.data_loader[ln]), global_loss_ema))
                self.print_log('\tTop1_ema: {:.2f}%'.format(global_acc_ema * 100))

            if save_score:
                score = np.concatenate(score_frag)

                if 'ucla' in self.arg.feeder:
                    self.data_loader[ln].dataset.sample_name = np.arange(len(score))
                score_dict = dict(
                    zip(self.data_loader[ln].dataset.sample_name, score))
                with open('{}/epoch{}_{}_score.pkl'.format(
                        self.arg.work_dir, epoch + 1, ln), 'wb') as f:
                    pickle.dump(score_dict, f)

                if self.arg.ema:
                    score_ema = np.concatenate(score_frag_ema)
                    score_dict_ema = dict(zip(self.data_loader[ln].dataset.sample_name, score_ema))
                    with open('{}/epoch{}_{}_score_ema.pkl'.format(
                            self.arg.work_dir, epoch + 1, ln), 'wb') as f:
                        pickle.dump(score_dict_ema, f)

    def start(self):
        if self.arg.phase == 'train':
            self.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            self.global_step = self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size
            def count_parameters(model):
                return sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.print_log(f'# Parameters: {count_parameters(self.model)}')
            self.log_dict = {}
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                save_model = (((epoch + 1) % self.arg.save_interval == 0) or (
                        epoch + 1 == self.arg.num_epoch)) and (epoch+1) > self.arg.save_epoch

                self.train(epoch, save_model=save_model)
                if self.arg.ema:
                    ema_update(self.model, self.model_ema, itr=epoch)

                if epoch >= self.arg.start_eval:
                    self.eval(epoch, save_score=self.arg.save_score, loader_name=['test'])

                if self.is_main_process():
                    if epoch == 0:
                        wandb.init(project=f"Hybridformer{self.num_classes}", name=self.arg.work_dir)
                    wandb.log(self.log_dict)

            # test the best model
            weights_path = glob.glob(os.path.join(self.arg.work_dir, 'runs-'+str(self.best_acc_epoch)+'*'))[0]
            weights = torch.load(weights_path)
            if isinstance(self.model, (nn.parallel.DistributedDataParallel, nn.DataParallel)):
                weights = OrderedDict([['module.'+k, v.cuda(self.arg.local_rank)] for k, v in weights.items()])
            self.model.load_state_dict(weights)

            wf = weights_path.replace('.pt', '_wrong.txt')
            rf = weights_path.replace('.pt', '_right.txt')
            self.arg.print_log = False
            self.eval(epoch=0, save_score=True, loader_name=['test'], wrong_file=wf, result_file=rf)
            self.arg.print_log = True

            num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            self.print_log(f'Best accuracy: {self.best_acc}')
            self.print_log(f'Epoch number: {self.best_acc_epoch}')
            self.print_log(f'Model name: {self.arg.work_dir}')
            self.print_log(f'Model total number of params: {num_params}')
            self.print_log(f'Weight decay: {self.arg.weight_decay}')
            self.print_log(f'Base LR: {self.arg.base_lr}')
            self.print_log(f'Batch Size: {self.arg.batch_size}')
            self.print_log(f'Test Batch Size: {self.arg.test_batch_size}')
            self.print_log(f'seed: {self.arg.seed}')

        elif self.arg.phase == 'test':
            wf = self.arg.weights.replace('.pt', '_wrong.txt')
            rf = self.arg.weights.replace('.pt', '_right.txt')

            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.arg.print_log = False
            self.print_log('Model:   {}.'.format(self.arg.model))
            self.print_log('Weights: {}.'.format(self.arg.weights))
            self.eval(epoch=0, save_score=self.arg.save_score, loader_name=['test'], wrong_file=wf, result_file=rf)
            self.print_log('Done.\n')


def init_distributed_mode(rank, world_size, master_port="12355"):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = master_port
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    """
    Clean up distributed training environment.
    """
    dist.destroy_process_group()


def worker(rank, world_size, master_port, args):
    """
    The main worker function to be called for each process.
    """
    print(f"Running DDP on rank {rank}.")
    init_distributed_mode(rank, world_size, master_port)

    try:
        args_new = deepcopy(args)
        init_seed(args_new.seed)
        args_new.local_rank = rank
        processor = Processor(args_new)
        processor.start()
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cleanup()


def main():
    parser = get_parser()

    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.safe_load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)

    args = parser.parse_args()
    world_size = torch.cuda.device_count()
    # master_port = args.port
    master_port = str(random.randint(12000, 31999))

    try:
        mp.spawn(worker, args=(world_size, master_port, args), nprocs=world_size)
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("Main process shutting down.")


if __name__ == '__main__':
    main()
