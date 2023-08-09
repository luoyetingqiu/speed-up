import numpy as np
import io
import os
import time
from collections import defaultdict, deque
import datetime

import torch
import torch.distributed as dist


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def global_avg(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {:.4f}".format(name, meter.global_avg)
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def compute_acc(logits, label, reduction='mean'):
    ret = (torch.argmax(logits, dim=1) == label).float()
    if reduction == 'none':
        return ret.detach()
    elif reduction == 'mean':
        return ret.mean().item()


def compute_n_params(model, return_str=True):
    tot = 0
    for p in model.parameters():
        w = 1
        for x in p.shape:
            w *= x
        tot += w
    if return_str:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


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


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


class Time(object):
    def __init__(self, start_time=time.time(), last_time=time.time()):
        self.start_time = start_time
        self.last_time = last_time

    def cost(self):
        time_cost = time.time() - self.last_time
        self.last_time = time.time()
        return time_cost

    def total_cost(self):
        time_cost = time.time() - self.start_time
        self.start_time = time.time()
        return time_cost

    def update_last_time(self):
        self.last_time = time.time()

    def clear(self):
        self.start_time = time.time()
        self.last_time = time.time()


class Screen(object):

    @staticmethod
    def mean(num_list):
        return sum(num_list) / len(num_list)

    @classmethod
    def recall(cls, img2text_screen, text2img_screen, img2text_gt, text2img_gt):
        img2text_recalls = [len(set(img2text_screen[img]) & set(img2text_gt[img])) / len(img2text_gt[img]) for img in
                            img2text_gt]
        text2img_recalls = [1 if img in text2img_screen[text] else 0 for text, img in text2img_gt.items()]
        return cls.mean(img2text_recalls), cls.mean(text2img_recalls)

    @classmethod
    def avg_len(cls, img2text_screen, text2img_screen):
        img2text_lens = [len(texts) for img, texts in img2text_screen.items()]
        text2img_lens = [len(imgs) for text, imgs in text2img_screen.items()]
        return cls.mean(img2text_lens), cls.mean(text2img_lens)

    @staticmethod
    def screen_id2order(img2text_id_screen, text2img_id_screen, img_id2order, text_id2order):
        img2text_screen = {}
        text2img_screen = {}
        all_img = [img_id2order[int(img_id)] for img_id in img2text_id_screen.keys()]
        all_text = [text_id2order[int(text_id)] for text_id in text2img_id_screen.keys()]
        empty_text_ids = [text_id for text_id, img_ids in text2img_id_screen.items() if img_ids == []]
        print('empty text screen size:', len(empty_text_ids))
        for img_id, text_ids in img2text_id_screen.items():
            img2text_screen[img_id2order[int(img_id)]] = screen if (screen := [text_id2order[text_id] for text_id in
                                                                               text_ids]) != [] else all_text
            # img2text_screen[img_id2order[int(img_id)]] = [text_id2order[text_id] for text_id in text_ids]
        for text_id, img_ids in text2img_id_screen.items():
            text2img_screen[text_id2order[int(text_id)]] = screen if (screen := [img_id2order[img_id] for img_id in
                                                                                 img_ids]) != [] else all_img
            # text2img_screen[text_id2order[int(text_id)]] = screen if len(screen := [img_id2order[img_id] for img_id in
            #                                                                         img_ids]) < config['k_test'] else all_img
            # text2img_screen[text_id2order[int(text_id)]] = [img_id2order[img_id] for img_id in img_ids]
        return img2text_screen, text2img_screen


def count_param(model):
    blank = ' '
    print('-' * 90)
    print('|' + ' ' * 11 + 'weight name' + ' ' * 10 + '|' \
          + ' ' * 15 + 'weight shape' + ' ' * 15 + '|' \
          + ' ' * 3 + 'number' + ' ' * 3 + '|')
    print('-' * 90)
    num_para = 0
    type_size = 4  ##如果是浮点数就是4

    exclusion = ['layer.'+str(i) for i in range(0, 6)]

    for index, (key, w_variable) in enumerate(model.named_parameters()):
        if w_variable.requires_grad:
            flag = True
            for ex in exclusion:
                if ex in key:
                    flag = False
            if not flag:
                if len(key) <= 30:
                    key = key + (30 - len(key)) * blank
                shape = str(w_variable.shape)
                if len(shape) <= 40:
                    shape = shape + (40 - len(shape)) * blank
                each_para = 1
                for k in w_variable.shape:
                    each_para *= k
                num_para += each_para
                str_num = str(each_para)
                if len(str_num) <= 10:
                    str_num = str_num + (10 - len(str_num)) * blank

                print('| {} | {} | {} |'.format(key, shape, str_num))
    print('-' * 90)
    print('The total number of parameters: ' + str(num_para))
    print('The parameters of Model {}: {:4f}M'.format(model._get_name(), num_para * type_size / 1000 / 1000))
    print('-' * 90)
