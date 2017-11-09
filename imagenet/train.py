from utillc import *
import os, sys
import re
import json
import numpy as np
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pandas as pd
from collections import OrderedDict
from tqdm import tqdm
import torch

from os import listdir
from os.path import isfile, join

from sklearn.utils import shuffle
from torchnet import dataset, meter
from torchnet.engine import Engine
 
from torch.autograd import Variable
from torch.nn import Parameter
from torch.backends import cudnn
import torch.nn.functional as F
from utils import get_iterator
 
#from scattering import Scattering
from scatwave.scattering import Scattering
 
import models
import argparse

import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
#import torchvision.models as models

hots = np.eye(1000).astype(long)

bbatch_size  = 2

def parse():
    parser = argparse.ArgumentParser(description='Scattering on Imagenet')
    # Model options
    parser.add_argument('--imagenetpath', default='/media/ssd/dataset/', type=str)
    parser.add_argument('--nthread', default=4, type=int)
    parser.add_argument('--resume', default='scatter_resnet_10_model.pt7', type=str)
 
    parser.add_argument('--ngpu', default=1, type=int,
                        help='number of GPUs to use for training')
    parser.add_argument('--gpu_id', default='0', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--scat', default=3, type=int,
                        help='scattering scale, j=0 means no scattering')
    parser.add_argument('--N', default=224, type=int,
                        help='size of the crop')
    parser.add_argument('--model', default='scatresnet6_2', type=str,
                        help='name of define of the model in models')
    parser.add_argument('--batchSize', default=256, type=int)
    parser.add_argument('--max_samples', default=10, type=int)


    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    
    return parser
 
 
 
 
cudnn.benchmark = True
 
parser = parse()
opt = parser.parse_args()
args = opt
print('parsed options:', vars(opt))
 
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
torch.randn(8).cuda()
os.environ['CUDA_VISIBLE_DEVICES'] = ''
 
data_time = 1
 
class Generator(Dataset) :
    def __init__(self, opt, training) :
        self.i = 0
        self.fn = '/data01/IN/descs'
        l = [f for f in listdir(self.fn) if f[1] < '5']
        N = self.N = len(l)
        EKOX(N)
        print 'N=', self.N
        l = shuffle(l, random_state=1)
        self.l = l[:N/4] if not training else l[N/4:]
        self.fs = iter(l)
        self.ld = {}
        pass
    def __len__(self): return len(self.l)
    def __getitem__(self, idx) :

        ti = idx/1
        pp = self.l[idx]
        inf = os.path.join(self.fn, pp)
        batch = np.load(inf)
        #EKOX((pp, TYPE(batch['img'])))
        X, y = [batch[k] for k in batch.files]
        return (X, y)

                
def main():
    model, params, stats = models.__dict__[opt.model](N=opt.N,J=opt.scat)

    
    gen_train = DataLoader(Generator(opt, True), batch_size=bbatch_size,
                           shuffle=True, num_workers=opt.workers, pin_memory=True)
    val_train = DataLoader(Generator(opt, False), batch_size=bbatch_size,
                           shuffle=True, num_workers=opt.workers, pin_memory=True)

    model.cuda()
    #model = torch.nn.parallel.DistributedDataParallel(model)

    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), opt.lr,
                                momentum=opt.momentum,
                                weight_decay=opt.weight_decay)
    cudnn.benchmark = True
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        # train for one epoch
        train(gen_train, model, criterion, optimizer, epoch)
        prec1 = validate(gen_val, model, criterion)



def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model = torch.nn.DataParallel(model).cuda()

    # switch to train mode
    model.train()

    end = time.time()

    chrono = []

    
    for i, (input, target) in enumerate(train_loader) :
        # measure data loading time
        data_time.update(time.time() - end)
        #EKOX(TYPE(input.numpy()))
        #EKOX(TYPE(target.numpy()))
        shp = input.numpy().shape
        N = shp[1]
        input = input.view(bbatch_size * N, shp[2],  shp[3],  shp[4],  shp[5])
        #target = torch.LongTensor(np.hstack((target[0], target[1])))
        target = target.view(bbatch_size * N) #, torch.Tensor(np.hstack((input[0], input[1])))
        
        #target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input.cuda())
        target_var = torch.autograd.Variable(target.cuda())

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        chrono.append((batch_time.val, data_time.val))
        if len(chrono) > 10 :
            chrono = chrono[-10:]
        EKOX(len(chrono))
        EKOX(np.mean([x[0] for x in chrono]))
        EKOX(np.mean([x[1] for x in chrono]))
        
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1, top5=top5))
        #EKOX(i)
    #EKO()

def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):

        target = torch.LongTensor(target).cuda(async=True)
        input = torch.Tensor(input).cuda(async=True)
        
        input_var = torch.autograd.Variable(input)#, volatile=True)
        target_var = torch.autograd.Variable(target) #, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""

    target = target.cuda()
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


        
if __name__ == '__main__':
    main()
