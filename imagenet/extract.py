from utillc import *
import cv2
import os
import re
import json
import numpy as np
import scipy.io
import pandas as pd
from collections import OrderedDict
from tqdm import tqdm
import torch
 
from torchnet import dataset, meter
from torchnet.engine import Engine
 
from torch.autograd import Variable
from torch.nn import Parameter
from torch.backends import cudnn
import torch.nn.functional as F
from utils import get_iterator
import torch.nn as nn
from os import listdir
from os.path import isfile, join

#from scattering import Scattering
from scatwave.scattering import Scattering
 
import models
import argparse
import main_test

import progressbar

import torch
from torch.autograd import Variable


class ScatModule(torch.nn.Module):
    def __init__(self, opt):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        self.scat = Scattering(M=opt.N, N=opt.N, J=opt.scat, pre_pad=False).cuda()        
        super(ScatModule, self).__init__()

    def forward(self, x):
        xx = x
        EKOX(x.size())
        #y_pred = self.scat(xx.data)
        y_pred = xx.data
        x.data = y_pred
        return x


    
parser =  main_test.parse()

# Model options
opt = parser.parse_args()
import torchvision.datasets
EKOX(torchvision.__file__)

scatModel = nn.DataParallel(ScatModule(opt))

meta = scipy.io.loadmat(os.path.join(opt.imagenetpath, 'ILSVRC2012_devkit_t12', 'data', 'meta.mat'))
gt_file = os.path.join(opt.imagenetpath, 'ILSVRC2012_devkit_t12', 'data','ILSVRC2012_validation_ground_truth.txt')
ll = [ (x[1][0], x[3][0]) for x in meta['synsets'][:,0] ]
din = dict(ll)
with open(gt_file) as f:
    gt_labels = map(int, f.readlines())

scat = Scattering(M=opt.N, N=opt.N, J=opt.scat, pre_pad=False).cuda()
folder = None
def cb(_folder) :
    global folder
    EKOX(_folder)
    ds =_folder #torchvision.datasets.ImageFolder(os.path.join(opt.imagenetpath, 'val'))
    #EKOX(ds.classes)
    #EKOX(len(ds))
    
    nono = 522
    vf = ds.classes[ds[nono][1]]
    EKOX(din[vf])
    EKOX(TYPE(ds[nono][0].cpu().numpy()))
    #IMG(ds[nono][0])

    folder = _folder

iterator = get_iterator(True, opt, cb, pin_memory=False)

classes = folder.classes

EKOX(len(iterator))
EKOX(len(iterator))

bar = progressbar.ProgressBar()

for i, v in bar(enumerate(iterator)) :
    #EKOX(TYPE(v[0].numpy()))
    #EKOX(TYPE(v[1].numpy()))
    img = v[0].numpy()[0]
    c = classes[v[1][0]]

    if False :
        input = Variable(v[0].cuda(async=False))
        desc = scatModel(input)
        data = desc.data.cpu().numpy()

    desc = scat(v[0].cuda())
    data = desc.cpu().numpy()
    fn = os.path.join(opt.imagenetpath, 'train_descs', c)
    fn = '/data01/IN/descs'
    np.savez(os.path.join(fn, 'f' + str(i).zfill(4) + 'npy'), img=data, cat=v[1].numpy())
    #np.save(os.path.join(opt.imagenetpath, 'train_descs', c, 'f' + str(i).zfill(4) + 'npy'), data)


EKO()

def process_file(ff) :
    path = os.path.join(opt.imagenetpath, 'train', c, ff)
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    desc = scat(img)

def process_dir(c) :
    try :
        os.mkdir(os.path.join(opt.imagenetpath, 'train_descs', c))
    except :
        pass
    ims = [f for f in listdir(os.path.join(opt.imagenetpath, 'train', c) )]
    map(process_file, ims)


#cat = [f for f in listdir(os.path.join(opt.imagenetpath, 'train') )]
#map(process_dir, cat)

    


