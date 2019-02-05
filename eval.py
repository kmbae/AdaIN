"""
    2018 Spring EE898
    Advanced Topics in Deep Learning
    for Robotics and Computer Vision

    Programming Assignment 2
    Neural Style Transfer

    Author : Jinsun Park (zzangjinsun@gmail.com)

    References
    [1] Gatys et al., "Image Style Transfer using Convolutional
        Neural Networks", CVPR 2016.
    [2] Huang and Belongie, "Arbitrary Style Transfer in Real-Time
        with Adaptive Instance Normalization", ICCV 2017.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import gc
import visdom
import os
import time
import numpy as np
from os import listdir
from PIL import Image
from datetime import datetime
import ipdb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torchvision import utils, transforms, models
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from train import *


# Some utilities



"""
    Task 2. Complete training code.

    Following skeleton code assumes that you have multiple GPUs
    You can freely change any of parameters
"""
def test():
    gc.disable()

    # Parameters
    path_snapshot = 'snapshots'
    path_content = 'dataset/test/content'
    path_style = 'dataset/test/style'

    if not os.path.exists(path_snapshot):
        os.makedirs(path_snapshot)

    batch_size = 1
    weight_decay = 1.0e-5
    num_epoch = 600
    lr_init = 0.0001#0.001
    lr_decay_step = num_epoch/2
    momentum = 0.9
    #device_ids = [0, 1, 2]
    w_style = 10
    alpha = 1
    disp_step = 1

    # Data loader
    dm = DataManager(path_content, path_style, random_crop=False)
    dl = DataLoader(dm, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)

    num_train = dm.num
    num_batch = np.ceil(num_train / batch_size)
    loss_train_avg = np.zeros(num_epoch)

    net = StyleTransferNet(w_style, alpha)
    net = nn.DataParallel(net.cuda(), device_ids=range(torch.cuda.device_count()))

    # Load model
    state_dict = torch.load('snapshots/epoch_000501.pth')
    net.load_state_dict(state_dict)

    # Start training
    net.eval()
    running_loss_train = 0

    for i, data in enumerate(dl, 0):
        img_con = data['content']
        img_sty = data['style']

        img_con = Variable(img_con, requires_grad=False).cuda()
        img_sty = Variable(img_sty, requires_grad=False).cuda()

        img_result = net(img_con, img_sty)
        img_result.insert(0, img_con)
        img_result.append(img_sty)
        img_cat = torch.cat(img_result, dim=3)
        img_cat = torch.unbind(img_cat, dim=0)
        img_cat = torch.cat(img_cat, dim=1)
        img_cat = dm.restore(img_cat.data.cpu())
        output_img = torch.clamp(img_cat, 0, 1)

        tt=transforms.ToPILImage()(output_img)
        tt.save('test_out/{}.png'.format(i))

        if (i+1)%disp_step==0:
            print('Testing {}/{} images'.format(i,len(dl)))


    gc_collected = gc.collect()
    gc.disable()

    print('Testing finished.')



if __name__ == '__main__':
    test()
