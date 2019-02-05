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



# Some utilities
class VisdomLine():
    def __init__(self, vis, opts):
        self.vis = vis
        self.opts = opts
        self.win = None

    def Update(self, x, y):
        if self.win is None:
            self.win = self.vis.line(X=x, Y=y, opts=self.opts)
        else:
            self.vis.line(X=x, Y=y, opts=self.opts, win=self.win)

class VisdomImage():
    def __init__(self, vis, opts):
        self.vis = vis
        self.opts = opts
        self.win = None

    def Update(self, image):
        if self.win is None:
            self.win = self.vis.image(image, opts=self.opts)
        else:
            self.vis.image(image, opts=self.opts, win=self.win)

def LearningRateScheduler(optimizer, epoch, lr_decay=0.1, lr_decay_step=10):
    if epoch % lr_decay_step:
        return optimizer

    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr_decay

    print('Learning rate is decreased by %f' % (param_group['lr']))

    return optimizer



# For data loading
class DataManager(Dataset):
    def __init__(self, path_content, path_style, random_crop=True):
        self.path_content = path_content
        self.path_style = path_style

        # Preprocessing for imagenet pre-trained network
        if random_crop:
            self.transform = transforms.Compose(
                [
                    transforms.RandomCrop((256, 256)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.CenterCrop((256,256)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                ]
            )

        # Convert pre-processed images to original images
        self.restore = transforms.Compose(
            [
                transforms.Normalize(mean=[-2.118, -2.036, -1.804],
                                     std=[4.367, 4.464, 4.444]),
            ]
        )

        self.list_content = listdir(self.path_content)
        self.list_style = listdir(self.path_style)

        self.num_content = len(self.list_content)
        self.num_style = len(self.list_style)

        assert self.num_content > 0
        assert self.num_style > 0

        self.num = min(self.num_content, self.num_style)

        print('Content root : %s' % (self.path_content))
        print('Style root : %s' % (self.path_style))
        print('Number of content images : %d' % (self.num_content))
        print('Number of style images : %d' % (self.num_style))
        print('Dataset size : %d' % (self.num))

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        path_to_con = self.path_content + '/' + self.list_content[idx]
        path_to_sty = self.path_style + '/' + self.list_style[idx]

        img_con = Image.open(path_to_con)
        img_con = self.transform(img_con)

        img_sty = Image.open(path_to_sty)
        img_sty = self.transform(img_sty)

        sample = {'content': img_con, 'style': img_sty}

        return sample



"""
    Task 1. Define your neural network here.
"""
decoder = nn.Sequential(
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 256, (3, 3)),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 128, (3, 3)),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 128, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 64, (3, 3)),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 3, (3, 3))
        )

class StyleTransferNet(nn.Module):
    def __init__(self, w_style=0.1, alpha=1):
        super(StyleTransferNet, self).__init__()

        self.w_style = w_style
        self.alpha = alpha
        encoder = list(models.vgg19(pretrained=True).features.children())
        self.enc_1 = nn.Sequential(*encoder[:2])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*encoder[2:7])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*encoder[7:12])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*encoder[12:21])  # relu3_1 -> relu4_1
        self.decoder = decoder
        self.mse_loss = nn.MSELoss()

        for param in self.enc_1.parameters():
            param.requires_grad = False
        for param in self.enc_2.parameters():
            param.requires_grad = False
        for param in self.enc_3.parameters():
            param.requires_grad = False
        for param in self.enc_4.parameters():
            param.requires_grad = False

        """
            Define your neural network in the StyleTransferNet class
            (e.g. encoder and decoder networks)
        """

    def AdaINLayer(self, x, y):
        Bx, Cx, Hx, Wx = x.shape
        By, Cy, Hy, Wy = y.shape

        assert Bx == By
        assert Cx == Cy

        """
            Define your AdaIN layer in here

            output : the result of AdaIN operation
        """
        style_mu, style_sigma = self.calc_mean_std(y)
        content_mu, content_sigma = self.calc_mean_std(x)

        normalized_feat = (x - content_mu.expand([Bx, Cx, Hx, Wx])) / content_sigma.expand([Bx, Cx, Hx, Wx])
        output = normalized_feat * style_sigma.expand([Bx, Cx, Hx, Wx]) + style_mu.expand([Bx, Cx, Hx, Wx])

        return output

    def encode_with_intermediate(self, x):
        results = [x]
        h = self.enc_1(x)
        results.append(h)
        h = self.enc_2(h)
        results.append(h)
        h = self.enc_3(h)
        results.append(h)
        h = self.enc_4(h)
        results.append(h)
        return results[1:]

    def encode(self, x):
        h = self.enc_1(x)
        h = self.enc_2(h)
        h = self.enc_3(h)
        h = self.enc_4(h)
        return h

    def calc_mean_std(self, feat, eps=1e-5):
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()
        assert (len(size) == 4)
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std

    def calc_loss(self, input, target, loss_type):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        assert loss_type=='content' or loss_type=='style'
        if loss_type=='content':
            return self.mse_loss(input, target)
        else:
            input_mean, input_std = self.calc_mean_std(input)
            target_mean, target_std = self.calc_mean_std(target)
            return self.mse_loss(input_mean, target_mean) + \
                     self.mse_loss(input_std, target_std)

    def forward(self, x, y):
        B, C, H, W = x.shape

        """
            Define forward process using torch operations

            loss : content loss + style loss
            img_result : style transferred image
                         (output of the decoder network)
        """
        style_feats = self.encode_with_intermediate(y)
        content_feat = self.encode(x)
        t = self.AdaINLayer(content_feat, style_feats[-1])
        if not self.training:
            img_result = []
            alpha = 0.
            a = alpha * t + (1 - alpha) * content_feat
            img_result.append(self.decoder(a))
            alpha = 0.25
            a = alpha * t + (1 - alpha) * content_feat
            img_result.append(self.decoder(a))
            alpha = 0.5
            a = alpha * t + (1 - alpha) * content_feat
            img_result.append(self.decoder(a))
            alpha = 0.75
            a = alpha * t + (1 - alpha) * content_feat
            img_result.append(self.decoder(a))
            alpha = 1.
            a = alpha * t + (1 - alpha) * content_feat
            img_result.append(self.decoder(a))

            return img_result

        img_result = self.decoder(t)
        g_t_feats = self.encode_with_intermediate(img_result)

        loss_c = self.calc_loss(g_t_feats[-1], t, 'content')
        loss_s = self.calc_loss(g_t_feats[0], style_feats[0], 'style')
        for i in range(1, 4):
            loss_s += self.calc_loss(g_t_feats[i], style_feats[i], 'style')

        loss = loss_c + self.w_style*loss_s

        return loss, img_result



"""
    Task 2. Complete training code.

    Following skeleton code assumes that you have multiple GPUs
    You can freely change any of parameters
"""
def train():
    gc.disable()

    # Parameters
    path_snapshot = 'snapshots'
    path_content = 'dataset/train/content'
    path_style = 'dataset/train/style'

    if not os.path.exists(path_snapshot):
        os.makedirs(path_snapshot)

    batch_size = 16
    weight_decay = 1.0e-5
    num_epoch = 600
    lr_init = 0.0001#0.001
    lr_decay_step = num_epoch/2
    momentum = 0.9
    #device_ids = [0, 1, 2]
    w_style = 10
    alpha = 1
    disp_step = 10

    # Data loader
    dm = DataManager(path_content, path_style, random_crop=True)
    dl = DataLoader(dm, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=False)

    num_train = dm.num
    num_batch = np.ceil(num_train / batch_size)
    loss_train_avg = np.zeros(num_epoch)

    net = StyleTransferNet(w_style, alpha)
    net = nn.DataParallel(net.cuda(), device_ids=range(torch.cuda.device_count()))

    """
        Define your optimizer in here to train only the decoder network
    """
    optimizer = torch.optim.Adam(net.module.decoder.parameters(), lr=lr_init)

    # For visualization
    vis = visdom.Visdom()

    vis_loss = VisdomLine(vis, dict(title='Training Loss', markers=True))
    vis_image = VisdomImage(vis, dict(title='Content / Style / Result'))
    vis_image1 = VisdomImage(vis, dict(title='Content / Alpha(0,0.25,0.5,0.75,1) / Style'))

    # Start training
    for epoch in range(0, num_epoch):
        net.train()
        running_loss_train = 0
        np.random.shuffle(dl.dataset.list_style)

        for i, data in enumerate(dl, 0):
            img_con = data['content']
            img_sty = data['style']

            img_con = Variable(img_con, requires_grad=False).cuda()
            img_sty = Variable(img_sty, requires_grad=False).cuda()

            optimizer.zero_grad()

            loss, img_result = net(img_con, img_sty)

            loss = torch.mean(loss)
            loss.backward()

            optimizer.step()

            running_loss_train += loss

            if (i+1)%disp_step==0:
                print('[%s] Epoch %3d / %3d, Batch %5d / %5d, Loss = %12.8f' %
                  (str(datetime.now())[:-3], epoch + 1, num_epoch,
                   i + 1, num_batch, loss))


        loss_train_avg[epoch] = running_loss_train / num_batch

        print('[%s] Epoch %3d / %3d, Avg Loss = %12.8f' % \
              (str(datetime.now())[:-3], epoch + 1, num_epoch,
               loss_train_avg[epoch]))

        #optimizer = LearningRateScheduler(optimizer, epoch + 1, lr_decay_step=lr_decay_step)

        # Display using visdom
        vis_loss.Update(np.arange(epoch + 1) + 1, loss_train_avg[0:epoch + 1])

        '''
        img_cat = torch.cat((img_con, img_sty), dim=3)
        img_cat = torch.unbind(img_cat, dim=0)
        img_cat = torch.cat(img_cat, dim=1)
        img_cat = dm.restore(img_cat.data.cpu())
        vis_image.Update(torch.clamp(img_cat, 0, 1))
        '''
        net.eval()
        img_result = net(img_con, img_sty)
        img_result.insert(0, img_con)
        img_result.append(img_sty)
        img_cat = torch.cat(img_result, dim=3)
        img_cat = torch.unbind(img_cat, dim=0)
        img_cat = torch.cat(img_cat, dim=1)
        img_cat = dm.restore(img_cat.data.cpu())
        output_img = torch.clamp(img_cat, 0, 1)
        vis_image1.Update(output_img)

        if epoch%10 == 0:
            tt=transforms.ToPILImage()(output_img)
            tt.save('Output/{}.png'.format(epoch+1))

        # Snapshot
        if (epoch % 100) == 0:
            #ipdb.set_trace()
            torch.save(net.state_dict(), '%s/epoch_%06d.pth' % (path_snapshot, epoch + 1))

        gc_collected = gc.collect()
        gc.disable()

    torch.save(net.state_dict(), '%s/epoch_%06d.pth' % (path_snapshot, epoch + 1))
    print('Training finished.')



if __name__ == '__main__':
    train()
