#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BEV_Unet(nn.Module):

    def __init__(self,n_class,n_height,dilation = 1,group_conv=False,input_batch_norm = False,dropout = 0.,circular_padding = False):
        super(BEV_Unet, self).__init__()
        self.n_class = n_class
        self.n_height = n_height
        self.network = UNet(n_class*n_height,n_height,dilation,group_conv,input_batch_norm,dropout,circular_padding)

    def forward(self, x):
        x = self.network(x)
        
        x = x.permute(0,2,3,1)
        new_shape = list(x.size())[:3] + [self.n_height,self.n_class]
        x = x.view(new_shape)
        x = x.permute(0,4,1,2,3)
        return x
    
class UNet(nn.Module):
    def __init__(self, n_class,n_height,dilation,group_conv,input_batch_norm, dropout,circular_padding):
        super(UNet, self).__init__()
        self.inc = inconv(n_height, 64, dilation, input_batch_norm, circular_padding)
        self.down1 = down(64, 128, dilation, group_conv, circular_padding)
        self.down2 = down(128, 256, dilation, group_conv, circular_padding)
        self.down3 = down(256, 512, dilation, group_conv, circular_padding)
        self.down4 = down(512, 512, dilation, group_conv, circular_padding)
        self.up1 = up(1024, 256, circular_padding, group_conv = group_conv)
        self.up2 = up(512, 128, circular_padding, group_conv = group_conv)
        self.up3 = up(256, 64, circular_padding, group_conv = group_conv)
        self.up4 = up(128, 64, circular_padding, group_conv = group_conv)
        self.dropout = nn.Dropout(p=dropout)
        self.outc = outconv(64, n_class)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(self.dropout(x))
        return x

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch,group_conv,dilation=1):
        super(double_conv, self).__init__()
        if group_conv:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1,groups = min(out_ch,in_ch)),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1,groups = out_ch),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        x = self.conv(x)
        return x

class double_conv_circular(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch,group_conv,dilation=1):
        super(double_conv_circular, self).__init__()
        if group_conv:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=(1,0),groups = min(out_ch,in_ch)),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_ch, out_ch, 3, padding=(1,0),groups = out_ch),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=(1,0)),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_ch, out_ch, 3, padding=(1,0)),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        #add circular padding
        x = F.pad(x,(1,1,0,0),mode = 'circular')
        x = self.conv1(x)
        x = F.pad(x,(1,1,0,0),mode = 'circular')
        x = self.conv2(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch, dilation, input_batch_norm, circular_padding):
        super(inconv, self).__init__()
        if input_batch_norm:
            if circular_padding:
                self.conv = nn.Sequential(
                    nn.BatchNorm2d(in_ch),
                    double_conv_circular(in_ch, out_ch,group_conv = False,dilation = dilation)
                )
            else:
                self.conv = nn.Sequential(
                    nn.BatchNorm2d(in_ch),
                    double_conv(in_ch, out_ch,group_conv = False,dilation = dilation)
                )
        else:
            if circular_padding:
                self.conv = double_conv_circular(in_ch, out_ch,group_conv = False,dilation = dilation)
            else:
                self.conv = double_conv(in_ch, out_ch,group_conv = False,dilation = dilation)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch, dilation, group_conv, circular_padding):
        super(down, self).__init__()
        if circular_padding:
            self.mpconv = nn.Sequential(
                nn.MaxPool2d(2),
                double_conv_circular(in_ch, out_ch,group_conv = group_conv,dilation = dilation)
            )
        else:
            self.mpconv = nn.Sequential(
                nn.MaxPool2d(2),
                double_conv(in_ch, out_ch,group_conv = group_conv,dilation = dilation)
            )                

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, circular_padding, bilinear=True, group_conv=False):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        elif group_conv:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2,groups = in_ch//2)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        if circular_padding:
            self.conv = double_conv_circular(in_ch, out_ch,group_conv = group_conv)
        else:
            self.conv = double_conv(in_ch, out_ch,group_conv = group_conv)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))
        
        # for padding issues, see 
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x