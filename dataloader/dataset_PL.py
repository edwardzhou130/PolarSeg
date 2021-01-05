#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import torch
import random
from plyfile import PlyData
from torch.utils import data

class PLY_dataset(data.Dataset):
  def __init__(self, data_path,sample_interval,time_step,label_convert_fun = None,return_ref = False,crop_data = None):
        'Initialization'
        self.return_ref = return_ref
        self.crop_data = crop_data

        # Load point cloud and labels
        plydata = PlyData.read(data_path)
        self.xyz = np.array(np.transpose(np.stack((plydata['vertex']['x'],plydata['vertex']['y'],plydata['vertex']['z'])))).astype(np.float32)        
        try:
            self.class_id = np.array(plydata['vertex']['class'])
        except:
            self.class_id = np.zeros_like(plydata['vertex']['x'],dtype=int)
        if label_convert_fun:  self.class_id = label_convert_fun(self.class_id)
        
        # parse data by timestamp
        GPS_time = plydata['vertex']['GPS_time']
        sample_start_times = np.arange(GPS_time[0]+sample_interval,GPS_time[-1]-sample_interval,time_step)
        start_ind = np.searchsorted(GPS_time,sample_start_times - sample_interval)
        end_ind = np.searchsorted(GPS_time,sample_start_times + sample_interval)
        end_ind[-1] = np.size(GPS_time)

        self.start_end = np.transpose(np.stack((start_ind,end_ind)))
        self.start_end = np.unique(self.start_end,axis=0)
        remain_ind = (self.start_end[:,1] - self.start_end[:,0]) > 1000
        self.start_end = self.start_end[remain_ind,:]

        if self.crop_data:
            remain_ind = (self.start_end[:,1] - self.start_end[:,0]) > crop_data
            self.start_end = self.start_end[remain_ind,:]
        if self.return_ref:
            self.reflectance = np.array(plydata['vertex']['reflectance']).astype(np.float32)
        
  def __len__(self):
        'Denotes the total number of samples'
        return self.start_end.shape[0]

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        start_ind,end_ind = self.start_end[index,:]
        data_indexes = range(start_ind,end_ind)
        if self.crop_data: data_indexes = random.sample(data_indexes , k = self.crop_data)
        xyz_data = np.float32(self.xyz[data_indexes,...]).copy()
        xyz_data = xyz_data - np.mean(xyz_data,keepdims = True,axis = 0)
        data_tuple = (xyz_data, self.class_id[data_indexes,np.newaxis].copy())
        if self.return_ref:
            data_tuple += (self.reflectance[data_indexes,np.newaxis],)
        return data_tuple


from xml.dom import minidom
xmldoc = minidom.parse('../data/paris_lille/coarse_classes.xml')
itemlist = xmldoc.getElementsByTagName('class')
class2coarse = np.array([[int(i.attributes['id'].value),int(i.attributes['coarse'].value)] for i in itemlist]).astype(np.uint32)
coarseID_name={int(i.attributes['coarse'].value):i.attributes['coarse_name'].value for i in itemlist}
for i in range(len(coarseID_name)-1): coarseID_name[i] = coarseID_name.pop(i+1)

def PLfine2coarse(label_array):
    new_label = label_array.copy()
    for i in range(class2coarse.shape[0]):
        new_label[label_array == class2coarse[i,0]] = class2coarse[i,1]
    new_label[new_label == 0] = 256
    new_label -= 1
    return np.uint8(new_label)