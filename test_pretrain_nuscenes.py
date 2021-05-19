#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import time
import argparse
import sys
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
import errno

from network.BEV_Unet import BEV_Unet
from network.ptBEV import ptBEVnet
from dataloader.dataset_nuscenes import Nuscenes, map_name_from_segmentation_class_to_segmentation_index
from dataloader.dataset import collate_fn_BEV,collate_fn_BEV_test,spherical_dataset,voxel_dataset
#ignore weird np warning
import warnings
warnings.filterwarnings("ignore")

def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    bin_count=np.bincount(
        n * label[k].astype(int) + pred[k], minlength=n ** 2)
    return bin_count[:n ** 2].reshape(n, n)

def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

def fast_hist_crop(output, target, unique_label):
    hist = fast_hist(output.flatten(), target.flatten(), np.max(unique_label)+1)
    hist=hist[unique_label,:]
    hist=hist[:,unique_label]
    return hist

def SemKITTI2train(label):
    if isinstance(label, list):
        return [SemKITTI2train_single(a) for a in label]
    else:
        return SemKITTI2train_single(label)

def SemKITTI2train_single(label):
    return label - 1 # uint8 trick

def train2SemKITTI(input_label):
    # delete 0 label (uses uint8 trick : 0 - 1 = 255 )
    return input_label + 1

def main(args):
    data_path = args.data_dir
    test_batch_size = args.test_batch_size
    model_save_path = args.model_save_path
    output_path = args.test_output_path
    compression_model = args.grid_size[2]
    grid_size = args.grid_size
    visibility = args.visibility
    pytorch_device = torch.device('cuda:0')
    model = args.model
    if model == 'polar':
        fea_dim = 9
        circular_padding = True
    elif model == 'traditional':
        fea_dim = 7
        circular_padding = False

    #prepare miou fun
    unique_label_str=list(map_name_from_segmentation_class_to_segmentation_index)[1:]
    unique_label=np.asarray([map_name_from_segmentation_class_to_segmentation_index[s] for s in unique_label_str]) - 1

    # prepare model
    my_BEV_model=BEV_Unet(n_class=len(unique_label), n_height = compression_model, input_batch_norm = True, dropout = 0.5, circular_padding = circular_padding, use_vis_fea=visibility)
    my_model = ptBEVnet(my_BEV_model, pt_model = 'pointnet', grid_size =  grid_size, fea_dim = fea_dim, max_pt_per_encode = 256,
                            out_pt_fea_dim = 512, kernal_size = 1, pt_selection = 'random', fea_compre = compression_model)
    if os.path.exists(model_save_path):
        my_model.load_state_dict(torch.load(model_save_path))
    my_model.to(pytorch_device)

    # prepare dataset
    test_pt_dataset = Nuscenes(data_path + '/test/', version = 'v1.0-test', split = 'test', return_ref = True)
    val_pt_dataset = Nuscenes(data_path + '/trainval/', version = 'v1.0-trainval', split = 'val', return_ref = True)
    if model == 'polar':
        test_dataset=spherical_dataset(test_pt_dataset, grid_size = grid_size, ignore_label = 0, fixed_volume_space = True, return_test= True,max_volume_space=[50,np.pi,3], min_volume_space=[0,-np.pi,-5])
        val_dataset=spherical_dataset(val_pt_dataset, grid_size = grid_size, ignore_label = 0, fixed_volume_space = True,max_volume_space=[50,np.pi,3], min_volume_space=[0,-np.pi,-5])
    elif model == 'traditional':
        test_dataset=voxel_dataset(test_pt_dataset, grid_size = grid_size, ignore_label = 0, fixed_volume_space = True, return_test= True,max_volume_space=[50,50,3], min_volume_space=[-50,-50,-5])
        val_dataset=voxel_dataset(val_pt_dataset, grid_size = grid_size, ignore_label = 0, fixed_volume_space = True, max_volume_space=[50,50,3], min_volume_space=[-50,-50,-5])
    test_dataset_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                                    batch_size = test_batch_size,
                                                    collate_fn = collate_fn_BEV_test,
                                                    shuffle = False,
                                                    num_workers = 4)
    val_dataset_loader = torch.utils.data.DataLoader(dataset = val_dataset,
                                                    batch_size = test_batch_size,
                                                    collate_fn = collate_fn_BEV,
                                                    shuffle = False,
                                                    num_workers = 4)

    # validation
    print('*'*80)
    print('Test network performance on validation split')
    print('*'*80)
    pbar = tqdm(total=len(val_dataset_loader))
    my_model.eval()
    hist_list = []
    time_list = []
    with torch.no_grad():
        for i_iter_val,(val_vox_fea,val_vox_label,val_grid,val_pt_labs,val_pt_fea) in enumerate(val_dataset_loader):
            val_vox_fea_ten = val_vox_fea.to(pytorch_device)
            val_vox_label = SemKITTI2train(val_vox_label)
            val_pt_labs = SemKITTI2train(val_pt_labs)
            val_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in val_pt_fea]
            val_grid_ten = [torch.from_numpy(i[:,:2]).to(pytorch_device) for i in val_grid]
            val_label_tensor=val_vox_label.type(torch.LongTensor).to(pytorch_device)

            torch.cuda.synchronize()
            start_time = time.time()
            if visibility:
                predict_labels = my_model(val_pt_fea_ten, val_grid_ten, val_vox_fea_ten)
            else:
                predict_labels = my_model(val_pt_fea_ten, val_grid_ten)
            torch.cuda.synchronize()
            time_list.append(time.time()-start_time)

            predict_labels = torch.argmax(predict_labels,dim=1)
            predict_labels = predict_labels.cpu().detach().numpy()
            for count,i_val_grid in enumerate(val_grid):
                hist_list.append(fast_hist_crop(predict_labels[count,val_grid[count][:,0],val_grid[count][:,1],val_grid[count][:,2]],val_pt_labs[count],unique_label))
            pbar.update(1)
    iou = per_class_iu(sum(hist_list))
    print('Validation per class iou: ')
    for class_name, class_iou in zip(unique_label_str,iou):
        print('%s : %.2f%%' % (class_name, class_iou*100))
    val_miou = np.nanmean(iou) * 100
    del val_vox_label,val_grid,val_pt_fea,val_grid_ten
    pbar.close()
    print('Current val miou is %.3f ' % val_miou)
    print('Inference time per %d is %.4f seconds\n' %
        (test_batch_size,np.mean(time_list)))
    
    # test
    print('*'*80)
    print('Generate predictions for test split')
    print('*'*80)
    pbar = tqdm(total=len(test_dataset_loader))
    with torch.no_grad():
        for i_iter_test,(test_vox_fea,_,test_grid,_,test_pt_fea,test_index) in enumerate(test_dataset_loader):
            # predict
            test_vox_fea_ten = test_vox_fea.to(pytorch_device)
            test_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in test_pt_fea]
            test_grid_ten = [torch.from_numpy(i[:,:2]).to(pytorch_device) for i in test_grid]

            if visibility:
                predict_labels = my_model(test_pt_fea_ten,test_grid_ten,test_vox_fea_ten)
            else:
                predict_labels = my_model(test_pt_fea_ten,test_grid_ten)
            predict_labels = torch.argmax(predict_labels,1).type(torch.uint8)
            predict_labels = predict_labels.cpu().detach().numpy()
            # write to label file
            for count,i_test_grid in enumerate(test_grid):
                test_pred_label = predict_labels[count,test_grid[count][:,0],test_grid[count][:,1],test_grid[count][:,2]]
                test_pred_label = train2SemKITTI(test_pred_label)
                test_pred_label = np.expand_dims(test_pred_label,axis=1)
                label_token = test_pt_dataset.train_token_list[test_index[count]]
                new_save_dir = os.path.join(output_path, label_token+'_lidarseg.bin')

                if not os.path.exists(os.path.dirname(new_save_dir)):
                    try:
                        os.makedirs(os.path.dirname(new_save_dir))
                    except OSError as exc:
                        if exc.errno != errno.EEXIST:
                            raise
                test_pred_label = test_pred_label.astype(np.uint32)
                test_pred_label.tofile(new_save_dir)
            pbar.update(1)
    del test_grid,test_pt_fea,test_index
    pbar.close()
    print('Predicted test labels are saved in %s.' % output_path)

if __name__ == '__main__':
    # Testing settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-d', '--data_dir', default='data')
    parser.add_argument('-p', '--model_save_path', default='pretrained_weight/Nuscenes_PolarSeg.pt')
    parser.add_argument('-o', '--test_output_path', default='out/Nuscenes/lidarseg/test')
    parser.add_argument('-m', '--model', choices=['polar','traditional'], default='polar', help='training model: polar or traditional (default: polar)')
    parser.add_argument('-s', '--grid_size', nargs='+', type=int, default = [480,360,32], help='grid size of BEV representation (default: [480,360,32])')
    parser.add_argument('--test_batch_size', type=int, default=1, help='batch size for training (default: 1)')
    parser.add_argument('--visibility', action='store_true', help='use visibility feature')
    
    args = parser.parse_args()
    if not len(args.grid_size) == 3:
        raise Exception('Invalid grid size! Grid size should have 3 dimensions.')

    print(' '.join(sys.argv))
    print(args)
    main(args)