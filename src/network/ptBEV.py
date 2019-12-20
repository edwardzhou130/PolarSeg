#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import multiprocessing
import torch_scatter

class ptBEVnet(nn.Module):
    
    def __init__(self, BEV_net, grid_size, pt_model = 'pointnet', fea_dim = 3, pt_pooling = 'max', kernal_size = 3,
                 out_pt_fea_dim = 64, max_pt_per_encode = 64, cluster_num = 4, pt_selection = 'farthest', fea_compre = None):
        super(ptBEVnet, self).__init__()
        assert pt_pooling in ['max','vlad']
        assert pt_selection in ['random','farthest']
        
        if pt_model == 'pointnet':
            self.PPmodel = nn.Sequential(
                nn.BatchNorm1d(fea_dim),
                
                nn.Linear(fea_dim, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                
                nn.Linear(64, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                
                nn.Linear(128, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                
                nn.Linear(256, out_pt_fea_dim)
            )
        
        self.pt_model = pt_model
        self.BEV_model = BEV_net
        self.pt_pooling = pt_pooling
        self.max_pt = max_pt_per_encode
        self.pt_selection = pt_selection
        self.fea_compre = fea_compre
        self.grid_size = grid_size
        
        #NN stuff
        if kernal_size != 1:
            if self.pt_pooling == 'max':
                self.local_pool_op = torch.nn.MaxPool2d(kernal_size, stride=1, padding=(kernal_size-1)//2, dilation=1)
            else: raise NotImplementedError
        else: self.local_pool_op = None
        
        #parametric pooling
        if self.pt_pooling == 'vlad':
            self.clu_cen = torch.nn.Parameter(torch.randn(cluster_num, out_pt_fea_dim))
        
        if self.pt_pooling == 'max':
            self.pool_dim = out_pt_fea_dim
        elif self.pt_pooling == 'vlad':
            self.pool_dim = self.clu_cen.shape[0]*out_pt_fea_dim
        
        #point feature compression
        if self.fea_compre is not None:
            self.fea_compression = nn.Sequential(
                    nn.Linear(self.pool_dim, self.fea_compre),
                    nn.ReLU())
            self.pt_fea_dim = self.fea_compre
        else:
            self.pt_fea_dim = self.pool_dim
        
    def forward(self, pt_fea, xy_ind):
        cur_dev = pt_fea[0].get_device()
        
        # concate everything
        cat_pt_ind = []
        for i_batch in range(len(xy_ind)):
            cat_pt_ind.append(F.pad(xy_ind[i_batch],(1,0),'constant',value = i_batch))

        cat_pt_fea = torch.cat(pt_fea,dim = 0)
        cat_pt_ind = torch.cat(cat_pt_ind,dim = 0)
        pt_num = cat_pt_ind.shape[0]

        # shuffle the data
        shuffled_ind = torch.randperm(pt_num,device = cur_dev)
        cat_pt_fea = cat_pt_fea[shuffled_ind,:]
        cat_pt_ind = cat_pt_ind[shuffled_ind,:]
        
        # unique xy grid index
        unq, unq_inv, unq_cnt = torch.unique(cat_pt_ind,return_inverse=True, return_counts=True, dim=0)
        unq = unq.type(torch.int64)
        
        # subsample pts
        if self.pt_selection == 'random':
            grp_ind = grp_range_torch(unq_cnt,cur_dev)[torch.argsort(torch.argsort(unq_inv))]
            remain_ind = grp_ind < self.max_pt
        elif self.pt_selection == 'farthest':
            unq_ind = np.split(np.argsort(unq_inv), np.cumsum(unq_cnt[:-1]))
            remain_ind = np.zeros((pt_num,),dtype = np.bool)
            np_cat_fea = cat_pt_fea.detach().cpu().numpy()[:,:3]
            pool_in = []
            for i_inds in unq_ind:
                if len(i_inds) > self.max_pt:
                    pool_in.append((np_cat_fea[i_inds,:],self.max_pt))
            if len(pool_in) > 0:
                pool = multiprocessing.Pool(multiprocessing.cpu_count())
                FPS_results = pool.starmap(parallel_FPS, pool_in)
                pool.close()
                pool.join()
            count = 0
            for i_inds in unq_ind:
                if len(i_inds) <= self.max_pt:
                    remain_ind[i_inds] = True
                else:
                    remain_ind[i_inds[FPS_results[count]]] = True
                    count += 1
            
        cat_pt_fea = cat_pt_fea[remain_ind,:]
        cat_pt_ind = cat_pt_ind[remain_ind,:]
        unq_inv = unq_inv[remain_ind]
        unq_cnt = torch.clamp(unq_cnt,max=self.max_pt)
        
        # process feature
        if self.pt_model == 'pointnet':
            processed_cat_pt_fea = self.PPmodel(cat_pt_fea)
        
        if self.pt_pooling == 'max':
            pooled_data = torch_scatter.scatter_max(processed_cat_pt_fea, unq_inv, dim=0)[0]
        elif self.pt_pooling == 'vlad':
            #get fea for each pt
            diff = torch.unsqueeze(processed_cat_pt_fea,1) - torch.unsqueeze(self.clu_cen,0)#diff.shape == (num_pt,num_clu,num_fea)
            clu_weight = torch.sqrt(torch.sum(diff**2,dim = 2))#diff.shape == (num_pt,num_clu)
            reweighted_data = torch.unsqueeze(clu_weight,2)*diff
            pt_data = reweighted_data.view(processed_cat_pt_fea.shape[0],self.pool_dim)#pt_data.shape == (num_pt,fea_dim)
            
            pooled_data = torch_scatter.scatter_add(pt_data, unq_inv, dim64=0)
            pt_per_grid_tensor = unq_cnt.view(-1,1)
            pooled_data = pooled_data/pt_per_grid_tensor.type(torch.float32)    
        else: raise NotImplementedError
        
        if self.fea_compre:
            processed_pooled_data = self.fea_compression(pooled_data)
        else:
            processed_pooled_data = pooled_data
        
        # stuff pooled data into 4D tensor
        out_data_dim = [len(pt_fea),self.grid_size[0],self.grid_size[1],self.pt_fea_dim]
        out_data = torch.zeros(out_data_dim, dtype=torch.float32).to(cur_dev)
        out_data[unq[:,0],unq[:,1],unq[:,2],:] = processed_pooled_data
        out_data = out_data.permute(0,3,1,2)
        if self.local_pool_op != None:
            out_data = self.local_pool_op(out_data)
        
        # run through network
        net_return_data = self.BEV_model(out_data)
        
        return net_return_data
    
def grp_range_torch(a,dev):
    idx = torch.cumsum(a,0)
    id_arr = torch.ones(idx[-1],dtype = torch.int64,device=dev)
    id_arr[0] = 0
    id_arr[idx[:-1]] = -a[:-1]+1
    return torch.cumsum(id_arr,0)

def parallel_FPS(np_cat_fea,K):
    return  nb_greedy_FRS(np_cat_fea,K)