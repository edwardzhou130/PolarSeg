import os
import numpy as np
import yaml
from pathlib import Path
from torch.utils import data

from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits

map_name_from_general_to_segmentation_class = {
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.wheelchair': 'ignore',
    'human.pedestrian.stroller': 'ignore',
    'human.pedestrian.personal_mobility': 'ignore',
    'human.pedestrian.police_officer': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'animal': 'ignore',
    'vehicle.car': 'car',
    'vehicle.motorcycle': 'motorcycle',
    'vehicle.bicycle': 'bicycle',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.truck': 'truck',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.emergency.ambulance': 'ignore',
    'vehicle.emergency.police': 'ignore',
    'vehicle.trailer': 'trailer',
    'movable_object.barrier': 'barrier',
    'movable_object.trafficcone': 'traffic_cone',
    'movable_object.pushable_pullable': 'ignore',
    'movable_object.debris': 'ignore',
    'static_object.bicycle_rack': 'ignore',
    'flat.driveable_surface': 'driveable_surface',
    'flat.other': 'other_flat',
    'flat.sidewalk': 'sidewalk',
    'flat.terrain': 'terrain',
    'static.manmade': 'manmade',
    'static.vegetation': 'vegetation',
    'noise': 'ignore',
    'static.other': 'ignore',
    'vehicle.ego': 'ignore'
}

map_name_from_segmentation_class_to_segmentation_index = {
    'ignore': 0,
    'barrier': 1,
    'bicycle': 2,
    'bus': 3,
    'car': 4,
    'construction_vehicle': 5,
    'motorcycle': 6,
    'pedestrian': 7,
    'traffic_cone': 8,
    'trailer': 9,
    'truck': 10,
    'driveable_surface': 11,
    'other_flat': 12,
    'sidewalk': 13,
    'terrain': 14,
    'manmade': 15,
    'vegetation': 16
}

class Nuscenes(data.Dataset):
    def __init__(self, data_path, version = 'v1.0-trainval', split = 'train', return_ref = False):
        assert version in ['v1.0-trainval', 'v1.0-test', 'v1.0-mini']
        if version == 'v1.0-trainval':
            train_scenes = splits.train
            val_scenes = splits.val
        elif version == 'v1.0-test':
            train_scenes = splits.test
            val_scenes = []
        elif version == 'v1.0-mini':
            train_scenes = splits.mini_train
            val_scenes = splits.mini_val
        else:
            raise NotImplementedError
        self.split = split
        self.data_path = data_path
        self.return_ref = return_ref

        self.nusc = NuScenes(version=version, dataroot=data_path, verbose=True)

        self.map_name_from_general_index_to_segmentation_index = {}
        for index in self.nusc.lidarseg_idx2name_mapping:
            self.map_name_from_general_index_to_segmentation_index[index] = map_name_from_segmentation_class_to_segmentation_index[map_name_from_general_to_segmentation_class[self.nusc.lidarseg_idx2name_mapping[index]]]

        
        available_scenes = get_available_scenes(self.nusc)
        available_scene_names = [s['name'] for s in available_scenes]
        train_scenes = list(filter(lambda x: x in available_scene_names, train_scenes))
        val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
        train_scenes = set([available_scenes[available_scene_names.index(s)]['token'] for s in train_scenes])
        val_scenes = set([available_scenes[available_scene_names.index(s)]['token'] for s in val_scenes])

        self.train_token_list,self.val_token_list = get_path_infos(self.nusc,train_scenes,val_scenes)

        print('%s: train scene(%d), val scene(%d)' % (version, len(train_scenes), len(val_scenes)))
        
    def __len__(self):
        'Denotes the total number of samples'
        if self.split == 'train':
            return len(self.train_token_list)
        elif self.split == 'val':
            return len(self.val_token_list)
        elif self.split == 'test':
            return len(self.train_token_list)
    
    def __getitem__(self, index):
        if self.split == 'train':
            sample_token = self.train_token_list[index]
        elif self.split == 'val':
            sample_token = self.val_token_list[index]
        elif self.split == 'test':
            sample_token = self.train_token_list[index]

        lidar_path = os.path.join(self.data_path, self.nusc.get('sample_data', sample_token)['filename'])
        raw_data = np.fromfile(lidar_path, dtype = np.float32).reshape((-1, 5))

        if self.split == 'test':
            annotated_data = np.expand_dims(np.zeros_like(raw_data[:,0],dtype=int),axis=1)
        else:
            lidarseg_path = os.path.join(self.data_path, self.nusc.get('lidarseg', sample_token)['filename'])
            annotated_data = np.fromfile(lidarseg_path, dtype=np.uint8).reshape((-1,1))
            annotated_data = np.vectorize(self.map_name_from_general_index_to_segmentation_index.__getitem__)(annotated_data)
        data_tuple = (raw_data[:,:3], annotated_data)
        if self.return_ref:
            data_tuple += (raw_data[:,3],)
        return data_tuple

    def change_split(self, s):
        assert s in ['train', 'val']
        self.split = s

def get_available_scenes(nusc):
    available_scenes = []
    print('total scene num:', len(nusc.scene))
    for scene in nusc.scene:
        scene_token = scene['token']
        scene_rec = nusc.get('scene', scene_token)
        sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
        sd_rec = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
        has_more_frames = True
        scene_not_exist = False
        while has_more_frames:
            lidar_path, boxes, _ = nusc.get_sample_data(sd_rec['token'])
            if not Path(lidar_path).exists():
                scene_not_exist = True
                break
            else:
                break

        if scene_not_exist:
            continue
        available_scenes.append(scene)
    print('exist scene num:', len(available_scenes))
    return available_scenes

def get_path_infos(nusc,train_scenes,val_scenes):
    train_token_list = []
    val_token_list = []
    for sample in nusc.sample:
        scene_token = sample['scene_token']
        data_token = sample['data']['LIDAR_TOP']
        if scene_token in train_scenes:
            train_token_list.append(data_token)
        else:
            val_token_list.append(data_token)
    return train_token_list, val_token_list

if __name__ == '__main__':
    data_path = '/home/zhou/work/3DSegmentation/point_seg/data/nuscenes'
    dataset = Nuscenes(data_path, version = 'v1.0-mini', return_ref = False)
    data = dataset[0]