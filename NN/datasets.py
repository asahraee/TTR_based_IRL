import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import torchvision.transforms as trans
import numpy as np
import pandas as pd

# for test
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

class Dubin3D_A(Dataset):
    # This class loads ground truth local maps with relative x,y and initial theta as inputs, and ttr as output
    def __init__(self, textfile, labels_file, image_dir,
            target_transform_threshold=None, normalize=False, state_mean=None,
            state_std= None, image_mean=None, image_std=None):

        #threshold for geometric sseries to consider a value equal to infinity
        self._infin_tresh = target_transform_threshold
        r = torch.as_tensor(0.999)
        dt = torch.as_tensor(0.01)
        a = 1 * dt

        # Read state and ttr from csv files
        pos_ttr = pd.read_csv(textfile, sep=',',
                header=0, index_col=0).to_numpy()
        self._n_samples = pos_ttr.shape[0]
        # for data number less than 8
        states = np.stack((pos_ttr[:,0], pos_ttr[:,1], pos_ttr[:,2]),
                axis=1).astype(np.float32)
        # for data_8 and after
        #states = np.stack((pos_ttr[:,6], pos_ttr[:,7], pos_ttr[:,2]),
        #        axis=1).astype(np.float32)
        ttrs = pos_ttr[:,-1].astype(np.float32)
        
        # Read image labels for indexing the image
        self._labels = pd.read_csv(labels_file, sep=',',
                header=None, index_col =0)
        self._image_dir = image_dir
        if self._infin_tresh:
            self._target_transform = trans.Lambda(lambda x: a*\
                    (1 - torch.pow(r, torch.floor(x/dt) + 1))/(1-r))
            self._target_trans_infin = a/(1-r)
        self.states = torch.from_numpy(states)
        self.ttrs = torch.from_numpy(ttrs)

        #self._im_transform = image_transform
        if normalize:
            # find state mean and standard deviation
            if state_mean and state_std:
                self._state_mean = state_mean
                self._state_std = state_std
            else:
                self._df = pd.read_csv(textfile, index_col=0)
                self._state_mean, self._state_std = self._compute_state_mean()
            
            ## find image mean and standard deviation
            #if image_mean and image_std:
            #    self._im_mean = image_mean
            #    self._im_std = image_std
            #else:
            #    self._im_mean, self._im_std = self._compute_im_mean()
            
            # use mean and std to create normalization transform
            #self._im_trans = trans.Normalize(self._im_mean, self._im_std)
            self._im_trans = trans.Lambda(lambda im: im / 255)
            self._state_trans = trans.Lambda(lambda st: \
                    (st - self._state_mean)/self._state_std)
        else:
            self._im_trans = None
            self._state_trans = None

    def __getitem__(self, index):
        
        # read image and apply transform
        image_path = os.path.join(self._image_dir,
                self._labels.iloc[index, 0])
        image = read_image(image_path).float()
        if self._im_trans:
            image = self._im_trans(image).float()
        # read state and apply transform
        state = self.states[index]
        if self._state_trans:
            state = self._state_trans(state).float()
        # read ttr and apply transform
        ttr = self.ttrs[index]
        if self._infin_tresh:
            if ttr.item() <= self._infin_tresh: 
                ttr = self._target_transform(ttr)
            else:
                ttr = self._target_trans_infin
        return image, state, ttr

    def __len__(self):
        '''to be done'''
        return self._n_samples
    
    def _compute_state_mean(self):
        df_mean = self._df.mean()
        df_std = self._df.std()
        pos_ttr_mean = [df_mean[key] for key in df_mean.keys()]
        pos_ttr_std = [df_std[key] for key in df_std.keys()]
        
        state_mean = pos_ttr_mean[:-1]
        state_std = pos_ttr_std[:-1]
        
        return torch.tensor(state_mean), torch.tensor(state_std)

    def _compute_im_mean(self):
        '''
        To be done ...
        '''

class Dubin3D_B(Dataset):
    # This class loads ground truth global maps with relative x,y and initial theta as inputs and ttr as output
    def __init__(self, textfile, labels_file, image_dir,
            image_transform=None, target_transform_threshold=None):

        #threshold for geometric sseries to consider a value equal to infinity
        self._infin_tresh = target_transform_threshold
        r = torch.as_tensor(0.999)
        dt = torch.as_tensor(0.01)
        a = 1 * dt
        pos_ttr = pd.read_csv(textfile, sep=',',
                header=0, index_col=0).to_numpy()
        self._n_samples = pos_ttr.shape[0]
        states = np.stack((pos_ttr[:,6], pos_ttr[:,7], pos_ttr[:,2]),
                axis=1).astype(np.float32)
        ttrs = pos_ttr[:,-1].astype(np.float32)

        self._labels = pd.read_csv(labels_file, sep=',',
                header=None, index_col =0)
        self._glob_labels = [l.split('_')[0][0]+'_'+l.split('_')[0][1:]+'.png'\
                for l in self._labels.iloc[:,0]]
        self._image_dir = image_dir
        self._im_transform = image_transform
        if self._infin_tresh:
            self._target_transform = trans.Lambda(lambda x: a*\
                    (1 - torch.pow(r, torch.floor(x/dt) + 1))/(1-r))
            self._target_trans_infin = a/(1-r)
        self.states = torch.from_numpy(states)
        self.ttrs = torch.from_numpy(ttrs)
    def __getitem__(self, index):
        '''to be done'''
        image_path = os.path.join(self._image_dir,
                self._glob_labels[index])
        image = read_image(image_path).float()
        if self._im_transform:
            image = self._im_transform(image)

        state = self.states[index]
        ttr = self.ttrs[index]
        if self._infin_tresh:
            if ttr.item() <= self._infin_tresh:
                ttr = self._target_transform(ttr)
            else:
                ttr = self._target_trans_infin
        return image, state, ttr

    def __len__(self):
        '''to be done'''
        return self._n_samples



class Dubin3D_C(Dataset):
    # This class loads ground truth local maps with relative x,y and initial theta and lidar readings at start position as inputs, and ttr as output
    def __init__(self, textfile, labels_file, image_dir,
            image_transform=None, target_transform_threshold=None):

        #pos_ttr = np.loadtxt(textfile, delimeter=',',
        #        dtype=np.float32, skiprows=1)
        #threshold for geometric sseries to consider a value equal to infinity
        self._infin_tresh = target_transform_threshold
        r = torch.as_tensor(0.999)
        dt = torch.as_tensor(0.01)
        a = 1 * dt
        pos_ttr = pd.read_csv(textfile, sep=',',
                header=0, index_col=0).to_numpy()
        self._n_samples = pos_ttr.shape[0]
        states = np.stack((pos_ttr[:,6], pos_ttr[:,7], pos_ttr[:,2]),
                axis=1).astype(np.float32)
        ttrs = pos_ttr[:,3].astype(np.float32)
        lidar_rngs = pos_ttr[4:-1].astype(np.float32)

        self._labels = pd.read_csv(labels_file, sep=',',
                header=None, index_col =0)
        self._image_dir = image_dir
        self._im_transform = image_transform
        #self._vec_transform = vector_transform
        if self._infin_tresh:
            #self._target_transform = trans.Lambda(lambda x:\
            #        torch.div(\
            #        torch.sub(torch.pow(r, torch.floor(torch.div(x, dt))), 1) ,\
            #        torch.sub(r,1)))
            self._target_transform = trans.Lambda(lambda x: a*\
                    (1 - torch.pow(r, torch.floor(x/dt) + 1))/(1-r))
            self._target_trans_infin = a/(1-r)
        self.states = torch.from_numpy(states)
        self.ttrs = torch.from_numpy(ttrs)
        self.rngs = torch.from_numpy(lidar_rngs)

    def __getitem__(self, index):
        '''to be done'''
        image_path = os.path.join(self._image_dir,
                self._labels.iloc[index, 0])
        image = read_image(image_path).float()
        if self._im_transform:
            image = self._im_transform(image)

        state = self.states[index]
        rng = self.rngs[index]
        ttr = self.ttrs[index]
        #if self._vec_transform:
        #    state = self._vec_transform(state)
        if self._infin_tresh:
            if ttr.item() <= self._infin_tresh:
                ttr = self._target_transform(ttr)
            else:
                ttr = self._target_trans_infin
        return image, state, rng, ttr

    def __len__(self):
        '''to be done'''
        return self._n_samples


def test():
    data_path = '/root/Desktop/data_and_log/data_7'
    image_path = os.path.join(data_path, 'images/local/')
    text_file = os.path.join(data_path, 'csv/pos_ttr.csv')
    label_file = os.path.join(data_path, 'csv/image_labels.csv')

    dataset = Dubin3D_A(text_file, label_file, image_path,
            target_transform_threshold=500, normalize=True)
    image_i, state_i, ttr_i = dataset[18]
    print('image_mean: ', image_i.mean(), ' image_std: ', image_i.std())
    print('state_18: ', state_i, 'ttr_i: ', ttr_i)
    image_i, state_i, ttr_i = dataset[19]
    print('state_19: ', state_i, 'ttr_i: ', ttr_i)
    image_i, state_i, ttr_i = dataset[17]
    print('state_17: ', state_i, 'ttr_i: ', ttr_i)
    


if __name__=='__main__':test()
