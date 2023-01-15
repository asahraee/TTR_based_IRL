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
        ttrs = pos_ttr[:,-1].astype(np.float32)
        
        self._labels = pd.read_csv(labels_file, sep=',',
                header=None, index_col =0)
        self._image_dir = image_dir
        self._im_transform = image_transform
        #self._vec_transform = vector_transform
        if self._infin_tresh:
            self._target_transform = trans.Lambda(lambda x: a*\
                    (1 - torch.pow(r, torch.floor(x/dt) + 1))/(1-r))
            self._target_trans_infin = a/(1-r)
        self.states = torch.from_numpy(states)
        self.ttrs = torch.from_numpy(ttrs)

    def __getitem__(self, index):
        '''to be done'''
        image_path = os.path.join(self._image_dir,
                self._labels.iloc[index, 0])
        image = read_image(image_path).float()
        if self._im_transform:
            image = self._im_transform(image)
        
        state = self.states[index]
        ttr = self.ttrs[index]
        #if self._vec_transform:
        #    state = self._vec_transform(state)
        if self._infin_tresh:
            if ttr.item() <= self._infin_tresh: 
                ttr = self._target_transform(ttr)
            else:
                ttr = self._target_trans_infin
        return image, state, ttr

    def __len__(self):
        '''to be done'''
        return self._n_samples

class Dubin3D_B(Dataset):
    # This class loads ground truth global maps with relative x,y and initial theta as inputs and ttr as output
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
        ttrs = pos_ttr[:,-1].astype(np.float32)

        self._labels = pd.read_csv(labels_file, sep=',',
                header=None, index_col =0)
        self._glob_labels = [l.split('_')[0][0]+'_'+l.split('_')[0][1:]+'.png'\
                for l in self._labels.iloc[:,0]]
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
    def __getitem__(self, index):
        '''to be done'''
        image_path = os.path.join(self._image_dir,
                self._glob_labels[index])
        image = read_image(image_path).float()
        if self._im_transform:
            image = self._im_transform(image)

        state = self.states[index]
        ttr = self.ttrs[index]
        #if self._vec_transform:
        #    state = self._vec_transform(state)
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
    data_path = '/root/Desktop/data_and_log/data_1'
    image_path = os.path.join(data_path, 'images/local/')
    text_file = os.path.join(data_path, 'csv/pos_ttr.csv')
    label_file = os.path.join(data_path, 'csv/image_labels.csv')

    dataset = TrueLocMapDubin3D(text_file, label_file, image_path,
            target_transform_threshold=1000)
    image_i, state_i, ttr_i = dataset[18]
    print('state_i: ', state_i, 'ttr_i: ', ttr_i)
    image_i, state_i, ttr_i = dataset[19]
    print('state_i: ', state_i, 'ttr_i: ', ttr_i)
    image_i, state_i, ttr_i = dataset[17]
    print('state_i: ', state_i, 'ttr_i: ', ttr_i)
    plt.imshow(image_i[0], cmap='gray')
    plt.show()


    dataset2 = TrueMapLidarRangeDubin3(text_file, label_file, image_path,
            target_transform_threshold=1000)
    image2i, state2i, rng2i, ttr2i = dataset2[10]
    print('state2i: ', state2i, 'ttr2i: ', ttr2i, 'lidar2i', rng2i)

if __name__=='__main__':test()
