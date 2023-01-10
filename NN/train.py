import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import time
import sys
sys.path.append('/root/Desktop/project/')

from models import *
from datasets import *
import params.dubins3D_params as db3


class TrainerDubins3D():
    def __init__(self, **kwargs):
        '''to be done'''
        # kwargs = {batch_size: , num_epochs: , log_dir: ,
        #   validation_ratio: , test_ratio: }
        self._batch_size = kwargs['batch_size']\
                if 'batch_size' in kwargs else db3.batch_size
        self._num_epochs = kwargs['num_epochs']\
                if 'num_epochs' in kwargs else db3.num_epochs
        self._val_ratio = kwargs['validation_ration']\
                if 'validation_ratio' in kwargs else\
                db3.validation_ratio
        self._test_ratio = kwargs['test_ratio']\
                if 'test_ratio' in kwargs else db3.test_ratio

        self._logdir = kwargs['log_dir']\
                if 'log_dir' in kwargs else\
                db3.trainer_log_dir
        self._saved_model_path = kwargs['saved_model']\
                if 'saved_model' in kwargs else\
                db3.saved_model

        self.device = torch.device("cuda")\
                if torch.cuda.is_available() else "cpu"
        
        if self.device == 'cpu':
            print("Couldn't recognize GPU")

        #tstr = time.strftime("%Y%m%d_%H%M%S")
        #self._logdir += \
        #        f'/lr_{db3.learning_rate}_bs_{db3.batch_size}_{tstr}/'
        #print('log_dir_now = ', self._logdir)
        
        #self.writer = SummaryWriter(self._logdir)

        if not os.path.exists(self._logdir):
            os.mkdir(self._logdir)
        if not os.path.exists(self._saved_model_path):
            os.mkdir(self._saved_model_path)

    
    def train_with_true_local_map(self, run_id=db3.run_id, data_dir=db3.data_log_dir,
            learning_rate=db3.learning_rate, reg=db3.regularization,
            reg_lambda=db3.regularization_lambda, image_size=db3.size_pix):
        
        tstr = time.strftime("%Y%m%d_%H%M%S")
        writer_log_dir = self._logdir + \
                f'/run_{run_id}_lr_{learning_rate}_reg_{reg}_{tstr}/'
        self.writer = SummaryWriter(writer_log_dir)

        saved_model_path = os.path.join(self._saved_model_path,
                f'best_model_{run_id}.pt')
        
        # Constructing file paths
        # for global images
        #im_path = os.path.join(data_dir, 'images/global/')
        # for local images
        im_path = os.path.join(data_dir, 'images/local/')
        text_file = os.path.join(data_dir, 'csv/pos_ttr.csv')
        label_file = os.path.join(data_dir, 'csv/image_labels.csv')

        # Defining initial dataset
        init_data = TrueLocMapDubin3D(text_file, label_file, im_path,
                target_transform_threshold=1000)
        #init_data = TrueGlobMapDubin3D(text_file, label_file, im_path,
                #target_transform_threshold=1000) 
        # Splitting data into train, validation and test sets

        #val_size = int(self._val_ratio*len(init_data))
        val_size = 60
        test_size = int(self._test_ratio*len(init_data))
        train_size = len(init_data) - test_size - val_size
        
        train_set, val_set, test_set = random_split(init_data,
                [train_size, val_size, test_size])
        dataset = {'train':train_set, 'val':val_set, 'test':test_set}
        
        # Defining dataloader for training and validation
        data_loader = {x: DataLoader(dataset[x],
            batch_size=self._batch_size, shuffle=True)\
                    for x in ['train', 'val']}

        # choosing model, loss, and optimizer

        #model = ImageStateResnet152(image_size[0], 3, 50, 1)
        #model = ImageStateResnet18(image_size[0], 3, 50, 1)
        model = ImageStateAlex(image_size[0], 3, 50, 1)
        model = model.to(self.device)
        
        #loss = nn.MSELoss()
        loss = nn.L1Loss()
        optimizer = torch.optim.Adam(model.parameters(),
                lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                factor=0.5, patience=40, cooldown=5)
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                #step_size=40, gamma=0.1)
        layers = [x.grad for x in model.parameters()]
        sum_of_loss={x:0 for x in ['train', 'val']}

        # For Saving the model
        best_loss = np.inf

        for epoch in range(self._num_epochs):
            # Each epoch has a training and a validation phase
            print ("epoch : {}".format(epoch))
            #for phase in ['train', 'val']:
            for phase in ['train', 'val']: 

                sum_of_loss[phase]=0

                if phase == 'train':
                    model.train()
                if phase == 'val':
                    model.eval()

                for i, (im, st, ttr) in enumerate(data_loader[phase]):
                    # im and st stand for image and state respectively
                    # Sending Data to gpu
                    im = im.to(self.device)
                    st = st.to(self.device)
                    ttr = ttr.to(self.device)
                    if reg == 'L1':
                        reg_loss = sum(torch.norm(p,1)\
                                for p in model.parameters())
                    elif reg == 'L2':
                        reg_loss = sum(torch.norm(p,2)\
                                for p in model.parameters())
                    else:
                        reg_loss = torch.tensor(0.)

                    # Forward pass
                    # if validation phase; grad is disabled
                    with torch.set_grad_enabled(phase=='train'):
                        prediction = model(im, st)
                        l = loss(ttr, prediction) + reg_lambda*reg_loss
                    
                    # Backward and weight update only in train phase
                    if phase=='train':
                        optimizer.zero_grad()
                        #self.print_grads(model, 0)
                        l.backward()
                        optimizer.step()
                        #self.print_grads(model, "after step", 0)
                        if i % 2 == 0:
                            print(f'iteration {i}, epoch {epoch}/{self._num_epochs}; Training Loss = {l.item()}')
                    sum_of_loss[phase] = sum_of_loss[phase] + l.item()
                
                loss_name = 'Epoch Training Loss' if phase=='train'\
                        else 'Epoch Validation Loss'
                epo_train_loss = sum_of_loss['train']/(i+1)
                epo_val_loss = sum_of_loss['val']/(i+1)
                self.writer.add_scalar(loss_name,
                        sum_of_loss[phase]/(i+1), epoch)
                #self.writer.add_scalar('sched_lr', scheduler.get_lr()[0], epoch)
                self.writer.add_scalar('optim_lr', optimizer.param_groups[0]['lr'],
                        epoch)

                print('-----')
            # Scheduler setup
            scheduler.step(epo_train_loss)
            if optimizer.param_groups[0]['lr'] <= 0.0625*learning_rate:
            #if optimizer.param_groups[0]['lr'] <= 0.015625*learning_rate:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = learning_rate

            # Saving best model
            if epo_val_loss < best_loss:
                print('Saving Model ...')
                best_loss = epo_val_loss
                torch.save(model, saved_model_path)


    def print_grads(self, model, phase, layer):
        #print (len(model.alex.features))
        print ("***********************")
        print ("phase :{}".format(phase))
        if model.alex.features[layer].weight.grad is not None:
            #print( model.alex.features[layer])
            #print (model.alex.features[layer].weight.grad.shape)
            #print (model.alex.features[0].weight.grad)
            grads = model.alex.features[layer].weight.grad
            #grads = model.linear0.weight.grad
            print (torch.count_nonzero(grads))
            print ("************************")        
#################here??????????!!!!!!!!!!!!!##################
    def train_with_raw_lidar(self, data_dir=db3.data_log_dir,
            learning_rate=db3.learning_rate, image_size=db3.size_pix):
        '''to be done'''

        print('dir: ', data_dir, type(data_dir))
        # Constructing file paths
        # for global images
        #im_path = os.path.join(data_dir, 'images/global/')
        # for local images
        im_path = os.path.join(data_dir, 'images/local/')
        text_file = os.path.join(data_dir, 'csv/pos_ttr.csv')
        label_file = os.path.join(data_dir, 'csv/image_labels.csv')

        # Defining initial dataset
        init_data = TrueLocMapDubin3D(text_file, label_file, im_path,
                target_transform_threshold=1000)
        #init_data = TrueGlobMapDubin3D(text_file, label_file, im_path,
                #target_transform_threshold=1000) 
        # Splitting data into train, validation and test sets

        #val_size = int(self._val_ratio*len(init_data))
        val_size = 60
        test_size = int(self._test_ratio*len(init_data))
        train_size = len(init_data) - test_size - val_size

        train_set, val_set, test_set = random_split(init_data,
                [train_size, val_size, test_size])
        dataset = {'train':train_set, 'val':val_set, 'test':test_set}

        # Defining dataloader for training and validation
        data_loader = {x: DataLoader(dataset[x],
            batch_size=self._batch_size, shuffle=True)\
                    for x in ['train', 'val']}

        # choosing model, loss, and optimizer
        #model = ImageStateResnet152(image_size[0], 3, 50, 1)
        #model = ImageStateResnet18(image_size[0], 3, 50, 1)
        model = ImageStateAlex(image_size[0], 3, 50, 1)
        model = model.to(self.device)
        #loss = nn.MSELoss()
        loss = nn.L1Loss()
        optimizer = torch.optim.Adam(model.parameters(),
                lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                factor=0.5, patience=40, cooldown=5)
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                #step_size=40, gamma=0.1)
        layers = [x.grad for x in model.parameters()]
        sum_of_loss={x:0 for x in ['train', 'val']}
        for epoch in range(self._num_epochs):
            # Each epoch has a training and a validation phase
            print ("epoch : {}".format(epoch))
            #for phase in ['train', 'val']:
            for phase in ['train', 'val']:  #TODO

                sum_of_loss[phase]=0

                if phase == 'train':
                    model.train()
                if phase == 'val':
                    model.eval()

                for i, (im, st, ttr) in enumerate(data_loader[phase]):
                    # im and st stand for image and state respectively
                    # Sending Data to gpu
                    im = im.to(self.device)
                    st = st.to(self.device)
                    ttr = ttr.to(self.device)

                    # Forward pass
                    # if validation phase; grad is disabled
                    with torch.set_grad_enabled(phase=='train'):
                        prediction = model(im, st)
                        #print ("prediction: {}".format(prediction))
                        #print ("ground truth: {}".format(ttr))
                        #print('prediction_size = ', prediction)
                        l = loss(ttr, prediction)
                        #print('ttr = ', ttr.size())

                    # Backward and weight update only in train phase
                    if phase=='train':
                        optimizer.zero_grad()
                        #self.print_grads(model, 0)
                        l.backward()
                        optimizer.step()
                        #self.print_grads(model, "after step", 0)
                        if i % 2 == 0:
                            print(f'iteration {i}, epoch {epoch}/{self._num_epochs}; Training Loss = {l.item()}')
                    sum_of_loss[phase] = sum_of_loss[phase] + l.item()

                loss_name = 'Epoch Training Loss' if phase=='train'\
                        else 'Epoch Validation Loss'
                epo_train_loss = sum_of_loss['train']/(i+1)
                self.writer.add_scalar(loss_name,
                        sum_of_loss[phase]/(i+1), epoch)
                #self.writer.add_scalar('sched_lr', scheduler.get_lr()[0], epoch)
                self.writer.add_scalar('optim_lr', optimizer.param_groups[0]['lr'],
                        epoch)

                print('-----')
            scheduler.step(epo_train_loss)
            if optimizer.param_groups[0]['lr'] <= 0.0625*learning_rate:
            #if optimizer.param_groups[0]['lr'] <= 0.015625*learning_rate:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = learning_rate

    def print_grads(self, model, phase, layer):
        #print (len(model.alex.features))
        print ("***********************")
        print ("phase :{}".format(phase))
        if model.alex.features[layer].weight.grad is not None:
            #print( model.alex.features[layer])
            #print (model.alex.features[layer].weight.grad.shape)
            #print (model.alex.features[0].weight.grad)
            grads = model.alex.features[layer].weight.grad
            #grads = model.linear0.weight.grad
            print (torch.count_nonzero(grads))
            print ("************************")
        

    def train_with_lidar_map():
        '''to be done'''

def main():
    print(db3.data_log_dir)
    T = TrainerDubins3D()
    T.train_with_true_local_map()



if __name__=='__main__': main()
