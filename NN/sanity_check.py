import numpy as np
import torch
import torch.nn as nn
import torchvision.utils as tv
import os
import sys
sys.path.append('/root/Desktop/project/')

from models import *
from datasets import *
import params.dubins3D_params as db3

# Los threshold for good or bad predictions
tresh = 0.6 
# loading the saved model
path = os.path.join(db3.saved_model,
                f'best_model_0.pt')
model = torch.load(path)
device = torch.device("cuda")
model.to(device)
model.eval()

# Defining dataset
data_dir = "/root/Desktop/project/data_6"
im_path = os.path.join(data_dir, 'images/local/')
text_file = os.path.join(data_dir, 'csv/pos_ttr.csv')
label_file = os.path.join(data_dir, 'csv/image_labels.csv')

dataset = TrueLocMapDubin3D(text_file, label_file, im_path,
                target_transform_threshold=1000)
# Defining dataloader
data_loader = DataLoader(dataset,
            batch_size=db3.batch_size, shuffle=True)
# Defining loss function
loss = nn.L1Loss(reduction='none')

# Loading sample data
im, st, ttr = next(iter(data_loader))
im = im.to(device)
st = st.to(device)
ttr = ttr.to(device)
with torch.no_grad():
    pred = model(im, st)
    l = loss(pred, ttr)

low_loss_dir = "/root/Desktop/project/NN/train_log/low_loss/" 
os.makedirs(low_loss_dir, exist_ok=True)
high_loss_dir = "/root/Desktop/project/NN/train_log/high_loss/"
os.makedirs(high_loss_dir, exist_ok=True)
low_i = 0
high_i = 0
low_txt = low_loss_dir + "low.txt" 
high_txt = high_loss_dir + "high.txt"

for ind in range(db3.batch_size):
    if l[ind].item() <= tresh:
        # save model in good list
        with open(low_txt, 'a') as l_txt:
            l_txt.write(f"id = {low_i}, loss = {l[ind].item()}, pred = {pred[ind]}, ttr = {ttr[ind]}, state = {st[ind]} \n")
        im_file = low_loss_dir + f"im_{low_i}.png"
        tv.save_image(im[ind], im_file)
        low_i += 1

    elif l[ind].item() >= 2*tresh:
        # save model in bad list
        with open(high_txt, 'a') as h_txt:
            h_txt.write(f"id = {high_i}, loss = {l[ind].item()}, pred = {pred[ind]}, ttr = {ttr[ind]}, state = {st[ind]} \n")
        im_file = high_loss_dir + f"im_{high_i}.png"
        tv.save_image(im[ind], im_file)
        high_i += 1

