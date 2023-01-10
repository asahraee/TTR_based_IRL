import torch
import torch.nn as nn
import torchvision.models as models

# For test

from datasets import *

class ImageStateResnet18(nn.Module):
    def __init__(self, image_size, scalar_size, hidden_size,
            output_size):
        '''to be done'''
        super(ImageStateResnet18, self).__init__()
        resnet18 = models.resnet18(pretrained=True)
        resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=(7,7),
                stride=(2,2), padding=(3,3), bias=False)
        self.resnet = nn.Sequential(*list(resnet18.children())[:-1])
        self.lin_input = resnet18.fc.in_features
        self.linear0 = nn.Linear(scalar_size, self.lin_input)
        self.linear1 = nn.Linear(self.lin_input*2, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.softmax = nn.Softmax()
        self.linear3 = nn.Linear(hidden_size, output_size)


    def forward(self, image, state):
        # state = [xr, yr, theta_s]
        out1 = self.resnet(image).view(-1, self.lin_input)
        out2 = self.linear0(state)
        out2 = self.relu(out2).view(-1, self.lin_input)
        #print('ou1shape: ', out1.shape)
        #print('ou2shape: ', out2.shape)
        out = torch.cat((out1, out2), dim=1)
        out = self.linear1(out)
        out = self.relu(out)
        #out = self.softmax(out)
        out = self.linear2(out)
        out = self.softmax(out)
        #out = self.relu(out)
        out = self.linear3(out)
        return out

class ImageStateLidarResnet18(nn.Module):

    #???????????????????? here
    def __init__(self, image_size, state_size, rng_size, hidden_size,
            output_size):
        '''to be done'''
        super(ImageStateResnet18, self).__init__()
        resnet18 = models.resnet18(pretrained=True)
        resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=(7,7),
                stride=(2,2), padding=(3,3), bias=False)
        self.resnet = nn.Sequential(*list(resnet18.children())[:-1])
        self.lin_input = resnet18.fc.in_features
        self.linear11 = nn.Linear(state_size, self.lin_input)
        self.linear12 = nn.linear(self.lin_input, self.lin_input)
        self.linear21 = nn.linear(rng_size, self.lin_input)
        self.linear22 = nn.linear(self.lin_input, self.lin_input)
        self.linear31 = nn.Linear(self.lin_input*3, hidden_size)
        self.relu = nn.ReLU()
        self.linear32 = nn.Linear(hidden_size, hidden_size)
        self.softmax = nn.Softmax()
        self.linear33 = nn.Linear(hidden_size, output_size)


    def forward(self, image, state):
        # state = [xr, yr, theta_s]
        out0 = self.resnet(image).view(-1, self.lin_input)
        
        out1 = self.linear11(state)
        out1 = self.relu(out1).view(-1, self.lin_input)
        out1 = self.linear12(out1)
        out1 = self.relu(out1)

        out2 = self.linear21(state)
        out2 = self.relu(out2).view(-1, self.lin_input)
        out2 = self.linear22(out2)
        out2 = self.relu(out2)

        #print('ou1shape: ', out1.shape)
        #print('ou2shape: ', out2.shape)
        out = torch.cat((out0, out1, out2), dim=1)
        out = self.linear31(out)
        out = self.relu(out)
        #out = self.softmax(out)
        out = self.linear32(out)
        out = self.softmax(out)
        #out = self.relu(out)
        out = self.linear33(out)
        return out


class ImageStateResnet152(nn.Module):
    def __init__(self, image_size, scalar_size, hidden_size,
            output_size):
        '''to be done'''
        super(ImageStateResnet152, self).__init__()
        resnet152 = models.resnet152(pretrained=True)
        resnet152.conv1 = nn.Conv2d(1, 64, kernel_size=(7,7),
                stride=(2,2), padding=(3,3), bias=False)
        self.resnet = nn.Sequential(*list(resnet152.children())[:-1])
        self.lin_input = resnet152.fc.in_features
        self.linear0 = nn.Linear(scalar_size, self.lin_input)
        self.linear1 = nn.Linear(self.lin_input*2, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.softmax = nn.Softmax()
        self.linear3 = nn.Linear(hidden_size, output_size)


    def forward(self, image, state):
        # state = [xr, yr, theta_s]
        out1 = self.resnet(image).view(-1, self.lin_input)
        out2 = self.linear0(state)
        out2 = self.relu(out2).view(-1, self.lin_input)
        #print('ou1shape: ', out1.shape)
        #print('ou2shape: ', out2.shape)
        out = torch.cat((out1, out2), dim=1)
        out = self.linear1(out)
        out = self.relu(out)
        #out = self.softmax(out)
        out = self.linear2(out)
        out = self.softmax(out)
        #out = self.relu(out)
        out = self.linear3(out)
        return out

class ImageStateAlex(nn.Module):
    def __init__(self, image_size, scalar_size, hidden_size, output_size):
        '''to be done'''
        super(ImageStateAlex, self).__init__()
        alex = models.alexnet(pretrained=True)
        alex.features[0] = nn.Conv2d(1, 64, kernel_size=(11,11),
                stride=(4,4), padding=(2,2))
        alex.classifier[6] = nn.Linear(in_features=4096, out_features=hidden_size, bias=True)
        self.alex = alex
        self.h_size = hidden_size
        self.linear0 = nn.Linear(scalar_size, hidden_size)
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size*2, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.softmax = nn.Softmax()
        self.linear1e = nn.Linear(hidden_size, hidden_size)
        self.linear2e = nn.Linear(hidden_size, hidden_size)
        self.linear3e = nn.Linear(hidden_size, hidden_size)
        self.linear4e = nn.Linear(hidden_size, hidden_size)
        self.linear5e = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, output_size)

    def forward(self, image, state):
        # state = [xr, yr, theta_s]
        out1 = self.alex(image)
        out2 = self.linear0(state)
        out2 = self.relu(out2)
        out2 = self.linear1(out2)
        out2 = self.relu(out2).view(-1, self.h_size)
        #print('ou1shape: ', out1.shape)
        #print('ou2shape: ', out2.shape)
        out = torch.cat((out1, out2), dim=1)
        #print('outshape: ', out.shape)
        out = self.linear2(out)
        out = self.relu(out)
        #out = self.softmax(out)
        out = self.linear3(out)
        #out = self.softmax(out)
        out = self.relu(out)
        
        ############ to se if the performance changes
        out = self.linear1e(out)
        out = self.relu(out)
        out = self.linear2e(out)
        out = self.relu(out)
        out = self.linear3e(out)
        out = self.relu(out)
        out = self.linear4e(out)
        out = self.relu(out)
        out = self.linear5e(out)
        out = self.relu(out)
        #############################
        out = self.linear4(out)
        out  = self.relu(out)
        #out = torch.sigmoid(out)*10
        return torch.squeeze(out)


class ImageStateSimpleConv(nn.Module):
    '''To be done'''
class ImageStateSimpleLinear(nn.Module):
    '''To be done'''


def test():
    model1 = models.alexnet(pretrained=True)
    print('model1: ', model1)
    model2 = ImageStateAlex(200, 3, 50, 1)
    #print('model2: ', model2)
    model3 = models.resnet18(pretrained=True)
    print('model3: ', model3)

    datadir = '/root/Desktop/project/data_1'
    imdir = os.path.join(datadir, 'images/local/')
    txt = os.path.join(datadir, 'csv/pos_ttr.csv')
    lb = os.path.join(datadir, 'csv/image_labels.csv')
    dataset = TrueLocMapDubin3D(txt, lb, imdir)
    im, st, ttr = dataset[0]
    st2 = st
    im2 = im.view(1,1,199,199)
    prediction = model2(im2, st2)
    print('prediction= ',prediction)
    print('Ground_truth= ', ttr)

if __name__=='__main__': test()
