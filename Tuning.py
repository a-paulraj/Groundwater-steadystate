#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from numpy.testing._private.nosetester import _numpy_tester
import pandas as pd
import matplotlib.pyplot as plt
import random
import os
#import sys
import torch 
#import importlib
import math
import seaborn as sns
import torch.nn as nn


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(DEVICE)

filename = 'Regression_Data.txt'
path = "/home/apaulraj/CONUS_GroundWater_Mapping"

#  Variable selection:  
# Choices:
# ['WTD', 'elev', 'PME', 'lnK', 'slope', 'TopoIndex', 'Nrelief', 'Neloc', 'Nefrac', 'Norder', 'Nslope', 'NlnK',
#    'Nvark', 'NPME', 'Rrelief', 'Reloc', 'Refrac', 'Rslope', 'Rlnk', 'Rvark', 'RPME', 'Rarea', 'Rareafrac']
target_name = 'WTD'    # variable to be used as the label
covar_names = ['PME', 'lnK', 'TopoIndex', 'slope'] #variables to use as channels

# Nvark has NaN so excluded for now

output_name = 'justcnn-50k'  # Name to give to the trained model

import Shapes
from Shapes import PFslice2D
import utilities as utilities


# In[2]:


ny = 1888
nx = 3342
ncell = nx*ny
index_list = np.array(list(range(ncell)))
from scipy.stats import zscore
# Read in raw data as pandas data frame
filepath = os.path.join(path, filename)
raw_data = pd.read_table(filepath, sep=',', header=0, index_col=None)
#raw_data = raw_data.apply(zscore)
#print(raw_data['WTD'])

#raw_data['WTD'] = np.sqrt((raw_data['WTD']))
#raw_data['PME'] = np.sqrt((-1*raw_data['PME']))
#print(raw_data[raw_data.isna().any(axis=0)])

#Convert the data into a 3D numpy array with 2D matrices for each column
data_array = np.zeros((ny,nx,raw_data.shape[1]))
ztemp=0
for i in list(raw_data):
    #print(i)
    #raw_data.loc[raw_data[i] == -9999, i] = np.nan #replace -9999's with NAs
    temp=raw_data[i].values
    data_array[:, :, ztemp] = np.flipud(temp.reshape(ny, nx, order='C'))
    ztemp += 1

from scipy import stats
print(data_array.shape)
x_min = 0
x_max = nx #1232
y_min = 0 #1150
y_max = ny
data_array = data_array[y_min:y_max,x_min:x_max,:]
print(data_array.shape)

def normalize_2d(matrix):
    return stats.zscore(matrix, axis=None)


WTD = data_array[:,:,0]
wtdmean = np.mean(WTD)
wtdstd = np.std(WTD)
PME = data_array[:,:,2]
lnK = data_array[:,:,3]
TopoIndex = data_array[:,:,5]

# for i in range(23):
#     data_array[:,:,i] = normalize_2d(data_array[:,:,i])

from matplotlib import pyplot as plt
# plt.hist(WTD.ravel(), color='#0504aa', alpha=0.7, rwidth=0.85)
# plt.show()

# plt.hist(PME.ravel(), color='#0504aa', alpha=0.7, rwidth=0.85)
# plt.show()

# plt.hist(lnK.ravel(), color='#0504aa', alpha=0.7, rwidth=0.85)
# plt.show()

# plt.hist(TopoIndex.ravel(), color='#0504aa', alpha=0.7, rwidth=0.85)
# plt.show()

var_names = list(raw_data)
nvar=len(var_names)
var_summary = pd.DataFrame(data=None, index=var_names)

var_summary['max_raw'] = raw_data.max()
var_summary['min_raw'] = raw_data.min()

var_summary['max_val'] = var_summary['max_raw']
var_summary['min_val'] = var_summary['min_raw']
var_summary.loc[['Reloc', 'Refrac', 'Neloc',
                 'Nefrac', 'Rareafrac'], 'min_val'] = 0.0

var_summary['min_scaled'] = np.zeros(nvar)
var_summary['max_scaled'] = np.ones(nvar)

# NOTE - might want a flag for log options on the scaling

#Scale data
data_scaled = data_array.copy()
data_unscaled = data_array.copy()
ztemp = 0
print("Var, Raw Min, Raw Max, Scaled Min, Scaled Max")
counter = 0
for i in list(raw_data):
    min_r = var_summary.loc[i, 'min_raw']
    max_r = var_summary.loc[i, 'max_raw']
    min_i = var_summary.loc[i, 'min_val']
    max_i = var_summary.loc[i, 'max_val']
    min_f = var_summary.loc[i, 'min_scaled']
    max_f = var_summary.loc[i, 'max_scaled']
    print(i, min_i, max_i, min_f, max_f)

    #adjust any values outside the desired range
    temp=data_array[:,:, ztemp]
    if min_r < min_i:
        print('truncating mins')
        temp[temp<min_i] = min_i

    if max_r > max_i:
        print('truncating maxs')
        temp[temp > max_i] = max_i
    
    if (counter==0):
        counter = counter + 1
        print("YOOOOO")
        wtdmin = min_i
        wtdmax = max_i

    #rescale
    data_scaled[:, :, ztemp] = (temp - min_i)/(max_i-min_i) * (max_f-min_f) + min_f
    #print(i, ztemp, np.min(data_scaled[:, :, ztemp]),
    #      np.max(data_scaled[:, :, ztemp]))
    ztemp += 1
    
# print("AFTER SCALING")

# plt.hist(data_scaled[:,:,0].ravel(), color='#0504aa', alpha=0.7, rwidth=0.85)
# plt.show()

# plt.hist(data_scaled[:,:,2].ravel(), color='#0504aa', alpha=0.7, rwidth=0.85)
# plt.show()

# plt.hist(data_scaled[:,:,3].ravel(), color='#0504aa', alpha=0.7, rwidth=0.85)
# plt.show()

# plt.hist(data_scaled[:,:,5].ravel(), color='#0504aa', alpha=0.7, rwidth=0.85)
# plt.show()



xs = []
ys = []
patch_width = 50

x_unique = [[] for i in range(x_max-x_min-(patch_width-1))]
#print(x_unique)

run_size = 50000      # total number of clips to take
#nx=2242
for i in range(run_size):
    currentx = np.random.randint(patch_width, x_max-x_min-(patch_width-1))
    xs.append(currentx)  # 0 to nx-(patch_width-1)
    currenty = np.random.randint(0, y_max-y_min-(patch_width-1))
    #print(currentx)
    #print(x_unique[currentx])
    while(currenty in x_unique[currentx]):
        #print("yo")
        currenty = np.random.randint(0, y_max-y_min-(patch_width-1))
    
    ys.append(currenty)
    x_unique[currentx].append(currenty)
    

    
val_start = int(run_size*0.6)
test_start = int(run_size*0.8)
train_x = xs[:val_start]
train_y = ys[:val_start]

# toggle for "ood"/true testing
# val_x = [0 for i in range(y_max-y_min-(patch_width-1))]
# test_x = [0 for i in range(y_max-y_min-(patch_width-1))]
# val_y = [i for i in range(y_max-y_min-(patch_width-1))]
# test_y = [i for i in range(y_max-y_min-(patch_width-1))]

# print(val_x)
# print(test_x)
# print(val_y)
# print(test_y)

val_x = xs[val_start:test_start]
val_y = ys[val_start:test_start]
test_x = xs[test_start:]
test_y = ys[test_start:]



target_index = var_names.index(target_name)
covar_index=[var_names.index(x) for x in covar_names]


# Figure out data sizes
nchannel = len(covar_names)
D_in = nchannel * patch_width^2
D_out = patch_width ** 2

#from UNet_Experiment import RMM_NN as ConvNet
#from NN_2D import ConvNet
from AndrewCNN import BasicConvNet as ConvNet

print('-----------------------------')
print("  NN Model information")
print()

model = ConvNet(input_channels = 4,
        hidden_channels = 64,
        output_channels = 1,
        kernel_size = 9,
        depth = 11,
        activation = nn.functional.selu)
model.to(DEVICE)

model.verbose = False
model.use_dropout = False

print("-- Model Definition --")
print(model)

print("-- Model Parameters --")
utilities.count_parameters(model)


# In[18]:


from WTDDataset import MyDataset
from WTDDatasetTranspose import MyDataset as transposed
from WTDDataset_Augmented import MyDataset as augmented
from torch.utils.data import DataLoader

train_dataset = MyDataset(data_scaled, train_x, train_y, patch_width, covar_index, target_index)
val_dataset = MyDataset(data_scaled, val_x, val_y, patch_width, covar_index, target_index)
test_dataset = MyDataset(data_scaled, test_x, test_y, patch_width, covar_index, target_index)

train_loader = DataLoader(train_dataset, batch_size = 500)
val_loader = DataLoader(val_dataset, batch_size = 20)
test_loader = DataLoader(test_dataset, batch_size = 20)


# In[19]:


import torch.nn.functional as F
class Trainer():
    def __init__(self,net=None,optim=None,sched=None, train_loader=None, val_loader=None):
        self.net = net
        self.optim = optim
        self.sched = sched
        #self.l1 = l1
        #self.l2 = l2
        #self.loss_function = loss_function
        self.train_loader = train_loader
        self.val_loader = val_loader
        # self.sched = sched
        #net = net.float()
    def train(self,epochs):
        losses = []
        val = []
        #self.net = self.net.float()
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_steps = 0
            val_loss = 0.0
            val_steps = 0
            # print("yo")
            for data in self.train_loader:
                #print("sup")
                # print(type(data[0]))
                data_in = data[0].float()
                y = data[1].float()
#                 print(type(data_in))
#                 print(type(y))
                #y = y.type(torch.LongTensor)
#                 torch_input = torch.from_numpy(data_in)
#                 torch_label = torch.from_numpy(y)
                torch_input = data_in.type(torch.FloatTensor).to(DEVICE)
                torch_label = y.type(torch.FloatTensor).to(DEVICE)
                #print(torch_input)
#                 print(y.shape)
#                 print(y)
                self.optim.zero_grad()
                output = np.squeeze(self.net(torch_input))
                output = output.reshape((500, patch_width**2)) # added this for andrew outputting an image instead of a line
                #print(output)
#                 print(output.shape)
                #loss = self.loss_function(output, torch_label)
                loss = F.mse_loss(output, torch_label)
#                 l2_loss = self.l2(output,torch_label)
#                 loss = l1_loss + l2_loss
                loss.backward()
                
                #torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1)
                
                
                self.optim.step()
                epoch_loss += loss.item()
                epoch_steps += 1
            
            for vdata in self.val_loader:
                vdata_in = vdata[0].float()
                vy = vdata[1].float()
                vtorch_input = vdata_in.type(torch.FloatTensor).to(DEVICE)
                vtorch_label = vy.type(torch.FloatTensor).to(DEVICE)
                voutput = np.squeeze(self.net(vtorch_input))
                voutput = voutput.reshape((20, patch_width**2)) # added this for andrew outputting an image instead of a line
                vloss = F.mse_loss(voutput, vtorch_label)
#                 v2loss = self.l2(voutput, vtorch_label)
#                 vloss = v1loss + v2loss
                val_loss += vloss.item()
                val_steps += 1
            
            losses.append(epoch_loss / epoch_steps)
            val.append(val_loss / val_steps)
#             print("epoch [%d]: train loss %.6f" % (epoch+1, losses[-1]), end=' '),
#             print("epoch [%d]: val loss %.6f" % (epoch+1, val[-1]), end=' '),
            print("Epoch: %3d, training loss: %5.3e, validation loss: %5.3e" % (epoch+1, losses[-1], val[-1]), end='\r')
            
            self.sched.step(val[-1])

            
        return losses, val


# In[ ]:


import torch.optim as optim
import torch.nn as nn
import time

learning_rate = 8e-5
opt = optim.AdamW(model.parameters(), lr=learning_rate)
#opt = optim.Adamax(model.parameters(), lr=learning_rate)
#opt = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
#scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min')
#loss_function = nn.SmoothL1Loss()
# l1 = myLoss()
# l2 = nn.MSELoss()

#scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=40, gamma=0.5, last_epoch=-1, verbose=False)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.2, patience=10)

trainer = Trainer(net=model, optim=opt, sched=scheduler, train_loader=train_loader, val_loader=val_loader)
#net = net.float()

losses, val = trainer.train(epochs=200)



ML_file_out = output_name + '.pt'
print(ML_file_out)
torch.save(model.state_dict(), ML_file_out)


from datetime import date
thedate = str(date.today())
import matplotlib.pyplot as plt
plt.plot(losses[::], linewidth=2, linestyle='-', marker='.')
plt.plot(val[::], linewidth=2, linestyle='-', marker='.')
plt.semilogy()
#plt.legend()
plt.xlabel('Epoch #')
plt.ylabel('smooth L1 Loss')
plt.savefig(f'../plots/loss_curve_{output_name}_{thedate}.png', dpi=200)


