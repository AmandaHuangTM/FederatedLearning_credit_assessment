# Vertical Federated Learning

import tensorflow as tf
import sys

# import the module from google drive
sys.path.insert(0,'path/')
import src
import torch
from torchvision import datasets, transforms
from torch import nn, optim
from torch.utils.data import TensorDataset
from torchvision.transforms import ToTensor
import syft as sy
hook = sy.TorchHook(torch)

from src.dataloader import VerticalDataLoader
from src.psi.util import Client, Server
from src.utils.split_data import add_ids
from imutils import paths
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from uuid import uuid4
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt


# load the data
df = pd.read_csv("path/dataset.csv")

# clean the data
# transform into binary columns
df = df.dropna()
df = pd.get_dummies(df)
cols = df.columns.tolist()
cols.remove('Indicators')
cols.append('Indicators')
df = df[cols]
df.head()

# split it into training and testing dataset
train, test = train_test_split( df , test_size=0.3, random_state=42)

# transform into tensor data
train_features = torch.as_tensor(train.drop("Indicators", axis=1).to_numpy()) # can only from numpy to tensor
train_target = torch.as_tensor(train["Indicators"].to_numpy())

# transform into tensor data
test_features = torch.as_tensor(train.drop("Indicators", axis=1).to_numpy()) # can only from numpy to tensor
test_target = torch.as_tensor(train["Indicators"].to_numpy())

# Implements a vertical dataset with TensorDataset
import torch
from torch.utils.data import Dataset
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


# self-designed VerticalTensorDataset
# to add id, set features and target
class VerticalTensorDataset(Dataset):
    def __init__(self, features, targets):
        super().__init__()

        self.data = features
        self.targets = targets
        self.size = features.shape[0]

        self.ids = np.array([uuid4() for _ in range(len(self))])

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        if self.data is None:
          data = None
        else:
          data = self.data[index]
        if self.targets is None:
          target = None
        else:
          target = self.targets[index]

        id = self.ids[index]

        return (*filter(lambda x: x is not None, (data, target, id)),)

    def get_ids(self):
        return [str(id) for id in self.ids]


# put training data into VerticalTensorDataset 
# to add ids and identify features and target column
train_dataset = VerticalTensorDataset(train_features, train_target)
# VerticalDataLoader is from PyVertical package
# it vetically patitions the data
train_dataloader = VerticalDataLoader(train_dataset, batch_size = 64)

# apply on testing data
test_dataset = VerticalTensorDataset(test_features, test_target)
test_dataloader = VerticalDataLoader(test_dataset, batch_size = 1)



# SplitNN
# assign model segments to each local nodes
# define how gradient is sent
class SplitNN:
    def __init__(self, models, optimizers):
        self.models = models
        self.optimizers = optimizers

        self.data = []
        self.remote_tensors = []

    def forward(self, x):
        data = []
        remote_tensors = []

        data.append(self.models[0](x))

        if data[-1].location == self.models[1].location:
            remote_tensors.append(data[-1].detach().requires_grad_())
        else:
            remote_tensors.append(
                data[-1].detach().move(self.models[1].location).requires_grad_()
            )

        i = 1
        while i < (len(models) - 1):
            data.append(self.models[i](remote_tensors[-1]))

            if data[-1].location == self.models[i + 1].location:
                remote_tensors.append(data[-1].detach().requires_grad_())
            else:
                remote_tensors.append(
                    data[-1].detach().move(self.models[i + 1].location).requires_grad_()
                )

            i += 1

        data.append(self.models[i](remote_tensors[-1]))

        self.data = data
        self.remote_tensors = remote_tensors

        return data[-1]
        
    def backward(self):
        for i in range(len(models) - 2, -1, -1):
            if self.remote_tensors[i].location == self.data[i].location:
                grads = self.remote_tensors[i].grad.copy()
            else:
                grads = self.remote_tensors[i].grad.copy().move(self.data[i].location)
    
            self.data[i].backward(grads)

    def zero_grads(self):
        for opt in self.optimizers:
            opt.zero_grad()

    def step(self):
        for opt in self.optimizers:
            opt.step()



torch.manual_seed(3)

# Define our model segments
input_size = 48
hidden_sizes = [32, 64]
output_size = 1


# Deep Learning model
# to apply it in SplitNN framework, it is divided into 2 parts
models = [
    nn.Sequential(
        nn.Linear(input_size, hidden_sizes[0]),
        nn.ReLU(),
        nn.BatchNorm1d(hidden_sizes[0]),
        nn.Linear(hidden_sizes[0], hidden_sizes[1]),
        nn.ReLU(),
        nn.BatchNorm1d(hidden_sizes[1]),
        nn.Dropout(p=0.1)
    ),
    nn.Sequential(nn.Linear(hidden_sizes[1], output_size))

]


# create optimisers for each segment and link to them
optimizers = [
    optim.SGD(model.parameters(), lr=0.001)
    for model in models
]

# create virtual workers as local node simulation
alice = sy.VirtualWorker(hook, id="alice")
bob = sy.VirtualWorker(hook, id="bob")

# send Model Segments to model locations
model_locations = [alice, bob]
for model, location in zip(models, model_locations):
    model.send(location)

# instantiate a SpliNN class with our distributed segments and their respective optimizers
splitNN = SplitNN(models, optimizers)


def train(x, target, splitNN):
    
    #1) Zero our grads
    splitNN.zero_grads()
    
    #2) Make a prediction
    pred = splitNN.forward(x)
    
    #3) Figure out how much we missed by
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(pred.reshape(-1), target.to(torch.float))
    
    #4) Backprop the loss on the end layer
    loss.backward()
    
    #5) Feed Gradients backward through the nework
    splitNN.backward()
    
    #6) Change the weights
    splitNN.step()
    
    return loss, pred


epochs = 100

e_loss = []
e_acc = []
timelist = []

for i in range(epochs):
    running_loss = 0
    correct_preds = 0
    total_preds = 0
    epoch_loss = 0
    epoch_acc = 0
    start_time = time.time()

    for (data, ids1), (labels, ids2) in train_dataloader:
        # Train a model
        data = data.send(models[0].location)
        data = data.view(data.shape[0], -1)
        labels = labels.send(models[-1].location)

        # Call model
        loss, preds = train(data.float(), labels, splitNN)
        
        # Convert predictions
        preds_ = preds.get().squeeze(1)
        labels_ = labels.get()
        preds_match = ((torch.sigmoid(preds_) > 0.5) == labels_)
        #print(non_exist)

        # Collect statistics
        running_loss += loss.get()
        correct_preds = correct_preds + int(preds_match.sum())
        total_preds = total_preds + int(preds_match.size()[0])

    total_time = time.time() - start_time
    timelist.append(total_time)
    e_loss.append((running_loss/len(train_dataloader)).item()) # for plt
    e_acc.append((100*correct_preds/total_preds)) # for plt

    print(f"Epoch {i} - Training loss: {running_loss/len(train_dataloader):.3f} - Accuracy: {100*correct_preds/total_preds:.3f}")


plt.plot(e_acc)

plt.plot(e_loss)

plt.plot(timelist)


# define test to apply trained model
def test(x, target, splitNN):
       
    #1) Make a prediction
    pred = splitNN.forward(x)
    
    #2) Figure out how much we missed by
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(pred.reshape(-1), target.to(torch.float))
        
    return loss, pred



y_pred_list = []
y_target_list = []
running_loss = 0
correct_preds = 0
total_preds = 0

# lock the models
models[0].eval()
models[1].eval()


# put testing data into the models
with torch.no_grad():
    for (data, ids1), (labels, ids2) in test_dataloader:
        # Train a model
        data = data.send(models[0].location)
        data = data.view(data.shape[0], -1)
        labels = labels.send(models[-1].location)

        # Call model
        loss, preds = test(data.float(), labels, splitNN)
        
        # Convert predictions
        preds_ = preds.get().squeeze(1)
        labels_ = labels.get()
        preds_match = ((torch.sigmoid(preds_) > 0.5) == labels_)
        y_pred_list.append(torch.sigmoid(preds_) > 0.5)
        y_target_list.append(labels_)

        # Collect statistics
        running_loss += loss.get()
        correct_preds = correct_preds + int(preds_match.sum())
        total_preds = total_preds + int(preds_match.size()[0])


    print(f" Testing loss: {running_loss/len(test_dataloader):.3f} - Accuracy: {100*correct_preds/total_preds:.3f}")

# see testing results
confusion_matrix(y_target_list, y_pred_list)

print(classification_report(y_target_list, y_pred_list))

