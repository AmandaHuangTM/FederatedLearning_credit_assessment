#!/usr/bin/env python
# coding: utf-8

# In[ ]:


pip install syft==0.2.9


# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import syft as sy
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler    
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


# In[ ]:


# hook PyTorch to PySyft i.e. add extra functionalities to support Federated Learning
# and other private AI tools
hook = sy.TorchHook(torch)
 
# we create two imaginary schools
node1 = sy.VirtualWorker(hook, id="node1")
node2 = sy.VirtualWorker(hook, id="node2")


# In[ ]:


df = pd.read_csv("data456.csv")
df.head()


# In[ ]:


df = df.dropna()
df = pd.get_dummies(df)
cols = df.columns.tolist()
cols.remove('Indicators')
cols.append('Indicators')
df = df[cols]
df.head() 


# In[ ]:


# df = df.astype(np.float64)
# df["Final_Status (Y/N)"].astype(int)
df.dtypes


# In[ ]:


# df['Class_att'] = df['Class_att'].astype('category')
# encode_map = {
#     'Abnormal': 1,
#     'Normal': 0
# }

# df['Class_att'].replace(encode_map, inplace=True)


# In[ ]:


sns.countplot(x = 'Indicators', data=df)


# In[ ]:


X = df.iloc[:, 0:-1]
y = df.iloc[:, -1]


# In[ ]:


print(X.shape, y.shape)


# In[ ]:


np.any(np.isnan(y))


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[ ]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


# In[ ]:


print(X_train.shape, X_test.shape)


# In[ ]:


EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 0.001


# In[ ]:


## train data
class trainData(Dataset):
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)


train_data = trainData(torch.FloatTensor(X_train), 
                       torch.FloatTensor(torch.Tensor(y_train.values)))
## test data    
class testData(Dataset):
    
    def __init__(self, X_data):
        self.X_data = X_data
        
    def __getitem__(self, index):
        return self.X_data[index]
        
    def __len__ (self):
        return len(self.X_data)
    

test_data = testData(torch.FloatTensor(X_test))


# In[ ]:


train_loader = sy.FederatedDataLoader(train_data.federate((node1, node2)), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=1) # may need increase later


# In[ ]:


class binaryClassification(nn.Module):
    def __init__(self):
        super(binaryClassification, self).__init__()
        # Number of input features is 12.
        self.layer_1 = nn.Linear(83, 160)
        self.layer_2 = nn.Linear(160, 40)
        self.layer_out = nn.Linear(40, 1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(160)
        self.batchnorm2 = nn.BatchNorm1d(40)
        
    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.layer_out(x)
        
        return x


# In[ ]:


# check to use GPU or not
use_cuda = True and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)
###############


# In[ ]:


model = binaryClassification()
model.to(device)
print(model)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)


# In[ ]:


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    
    return acc


# In[ ]:


model.train()
e_loss = []
e_acc = []
timelist = []
for e in range(1, EPOCHS+1):
    epoch_loss = 0
    epoch_acc = 0
    start_time = time.time()
    for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
        model = model.send(X_batch.location)

        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        
        y_pred = model(X_batch)
        
        loss = criterion(y_pred, y_batch.unsqueeze(1))
        acc = binary_acc(y_pred, y_batch.unsqueeze(1))
        
        loss.backward()
        optimizer.step()
        
        model.get()

        epoch_loss += loss.get()
        epoch_acc += acc.get()
    total_time = time.time() - start_time
    timelist.append(total_time)
    e_loss.append((epoch_loss/len(train_loader)).item()) # for plt
    e_acc.append((epoch_acc/len(train_loader)).item()) # for plt
    print(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Acc: {epoch_acc/len(train_loader):.3f} | Time: {total_time:.3f}')


# In[ ]:


plt.plot(e_loss)


# In[ ]:


plt.plot(e_acc)


# In[ ]:


plt.plot(timelist)


# In[ ]:



import statistics
statistics.mean(timelist)


# In[ ]:


y_pred_list = []
model.eval()
with torch.no_grad():
    for X_batch in test_loader:
        X_batch = X_batch.to(device)
        y_test_pred = model(X_batch)
        y_test_pred = torch.sigmoid(y_test_pred)
        y_pred_tag = torch.round(y_test_pred)
        y_pred_list.append(y_pred_tag.cpu().numpy())

y_pred_list = [a.squeeze().tolist() for a in y_pred_list]


# In[ ]:


confusion_matrix(y_test, y_pred_list)


# In[ ]:


print(classification_report(y_test, y_pred_list))


# In[ ]:




