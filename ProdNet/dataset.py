import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

size = 2
num_of_matrecies = 46000


np.random.seed(42)
A_matrecies = np.random.uniform(-1, 1, size=(num_of_matrecies, size, size))

np.random.seed(43)
B_matrecies = np.random.uniform(-1, 1, size=(num_of_matrecies, size, size))

C_matrecies = np.zeros((num_of_matrecies, size, size))
for i in range (num_of_matrecies):
    C_matrecies[i] = A_matrecies[i] @ B_matrecies[i]
    

# flattening
A_matrecies = A_matrecies.reshape(46000,-1)
B_matrecies = B_matrecies.reshape(46000,-1)
C_matrecies = C_matrecies.reshape(46000,-1)

# Splitting (70% for train and 30 for test) and converting the dataframe to a tensor
A_train  = torch.tensor(A_matrecies[:32000])
B_train = torch.tensor(B_matrecies[:32000])
C_train = torch.tensor(C_matrecies[:32000])

A_test = torch.tensor(A_matrecies[32000:])
B_test = torch.tensor(B_matrecies[32000:])
C_test = torch.tensor(C_matrecies[32000:])
