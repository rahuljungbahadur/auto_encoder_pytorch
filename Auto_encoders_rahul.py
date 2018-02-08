# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

##Importing libraries
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable


movies = pd.read_csv('ml-1m/ml-1m/movies.dat', sep = '::',
                     header = None, engine = 'python', encoding = 'latin-1')

movies.head(10)

##Importing the users
users = pd.read_csv('ml-1m/ml-1m/users.dat', sep = '::',
                     header = None, engine = 'python', encoding = 'latin-1')
users.head()
##Importing the users
ratings = pd.read_csv('ml-1m/ml-1m/ratings.dat', sep = '::',
                     header = None, engine = 'python', encoding = 'latin-1')


##Creating the training set
train_set = pd.read_csv('ml-100k/ml-100k/u1.base', delimiter = '\t')
##Converting it to a numpy array
train_set = np.array(train_set, dtype = 'int')

#train_set.head()
##Creating the test set
test_set = pd.read_csv('ml-100k/ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')


ratings.head()
##Calculating the max numbers of users across the train and test set
max_users = int(max(max(train_set[:,0]), max(test_set[:,0])))
max_movies = int(max(max(train_set[:,1]), max(test_set[:,1])))



##Function for creating the req matrix
def matrix_create(data):
    data_new = []
    for user in range(1, max_users+1):
        user_movie = data[:,1][data[:,0] == user]
        user_rating = data[:,2][data[:,0] == user]
        ##creating a list of zeros to serve as default
        ratings = np.zeros(max_movies)
        ratings[user_movie - 1] = user_rating
        data_new.append(list(ratings))
    return(data_new)
    
train_set = matrix_create(train_set)
test_set = matrix_create(test_set)
        #user_ratings.append(user)
            
torch_train_set = torch.FloatTensor(train_set)
torch_test_set = torch.FloatTensor(test_set)
        
        
        
##Creating a class for nn module

class AutoEncoder(nn.Module):
    def __init__(self, ):
        super(AutoEncoder, self).__init__()
        self.layer1 = nn.Linear(max_movies,20)
        self.layer2 = nn.Linear(20,10)
        self.layer3 = nn.Linear(10,20)
        self.layer4 = nn.Linear(20,max_movies)
        self.activation = nn.Sigmoid()
##Creating the forward propagation neural net
    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        x = self.activation(self.layer3(x))
        x = self.layer4(x)
        return(x)

auto_encoder = AutoEncoder()
criterion = nn.MSELoss()   ###loss function criteria
optimizer = optim.RMSprop(auto_encoder.parameters(), lr = 0.01, weight_decay = 0.5)

##Training the auto Encoder

epochs = 100 ##number of Epochs
for i in range(1, epochs + 1):
    train_loss = 0
    s = 0.  ##no of users who rated the movie as non-zerowould be used for computation of RMSE which is a float
    for user in range(max_users):
        input = Variable(torch_train_set[user]).unsqueeze(0)
        targets = input.clone()
        if torch.sum(targets.data > 0) > 0:    ##To exclude obs where users gave no rating
            output = auto_encoder(input)
            targets.require_grad = False  ##gradient is not calculated on the target set
            output[targets == 0] = 0   ##set output to 0 manually where predicted was not zero
            loss = criterion(output, targets)
            mean_corrector = max_movies/float(torch.sum(targets.data > 0)+ 1e-10)
            loss.backward()
            train_loss += np.sqrt(loss.data[0] * mean_corrector)
            s += 1.
            optimizer.step()
    print('epoch :' + str(i) + ' loss :' + str(train_loss/s))
            

torch.save(auto_encoder, 'auto_Enc2.pt')
##The train loss was approx : 0.95 on 100 epochs
##loading the model
saved_model = torch.load('auto_Enc2.pt')


###testing on the test file
test_loss = 0
s = 0.  ##no of users who rated the movie as non-zerowould be used for computation of RMSE which is a float
for user in range(max_users):
    input = Variable(torch_train_set[user]).unsqueeze(0)
    targets = Variable(torch_test_set[user])
    if torch.sum(targets.data > 0) > 0:    ##To exclude obs where users gave no rating
        output = saved_model(input)
        targets.require_grad = False  ##gradient is not calculated on the target set
        output[targets == 0] = 0   ##set output to 0 manually where predicted was not zero
        loss = criterion(output, targets)
        mean_corrector = max_movies/float(torch.sum(targets.data > 0)+ 1e-10)
        test_loss += np.sqrt(loss.data[0] * mean_corrector)
        s += 1.
print( 'test_loss :' + str(test_loss/s))
            
##test_loss :0.9900578615791735. Hence, our model would be off by 1 rating on average.












    

        
        
        
        
        
        
        
        
        
        
        
    
    







