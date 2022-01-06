# -*- coding: utf-8 -*-
"""

@author: Satnam Singh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error

def get_H(x1,x2,sigma,c):
    n = x1.shape[0]
    m = Gaussian_Kernel(x1,x2,sigma)+ np.eye(n)/c + np.dot(np.ones(n).reshape(n,1),np.ones(n).reshape(1,n))
    return m

def Gaussian_Kernel(x1,x2,sigma):
    X1 = sum((x1**2).T).reshape(x1.shape[0],1)
    X2 = sum((x2**2).T).reshape(x2.shape[0],1)          
    norm = X2 + X1.T - 2*np.dot(x2,x1.T) 
    #print(norm)
    return np.exp(-norm/sigma**2)
   

def Predict(x_sample,x_test,sigma,alpha,b):   
    matrix = Gaussian_Kernel(x_sample,x_test,sigma)
    output = np.dot(matrix,alpha)+b
    return output

def Parameters(x_sample,y_sample,sigma,c):
    n = y_sample.shape[0]
    H = get_H(x_sample,x_sample,sigma,c)
    alpha = np.dot(np.linalg.pinv(H),y_sample)
    #print(alpha)
    bias = np.dot(np.ones(n).T,alpha)
    #print(bias)
    return alpha, bias
    #print(alpha)
    #print(bias)
    return alpha, bias


#error function
def error_function(y_pred,y_test):
    MSE  = mean_squared_error(y_pred, y_test)
    RMSE = np.sqrt(MSE)
    R2   = r2_score(y_pred,y_test)
    MAPE = mean_absolute_percentage_error(y_test,y_pred)
    
    return MSE,RMSE,R2,MAPE

#result of y_pred and y_test
def result(y_pred, y_test):
    
    MSE,RMSE,R2,MAPE = error_function(y_pred,y_test)
    print(f"mean square error is {MSE}")
    print(f"root mean square error is {np.sqrt(RMSE)}")
    print(f"r2_score is {R2}")
    print(f"Mean absolute percetage error is {MAPE}")
    
def time_series_cv(train,step=5):
    index = []
    length = len(train)
    for i in range(length-step):
        p = np.arange(len(train))
        folds = []
        r = p[i:i+step]
        m = p[step+i]
        folds.append(r)
        folds.append(m)
        index.append(folds)
    return index


def test_train_split(data,train_ratio=0.7):
    n = len(data)
    train_index = int(n*train_ratio)
    train = data[:train_index]
    test = data[train_index:]
    return train, test

def multi_step_ahead(train, step=5):
    x_train = []
    y_train = []
    for train_index, test_index in time_series_cv(train,step):
        x_train_set = train[train_index]
        y_train_set = train[test_index]
        merged_train = []
        for l in x_train_set:
            for i in l:
                merged_train.append(i)
        x_train.append(merged_train)
        y_train.append(y_train_set)
    return np.array(x_train), np.array(y_train)
    

#import the dataset
dataset = pd.read_csv(r"C:\Users\91931\OneDrive\Desktop\SVM dropbox\regression data\TIME SERIES\lorenz1.csv")
#print(dataset)
x = (dataset.iloc[:,-1:].values)
train, test = test_train_split(x)

x_train, y_train = multi_step_ahead(train, step =5)
x_test, y_test = multi_step_ahead(test, step = 5)


#Standard values of sigma
sigma = []
for i in range(10):
    if i ==0:
        sigma.append(2**0)
    else:
        sigma.append(2**i)
        sigma.append(2**(-i))
  
#standard values of C
C = []
for i in range(5):
    if i ==0:
        C.append(2**0)
    else:
        C.append(2**(i))
        C.append(2**(-i))

#Error in time series cross validation
Error = []
for sigma_rbf in sigma:
    for c_value in C: 
        alpha,bias = Parameters(x_train,y_train,sigma_rbf,c_value)
        y_pred =  Predict(x_train,x_test,sigma_rbf,alpha,bias)
        MSE,RMSE,R2,MAPE = error_function(y_pred, y_test)
        Error.append([sigma_rbf, c_value, MSE, RMSE, R2, MAPE])
        print(Error)
        print("\n")

#choose value of c and e which have min MSE
min_mse = []
for i in Error:
    min_mse.append(i[2])
index = min_mse.index(np.min(min_mse))
print(min_mse)
print(index)

final_sigma = Error[index][0]
final_c = Error[index][1]


print(f"Final sigma is {final_sigma}")
print(f"Final value of c is {final_c}")

alpha,bias = Parameters(x_train,y_train,final_sigma,final_c)
y_pred =  Predict(x_train,x_test,final_sigma,alpha,bias)



print(y_test[:10])
print(y_pred[:10])

result(y_pred, y_test)

f1 = plt.figure()
plt.plot(y_test, label='True Data')
plt.plot(y_pred,linestyle='dashed',label='RBF model Predict')
plt.title('PSVR (RBF)')
plt.xlabel("Test dataset")
plt.ylabel("Output Values")
plt.xlim(0,len(y_test))
plt.legend()
plt.show()
#f1.savefig(r'C:\Users\91931\OneDrive\Desktop\SVM dropbox\regression\TIME SERIES\mcg_rbf.pdf')