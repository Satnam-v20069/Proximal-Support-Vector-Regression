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

def get_H(x1,x2,degree,c):
    n = x1.shape[0]
    #print(x1.shape)
    m = Polynomial_Kernel(x1,x2,degree)+ np.eye(n)/c + np.dot(np.ones(n).reshape(n,1),np.ones(n).reshape(1,n))
    return m

def Polynomial_Kernel(x1,x2,degree):
    gram = (1+(np.dot(x1,x2.T))**degree)
    #print(gram.shape)
    return gram

def Predict(x_sample,x_test,degree,alpha,b):   
    matrix = Polynomial_Kernel(x_sample,x_test,degree)
    output = np.dot(matrix.T,alpha)+b
    return output

def Parameters(x_sample,y_sample,degree,c):
    n = y_sample.shape[0]
    n = y_sample.shape[0]
    H = get_H(x_sample,x_sample,degree,c)
    alpha = np.dot(np.linalg.pinv(H),y_sample)
    #print(alpha)
    bias = np.dot(np.ones(n).T,alpha)
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

x_train, y_train = multi_step_ahead(train, step =10)
x_test, y_test = multi_step_ahead(test, step = 10)


#Standard values of degree
poly_degree = []
for i in range(5):
    if i ==0:
        poly_degree.append(2**0)
    else:
        poly_degree.append(2**i)
  
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
for degree in poly_degree:
    for c_value in C: 
        alpha,bias = Parameters(x_train,y_train,degree,c_value)
        y_pred =  Predict(x_train,x_test,degree,alpha,bias)
        MSE,RMSE,R2,MAPE = error_function(y_pred, y_test)
        Error.append([degree, c_value, MSE, RMSE, R2, MAPE])
        

#choose value of c and e which have min MSE
min_mse = []
for i in Error:
    min_mse.append(i[2])
index = min_mse.index(np.min(min_mse))
print(min_mse)
print(index)

final_degree = Error[index][0]
final_c = Error[index][1]


print(f"Final degree is {final_degree}")
print(f"Final value of c is {final_c}")

alpha,bias = Parameters(x_train,y_train,final_degree,final_c)
y_pred =  Predict(x_train,x_test,final_degree,alpha,bias)



print(y_test[:10])
print(y_pred[:10])

result(y_pred, y_test)

f1 = plt.figure()
plt.plot(y_test, label='True Data')
plt.plot(y_pred,linestyle='dashed',label='Poly model Predict')
plt.title('PSVR (Poly)')
plt.xlabel("Test dataset")
plt.ylabel("Output Values")
plt.xlim(0,len(y_test))
plt.legend()
plt.show()
#f1.savefig(r'C:\Users\91931\OneDrive\Desktop\SVM dropbox\regression\TIME SERIES\mcg_rbf.pdf')