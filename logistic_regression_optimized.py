from __future__ import division
import numpy as np
import matplotlib.pyplot as plt 
import scipy.optimize as op



# Hypothesis

def Hypothesis(X,theta):
    a=np.dot(X,theta)
    sigmoid=1/(1+np.exp(-a))
    return sigmoid
    

# cost function
def costFunction(theta,X,Y):
    hypothesis=Hypothesis(X,theta)
    num_trainning=Y.size
    cost_result = -(1/num_trainning)*(np.dot(Y.T,np.log(hypothesis))+np.dot((1-Y).T,np.log(1-hypothesis)))
    return cost_result

def gradient(theta,X,Y):
    
    theta=np.matrix(theta)
    num_trainning=Y.size
    #theta=np.reshape(theta,(X.shape[1],1))
    
    gradient=np.dot(X.T,np.subtract(Hypothesis(X,theta.T),Y))/num_trainning                                                                                                 
    return gradient
    
    
def predict(theta, X):  
    probability = Hypothesis(X,theta)
    return [1 if x >= 0.5 else 0 for x in probability]


#----------------------------------------- Code Start here ------------------------------------------------
# load data from file. here data consist of two features and the label.
data=np.genfromtxt("/directory/to/dataset",
                  delimiter=",") 
          

# this for controlling the packing process of data to get x and y.
features_numbers=data.shape[1]-1
label=data.shape[1]-1 # just for clarity

# getting x and y , setting the algorithm values.          
x=data[:,0:features_numbers]
y=data[:,label]

y=np.reshape(data[:,label],(y.shape[0],1))  # just for reshape the y to be (n,1) instead of (n,).
m=y.size  # number of trainning set.
theta=np.zeros((x.shape[1],1))  

 #normalize x (features)
mean_vector=np.zeros((x.shape[1],1))
std_vector=np.zeros((x.shape[1],1))

for i in range(features_numbers):
    
    mean_vector[i,0]=np.mean(x[:,i])
    mean_vector[i,0]=np.mean(x[:,i])

    std_vector[i,0]=np.std(x[:,i])
    std_vector[i,0]=np.std(x[:,i])

x=np.divide(np.subtract(x,mean_vector.T),std_vector.T)

# --------------------------------------- Cost function --------------------------------------------------
cost=costFunction(theta,x,y)
# you can print the cost value befire gradient descent works
print cost

result=op.fmin_tnc(func=costFunction,x0=theta,fprime=gradient,args=(x,y))
new_theta=np.matrix(result[0])


# print gradient descent after working
print costFunction(new_theta.T,x,y)
 
# calculating model accuracy
predictions = predict(new_theta.T, x)  
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]  
accuracy = (sum(map(int, correct)) % len(correct))  
print 'accuracy = {0}%'.format(accuracy)  